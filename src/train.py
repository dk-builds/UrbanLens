"""
Single end-to-end training run — no pretraining/fine-tuning stage, per
project instructions section 7 (every patch already carries real per-pixel
supervision). Run twice under the same MLflow experiment: once at the
default in_channels=8, once with --in-channels 5 for the section 9 ablation.

Usage:
    python src/train.py
    python src/train.py --in-channels 5 --run-name unet5ch_ablation
"""

import argparse
import json
import time
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from dataset import UrbanLensDataset, class_pixel_counts, load_config, DATA_PROCESSED, REPO_ROOT
from model import UNet

CHECKPOINT_DIR = REPO_ROOT / "outputs" / "checkpoints"
METRICS_DIR = REPO_ROOT / "outputs" / "metrics"
MLFLOW_EXPERIMENT = "urbanlens_segmentation"


def load_patch_index():
    with open(DATA_PROCESSED / "patch_index.json") as f:
        return json.load(f)


def compute_class_weights(counts, smoothing_c=1.02):
    """Log-smoothed inverse frequency (ENet-style: 1/ln(c + p_c)) rather than
    raw inverse frequency, which overreacts to rare classes and destabilizes
    training — see instructions section 7. Normalized to mean 1 so the loss
    magnitude stays comparable to unweighted CE."""
    freq = counts / counts.sum()
    weights = 1.0 / np.log(smoothing_c + freq)
    return weights / weights.mean()


def confusion_matrix(preds, labels, num_classes, ignore_index):
    valid = labels != ignore_index
    p, l = preds[valid], labels[valid]
    idx = l * num_classes + p
    cm = torch.bincount(idx, minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return cm


def iou_from_confusion(cm):
    """Per-class IoU; NaN for classes absent from both preds and labels
    (e.g. Bare Soil in some val splits — see state.md) rather than a
    misleading 0."""
    cm = cm.astype("float64")
    tp = np.diag(cm)
    union = cm.sum(0) + cm.sum(1) - tp
    with np.errstate(invalid="ignore"):
        iou = np.where(union > 0, tp / union, np.nan)
    return iou


@torch.no_grad()
def evaluate(model, loader, device, num_classes, ignore_index):
    model.eval()
    cm_total = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        preds = model(x).argmax(dim=1)
        cm_total += confusion_matrix(preds.flatten(), y.flatten(), num_classes, ignore_index).cpu()
    iou = iou_from_confusion(cm_total.numpy())
    return iou


def run(args):
    config = load_config()
    class_names = config["classes"]["names"]
    ignore_index = config["classes"]["ignore_index"]
    num_classes = len(class_names)

    index = load_patch_index()
    print(f"train={len(index['train'])} val={len(index['val'])} patches")

    train_ds = UrbanLensDataset(index["train"], config)
    val_ds = UrbanLensDataset(index["val"], config)

    class ChannelSubset(torch.utils.data.Dataset):
        """For the 5-channel ablation: keep the 5 normalized raw bands,
        drop the 3 spectral-index channels the 8-channel model gets."""
        def __init__(self, base, n_channels):
            self.base, self.n_channels = base, n_channels

        def __len__(self):
            return len(self.base)

        def __getitem__(self, i):
            x, y = self.base[i]
            return x[: self.n_channels], y

    if args.in_channels == 5:
        train_ds = ChannelSubset(train_ds, 5)
        val_ds = ChannelSubset(val_ds, 5)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    print("computing class weights from full training index...")
    counts, ignored, _ = class_pixel_counts(index["train"], config)
    weights = compute_class_weights(counts)
    print("class weights:", dict(zip(class_names, np.round(weights, 3))))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet(in_channels=args.in_channels, num_classes=num_classes).to(device)
    loss_fn = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device),
        ignore_index=ignore_index,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    run_name = args.run_name or f"unet{args.in_channels}ch"
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CHECKPOINT_DIR / f"{run_name}_best.pt"
    last_ckpt_path = CHECKPOINT_DIR / f"{run_name}_last.pt"

    start_epoch, best_miou, epochs_no_improve, resumed_run_id = 0, -1.0, 0, None
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        if ckpt.get("in_channels", args.in_channels) != args.in_channels:
            raise ValueError(f"--resume checkpoint was trained with in_channels="
                              f"{ckpt.get('in_channels')}, but --in-channels={args.in_channels}")
        model.load_state_dict(ckpt["model_state"])
        if "optimizer_state" in ckpt:
            # Full-state checkpoint (saved every epoch since resume support was
            # added) — exact resume: optimizer/scheduler position, patience
            # counter, and MLflow run all carry over.
            optimizer.load_state_dict(ckpt["optimizer_state"])
            scheduler.load_state_dict(ckpt["scheduler_state"])
            best_miou = ckpt["best_miou"]
            epochs_no_improve = ckpt["epochs_no_improve"]
            resumed_run_id = ckpt.get("run_id")
            start_epoch = ckpt["epoch"] + 1
            print(f"resumed full state from {args.resume}: starting epoch {start_epoch+1}, "
                  f"best_miou={best_miou:.4f}, epochs_no_improve={epochs_no_improve}")
        else:
            # Older best-only checkpoint (model weights only, from before resume
            # support existed) — warm start: reload weights, fast-forward the
            # cosine schedule to match, but optimizer momentum/variance and
            # patience count are lost (reset).
            best_miou = ckpt.get("val_miou", -1.0)
            start_epoch = ckpt.get("epoch", -1) + 1
            for _ in range(start_epoch):
                scheduler.step()
            print(f"WARNING: {args.resume} has no optimizer/scheduler state (pre-resume-support "
                  f"checkpoint) — warm-starting model weights only from epoch {start_epoch}. "
                  f"Optimizer momentum and early-stopping patience are reset; best_miou seeded to "
                  f"{best_miou:.4f} so it won't be spuriously overwritten.")

    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    run_kwargs = {"run_id": resumed_run_id} if resumed_run_id else {"run_name": run_name}
    with mlflow.start_run(**run_kwargs):
        if not resumed_run_id:
            mlflow.log_params({
                "in_channels": args.in_channels, "num_classes": num_classes,
                "base_features": 64, "epochs": args.epochs, "batch_size": args.batch_size,
                "lr": args.lr, "weight_decay": args.weight_decay, "patience": args.patience,
                "train_patches": len(index["train"]), "val_patches": len(index["val"]),
                **{f"class_weight_{n}": round(float(w), 4) for n, w in zip(class_names, weights)},
            })
        run_id = mlflow.active_run().info.run_id

        for epoch in range(start_epoch, args.epochs):
            model.train()
            t0 = time.time()
            running_loss = 0.0
            for x, y in train_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                optimizer.zero_grad()
                loss = loss_fn(model(x), y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * x.size(0)
            scheduler.step()
            train_loss = running_loss / len(train_ds)

            iou = evaluate(model, val_loader, device, num_classes, ignore_index)
            val_miou = float(np.nanmean(iou))

            metrics = {"train_loss": train_loss, "val_miou": val_miou,
                       "lr": scheduler.get_last_lr()[0]}
            for name, v in zip(class_names, iou):
                metrics[f"val_iou_{name.lower().replace(' ', '_')}"] = float(v) if not np.isnan(v) else -1.0
            mlflow.log_metrics(metrics, step=epoch)

            dt = time.time() - t0
            print(f"epoch {epoch+1}/{args.epochs}  loss={train_loss:.4f}  "
                  f"val_mIoU={val_miou:.4f}  ({dt:.0f}s)")

            if val_miou > best_miou:
                best_miou, epochs_no_improve = val_miou, 0
                torch.save({"model_state": model.state_dict(), "in_channels": args.in_channels,
                            "epoch": epoch, "val_miou": val_miou}, ckpt_path)
            else:
                epochs_no_improve += 1

            # Full-state checkpoint, overwritten every epoch (not just on new
            # best) so a killed/crashed run can resume via --resume without
            # losing optimizer/scheduler/patience state.
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "in_channels": args.in_channels,
                "epoch": epoch,
                "best_miou": best_miou,
                "epochs_no_improve": epochs_no_improve,
                "run_id": run_id,
            }, last_ckpt_path)

            if epochs_no_improve >= args.patience:
                print(f"early stopping at epoch {epoch+1} (no improvement for {args.patience} epochs)")
                break

        mlflow.log_metric("best_val_miou", best_miou)
        mlflow.log_artifact(str(ckpt_path))

    print(f"done. best val mIoU={best_miou:.4f}, checkpoint -> {ckpt_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-channels", type=int, default=8, choices=[5, 8])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--run-name", type=str, default=None)
    ap.add_argument("--resume", type=str, default=None,
                     help="path to a *_last.pt checkpoint to resume training from")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
