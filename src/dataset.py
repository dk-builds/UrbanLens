import rasterio
import numpy as np
from pathlib import Path
import random
import yaml
import torch
from torch.utils.data import DataLoader, Dataset


CONFIG_PATH = "configs/config.yaml"

EUROSAT_BAND_INDICES = [1,2,3,7,11]

def load_config(config):
    with open(config) as f:
        return yaml.safe_load(f)

CONFIG = load_config(CONFIG_PATH)
EUROSAT_MAPPING = CONFIG["eurosat_mapping"]
SEED = CONFIG["project"]["seed"]
BAND_NAMES = CONFIG["data"]["eurosat_bands"]
OUR_CLASSES = CONFIG["classes"]["names"]
EUROSAT_BAND_INDICES = CONFIG["data"]["eurosat_band_indices"]
CLS_TO_IDS = {name:id for id, name in enumerate(OUR_CLASSES)}
LABEL_MAPPING = {eurosat_cls:CLS_TO_IDS[our_cls]for eurosat_cls, our_cls in EUROSAT_MAPPING.items()}


#aggregate (path,label) of data
def build_indices(eurosat_dir):
    
    mapping = EUROSAT_MAPPING
    eurosat_path = Path(eurosat_dir)
    
    index = []
    for cls_dir in eurosat_path.iterdir():
        if not cls_dir.is_dir():
            continue
        folder_name = cls_dir.name
        if folder_name not in mapping:
            continue
        tifs = cls_dir.glob("*.tif")
        for p in tifs:
            index.append((str(p), LABEL_MAPPING[folder_name]))
        
    return index

# Stats for standardisation
def compute_dataset_stats(index, max_samples = 5000):
    
    sample = random.sample(index, min(len(index),max_samples))
    mean = np.zeros(len(EUROSAT_BAND_INDICES), dtype = np.float64)
    n = np.zeros(len(EUROSAT_BAND_INDICES), dtype = np.float64)
    M2 = np.zeros(len(EUROSAT_BAND_INDICES), dtype = np.float64)
    stats = {}
    for path,_ in sample:
        with rasterio.open(path) as src:
            arr = src.read([b+1 for b in EUROSAT_BAND_INDICES]).astype (np.float64)
        for i in range(len(EUROSAT_BAND_INDICES)):
            pixels = arr[i].ravel()
            count = len(pixels)
            delta = pixels - mean[i]
            n[i]+= count
            mean[i] += delta.sum()/n[i]
            delta2 = pixels - mean[i]
            M2[i] += (delta*delta2).sum()
    std = np.sqrt(M2/np.maximum(n-1,1))
    for i, band_name in enumerate(BAND_NAMES):
        stats[band_name] = (mean[i], std[i])
    return stats

# Data Augmentation
def augment_data(tensor):
    if random.random() > 0.5:
        tensor = torch.flip(tensor, dims = [2])
    if random.random() > 0.5:
        tensor = torch.flip(tensor, dims = [1])
    k = random.randint(0,3)
    if k > 0:
        tensor = torch.rot90(tensor, k, dims = [1,2])
    return tensor

#dataset class
class EurosatDataSet (Dataset):
    def __init__(self, file_index, band_indices = EUROSAT_BAND_INDICES,augment=False, band_stats=None):
        self.index = file_index
        self.band_indices = band_indices
        self.augment = augment
        self.means = torch.tensor([band_stats[b][0] for b in BAND_NAMES], dtype=torch.float32).view(-1,1,1)
        self.std = torch.tensor([band_stats[b][1]for b in BAND_NAMES], dtype=torch.float32).view(-1,1,1)
    
    def __len__(self,):
        return len(self.index)
    
    def __getitem__(self, idx):
        path, lbl = self.index[idx]
        with rasterio.open(path) as src:
            arr = src.read ([ b+1 for b in self.band_indices]).astype(np.float32)
        
        image = torch.from_numpy(arr)
        image = torch.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        image = (image - self.means)/(self.std + 1e-8)

        if self.augment:
            image = augment_data(image)

        return image, torch.tensor(lbl, dtype = torch.long)
    
    #class weights to handle imbalanced classes
    def class_weights(self):
        labels = torch.tensor([lbl for _, lbl in self.index], dtype=torch.long)
        counts = torch.bincount(labels, minlength=len(OUR_CLASSES)).float()
        counts = torch.clamp(counts, min=1.0)
        weights = 1.0 / counts
        return weights / weights.sum() * len(OUR_CLASSES)
    

def build_dataloaders(band_stats=None, num_workers=4):

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    full_index = build_indices(CONFIG["data"]["eurosat_dir"])

    total = len(full_index)
    test_size = int(total*CONFIG["training"]["test_split"])
    val_size = int(total*CONFIG["training"]["val_split"])
    train_size = total - (test_size + val_size)

    shuffled = full_index.copy()
    random.shuffle(shuffled)

    train_idx = shuffled[:train_size]
    val_idx = shuffled[train_size:train_size+val_size]
    test_idx = shuffled[train_size+val_size:]

    if band_stats is None:
        band_stats = compute_dataset_stats(train_idx)

    train_ds = EurosatDataSet(train_idx,band_stats = band_stats,augment = True)
    test_ds = EurosatDataSet(test_idx,band_stats = band_stats,augment = False)
    val_ds = EurosatDataSet(val_idx,band_stats = band_stats,augment = False)

    weights = train_ds.class_weights()

    batch_size = CONFIG["training"]["batch_size"]
    use_persistent = num_workers> 0

    train_loader = DataLoader(train_ds, batch_size = batch_size, shuffle=True, 
                              drop_last=True, num_workers=num_workers, pin_memory=True,
                              persistent_workers=use_persistent)
    val_loader = DataLoader(val_ds, batch_size = batch_size, shuffle=False, 
                            num_workers=num_workers, pin_memory=True,
                            persistent_workers=use_persistent)
    test_loader = DataLoader(test_ds, batch_size = batch_size, shuffle=False, 
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=use_persistent)
    return train_loader, val_loader, test_loader, weights

if __name__ == "__main__":
    config = load_config("configs/config.yaml")
    file_index = build_indices(config["data"]["eurosat_dir"])
    assert file_index, "no patches found"

    band_stats = compute_dataset_stats(file_index, max_samples=500)
    ds = EurosatDataSet(file_index, band_stats=band_stats, augment=False)
    img, lbl = ds[0]

    assert img.shape == torch.Size([5, 64, 64])
    assert img.dtype == torch.float32
    assert not torch.isnan(img).any()

    print(f"OK — shape={img.shape} label={lbl.item()} ({OUR_CLASSES[lbl.item()]})")
    print(f"   min={img.min():.3f}  max={img.max():.3f}  mean={img.mean():.3f}  std={img.std():.3f}")