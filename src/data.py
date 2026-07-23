from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import transforms
from torchvision.datasets import EuroSAT
import torchvision.transforms.functional as TF

import random

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
ds = EuroSAT(root=DATA_ROOT, download=True)
print(len(ds), ds.classes)
# 'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake'

CLASS_NAMES = [ # ALPHABETICAL ORDER SAME AS TORCHVISION DATASET
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
]

# EUROSAT_MEAN = (0.3444, 0.3803, 0.4078)
# EUROSAT_STD = (0.2037, 0.1366, 0.1148)

EUROSAT_MEAN = (0.3436920642852783, 0.37974652647972107, 0.40759754180908203)
EUROSAT_STD = (0.20261552929878235, 0.13720497488975525, 0.11578831076622009)

class RandomRotation90:
    def __call__(self, img):
        angle = random.choice([0, 90, 180, 270])
        return TF.rotate(img, angle)

def build_transforms(augment: bool = False):
    
    steps = []

    if augment:
        steps += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            RandomRotation90(),
        ]

    steps += [
        transforms.ToTensor(), # PIL (H,W,C) uint8 [0,255] -> tensor (C,H,W) float [0,1]
        transforms.Normalize(mean=EUROSAT_MEAN, std=EUROSAT_STD),
    ]

    return transforms.Compose(steps)

class _TransformedSubset(torch.utils.data.Dataset):
    
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, index):
        image, label = self.subset[index]   
        if self.transform is not None:
            image = self.transform(image)
        return image, label

def get_datasets(root=DATA_ROOT, seed=42, download=True, augment_train=False):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    base = EuroSAT(root=str(root), download=download, transform=None)

    n_total = len(base)
    n_val = int(0.15 * n_total)
    n_test = int(0.15 * n_total)
    n_train = n_total - n_val - n_test

    generator = torch.Generator().manual_seed(seed)
    train_sub, val_sub, test_sub = random_split(
        base, [n_train, n_val, n_test], generator=generator
    )

    train_ds = _TransformedSubset(train_sub, build_transforms(augment=augment_train))
    val_ds   = _TransformedSubset(val_sub,   build_transforms(augment=False))
    test_ds  = _TransformedSubset(test_sub,  build_transforms(augment=False))
    return train_ds, val_ds, test_ds


def get_dataloaders(root=DATA_ROOT, seed=42, batch_size=64,
                    num_workers=0, download=True, augment_train=False):
    train_ds, val_ds, test_ds = get_datasets(
        root=root, seed=seed, download=download, augment_train=augment_train
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader

def compute_channel_stats(root=DATA_ROOT, seed=42):
    root = Path(root)
    base = EuroSAT(root=str(root), download=True, transform=transforms.ToTensor())

    n_total = len(base)
    n_val = int(0.15 * n_total)
    n_test = int(0.15 * n_total)
    n_train = n_total - n_val - n_test

    generator = torch.Generator().manual_seed(seed)
    train_sub, _, _ = random_split(base, [n_train, n_val, n_test], generator=generator)

    n_pixels = 0
    channel_sum = torch.zeros(3)
    channel_sum_sq = torch.zeros(3)
    for image, _ in train_sub:                      
        n_pixels += image.shape[1] * image.shape[2]
        channel_sum += image.sum(dim=(1, 2))        
        channel_sum_sq += (image ** 2).sum(dim=(1, 2))

    mean = channel_sum / n_pixels
    std = (channel_sum_sq / n_pixels - mean ** 2).sqrt()
    return tuple(mean.tolist()), tuple(std.tolist())

mean, std = compute_channel_stats()
print("mean:", mean)
print("std: ", std)

if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders(augment_train=True)

    print("Classes:", CLASS_NAMES)
    print("Train:", len(train_loader.dataset))   # expect 18900
    print("Val:  ", len(val_loader.dataset))     # expect 4050
    print("Test: ", len(test_loader.dataset))    # expect 4050

    images, labels = next(iter(train_loader))
    print("Batch images:", tuple(images.shape))  # expect (64, 3, 64, 64)
    print("Batch labels:", tuple(labels.shape))  # expect (64,)
    print("Pixel range: [%.3f, %.3f]" % (images.min(), images.max()))  # normalised, so roughly [-2, 2]