from __future__ import annotations

from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.transforms import v2
import PIL.Image as Image
import numpy as np
import random

NUM_CLASSES = 9
CLASS_NAMES = [
    "Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia",
    "Rubber Vine", "Siam Weed", "Snake Weed", "Negatives",
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_split(labels_dir: str | Path, fold: int = 0):
    labels_dir = Path(labels_dir)
    train_df = pd.read_csv(labels_dir / f"train_subset{fold}.csv")
    val_df = pd.read_csv(labels_dir / f"val_subset{fold}.csv")
    test_df = pd.read_csv(labels_dir / f"test_subset{fold}.csv")
    return train_df, val_df, test_df


def check_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                images_dir: str | Path) -> dict:
    frames = dict(train=train_df, val=val_df, test=test_df)
    sets = {}
    for name, df in frames.items():
        if df.Filename.duplicated().any() or not df.Label.isin(range(9)).all():
            raise ValueError(f"Invalid filenames/labels in {name}")
        sets[name] = set(df.Filename)
        missing = [x for x in df.Filename if not (Path(images_dir) / x).is_file()]
        if missing:
            raise FileNotFoundError(f"{name}: {len(missing)} missing images, e.g. {missing[:3]}")
    if any(sets[a] & sets[b] for a,b in [('train','val'),('train','test'),('val','test')]):
        raise ValueError("Overlapping splits")
    if len(set.union(*sets.values())) != 17509:
        raise ValueError("Expected union of exactly 17509 images")
    for name, ratio in [('train', .6), ('val', .2), ('test', .2)]:
        if abs(len(frames[name])/17509-ratio) > .01:
            raise ValueError(f"Unexpected split ratio: {name}")
    return {"n": {k:len(v) for k,v in frames.items()},
            "per_class": {k:v.Label.value_counts().reindex(range(9), fill_value=0).to_dict()
                          for k,v in frames.items()}}


def build_transforms(train: bool, img_size: int = 224, aug: str = "basic"):
    if aug not in {"basic", "color", "trivial", "randaug"}:
        raise ValueError(f"Unknown augmentation: {aug}")
    if not train:
        return v2.Compose([
            v2.Resize(round(img_size * 256 / 224)),
            v2.CenterCrop(img_size),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        
    transforms_list = [v2.RandomResizedCrop(img_size), v2.RandomHorizontalFlip()]
    
    if aug == "color":
        transforms_list.append(v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2))
    elif aug == "trivial":
        transforms_list.append(v2.TrivialAugmentWide())
    elif aug == "randaug":
        transforms_list.append(v2.RandAugment())
        
    transforms_list.extend([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    return v2.Compose(transforms_list)


class DeepWeedsDataset(Dataset):
    def __init__(self, df: pd.DataFrame, images_dir: str | Path, transform=None):
        self.df = df
        self.images_dir = Path(images_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        filename = row['Filename']
        label = int(row['Label'])
        
        img_path = self.images_dir / filename
        with Image.open(img_path) as source:
            img = source.convert("RGB")
        
        if self.transform:
            img = self.transform(img)
            
        return img, label, filename


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loader(df: pd.DataFrame, images_dir: str | Path, transform, batch_size: int,
                train: bool, sampler: str | None = None, num_workers: int = 2):
    
    dataset = DeepWeedsDataset(df, images_dir, transform)
    
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
        "worker_init_fn": seed_worker
    }
    
    if train:
        if sampler == "balanced":
            class_counts = df['Label'].value_counts().reindex(range(9), fill_value=0).values
            weights = 1.0 / np.maximum(class_counts, 1)
            sample_weights = weights[df['Label'].values]
            weighted_sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )
            loader_kwargs["sampler"] = weighted_sampler
            loader_kwargs["shuffle"] = False
        else:
            loader_kwargs["shuffle"] = True
        loader_kwargs["drop_last"] = True
    else:
        loader_kwargs["shuffle"] = False
        loader_kwargs["drop_last"] = False
        
    return DataLoader(dataset, **loader_kwargs)
