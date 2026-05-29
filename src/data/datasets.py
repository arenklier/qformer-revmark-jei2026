"""
Unified DataLoader for QFormer-RevMark.

Datasets are stored under data/<name>/ and discovered via glob.
Each dataset wraps the same ImageFolder-style logic.
"""
from pathlib import Path
from typing import Optional, Sequence

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T


_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


class ImageFolderFlat(Dataset):
    """Loads all images from a directory tree, ignoring labels."""

    def __init__(
        self,
        root: str,
        image_size: int = 256,
        crop: str = "center",
        normalize: bool = False,
    ):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root}")

        self.files = sorted(
            f for f in self.root.rglob("*")
            if f.is_file() and f.suffix.lower() in _EXTS and not f.name.startswith(".") and "__MACOSX" not in str(f)
        )
        if not self.files:
            raise RuntimeError(f"No images found under {self.root}")

        tfms = []
        if crop == "random":
            tfms.append(T.RandomCrop(image_size, pad_if_needed=True))
        elif crop == "center":
            tfms.append(T.Resize(image_size + 32))
            tfms.append(T.CenterCrop(image_size))
        elif crop == "none":
            pass
        else:
            raise ValueError(f"unknown crop mode: {crop}")
        tfms.append(T.ToTensor())  # -> (3, H, W) in [0,1]
        if normalize:
            tfms.append(T.Normalize([0.5] * 3, [0.5] * 3))
        self.transform = T.Compose(tfms)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        return self.transform(img), str(self.files[idx])


def build_loader(
    name: str,
    data_root: str,
    image_size: int = 256,
    batch_size: int = 16,
    num_workers: int = 4,
    split: str = "train",
    crop: Optional[str] = None,
    shuffle: Optional[bool] = None,
) -> DataLoader:
    """
    Build a DataLoader by dataset name.
    name: kodak | sipi | clic | coco | isic
    """
    root = Path(data_root) / name
    if crop is None:
        crop = "random" if split == "train" else "center"
    if shuffle is None:
        shuffle = split == "train"

    ds = ImageFolderFlat(root, image_size=image_size, crop=crop)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train"),
    )


def quick_stats(data_root: str = "data"):
    """Print quick stats for each dataset folder."""
    for name in ["kodak", "sipi", "clic", "coco", "isic"]:
        d = Path(data_root) / name
        if not d.exists():
            print(f"{name:8s}: MISSING")
            continue
        files = [f for f in d.rglob("*") if f.is_file() and f.suffix.lower() in _EXTS and not f.name.startswith(".") and "__MACOSX" not in str(f)]
        size_mb = sum(f.stat().st_size for f in files) / 1024 / 1024
        print(f"{name:8s}: {len(files):5d} images, {size_mb:7.1f} MB")


if __name__ == "__main__":
    import sys
    data_root = sys.argv[1] if len(sys.argv) > 1 else "data"
    quick_stats(data_root)
