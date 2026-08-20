"""
Dataset for HAM10000.

Reads a split CSV produced by src/splits.py, resolves each image_id to a
JPG file on disk, applies a torchvision transform, and returns (tensor, label).

Images are spread across two folders (HAM10000_images_part_1 and part_2),
so we build a lookup from image_id -> full path up front. Doing this once
in __init__ is much faster than searching the filesystem on every __getitem__.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class HAM10000Dataset(Dataset):
    """A HAM10000 split (train, val, or test) as a PyTorch Dataset.

    Parameters
    ----------
    split_csv : Path
        Path to a CSV produced by src/splits.py. Must contain columns
        `image_id` and `label` (integer class index).
    data_root : Path
        Directory containing the HAM10000 image folders.
    image_dirs : list[str]
        Subdirectories of data_root that contain image files.
    transform : callable | None
        A torchvision transform applied to the PIL image before it is returned.
        If None, the raw PIL image is returned (rare; only useful for debugging).
    """

    def __init__(
        self,
        split_csv: Path,
        data_root: Path,
        image_dirs: list[str],
        transform=None,
    ):
        self.df = pd.read_csv(split_csv)
        self.transform = transform

        # Sanity-check the CSV shape early. A bad CSV here would produce
        # very confusing errors deep inside the DataLoader.
        required_cols = {"image_id", "label"}
        missing = required_cols - set(self.df.columns)
        if missing:
            raise ValueError(f"{split_csv} missing required columns: {missing}")

        # Build image_id -> path lookup once, up front.
        # HAM10000 images live in two directories; we index both.
        self.image_id_to_path: dict[str, Path] = {}
        for subdir in image_dirs:
            for jpg in (Path(data_root) / subdir).glob("*.jpg"):
                self.image_id_to_path[jpg.stem] = jpg

        # Fail loudly if the CSV references images that don't exist on disk.
        # Silent misses here would show up as random KeyErrors mid-training.
        csv_ids = set(self.df["image_id"])
        disk_ids = set(self.image_id_to_path)
        missing_on_disk = csv_ids - disk_ids
        if missing_on_disk:
            raise FileNotFoundError(
                f"{len(missing_on_disk)} image_ids in {split_csv} not found on disk. "
                f"First few: {sorted(missing_on_disk)[:3]}"
            )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        image_id = row["image_id"]
        label = int(row["label"])

        path = self.image_id_to_path[image_id]
        # .convert("RGB") normalizes any grayscale or RGBA quirks to a 3-channel
        # image, matching what the pretrained ResNet50 expects.
        image = Image.open(path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label