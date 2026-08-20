"""
Phase 1 gate: prove the data pipeline works end to end.

Loads config, builds the training Dataset, wraps it in a DataLoader,
pulls one batch, prints its shape and label distribution. If this
passes, Phase 1 is done.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from torch.utils.data import DataLoader

from src.data import HAM10000Dataset
from src.transforms import build_train_transforms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="train",
        help="Which split CSV to load a batch from.",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    transform = build_train_transforms(
        image_size=cfg["training"]["image_size"],
        mean=cfg["normalize"]["mean"],
        std=cfg["normalize"]["std"],
    )

    dataset = HAM10000Dataset(
        split_csv=Path(cfg["data"]["splits_dir"]) / f"{args.split}.csv",
        data_root=Path(cfg["data"]["root"]),
        image_dirs=cfg["data"]["image_dirs"],
        transform=transform,
    )
    print(f"Dataset size: {len(dataset)} images")

    loader = DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        # num_workers=0 keeps things simple and debuggable in this smoke test.
        # We'll turn parallelism on in the real training loop.
        num_workers=0,
    )

    images, labels = next(iter(loader))
    print(f"Batch images shape: {tuple(images.shape)}")
    print(f"Batch labels shape: {tuple(labels.shape)}")
    print(f"Image dtype: {images.dtype}, label dtype: {labels.dtype}")
    print(f"Pixel range: min={images.min():.3f}, max={images.max():.3f}")
    print(f"Labels in this batch: {labels.tolist()}")


if __name__ == "__main__":
    main()