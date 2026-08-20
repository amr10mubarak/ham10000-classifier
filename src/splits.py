"""
Grouped train/val/test split for HAM10000.

Splits are by lesion_id, not image_id. HAM10000 contains multiple images
of the same physical lesion; a naive per-image split leaks lesions across
train and validation and inflates apparent performance.

Produces three CSVs in data/splits/: train.csv, val.csv, test.csv.
Each row keeps the original metadata columns plus a `label` column
(the integer index into cfg.data.classes).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import GroupShuffleSplit


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def add_label_column(df: pd.DataFrame, classes: list[str]) -> pd.DataFrame:
    """Map the string diagnosis in `dx` to an integer label.

    The mapping is defined by the order of `classes` in config.yaml.
    Freezing this mapping in config (not inferring from the data) is what
    prevents label indices from silently reordering between runs.
    """
    class_to_idx = {name: i for i, name in enumerate(classes)}
    unknown = set(df["dx"]) - set(class_to_idx)
    if unknown:
        raise ValueError(f"Unknown dx values in metadata: {unknown}")
    df = df.copy()
    df["label"] = df["dx"].map(class_to_idx).astype(int)
    return df


def grouped_three_way_split(
    df: pd.DataFrame,
    group_col: str,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split df into train/val/test so that no group appears in more than one set.

    Implemented as two successive GroupShuffleSplits, because sklearn does
    not ship a native three-way group split.
    """
    if not abs(train_frac + val_frac + test_frac - 1.0) < 1e-9:
        raise ValueError(
            f"Split fractions must sum to 1.0, got {train_frac + val_frac + test_frac}"
        )

    # Step 1: peel off the test set.
    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    trainval_idx, test_idx = next(gss1.split(df, groups=df[group_col]))
    trainval_df = df.iloc[trainval_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    # Step 2: split the remainder into train and val.
    # We want val to be `val_frac` of the ORIGINAL data, which is
    # val_frac / (train_frac + val_frac) of the remainder.
    val_size_of_remainder = val_frac / (train_frac + val_frac)
    gss2 = GroupShuffleSplit(
        n_splits=1, test_size=val_size_of_remainder, random_state=seed
    )
    train_idx, val_idx = next(gss2.split(trainval_df, groups=trainval_df[group_col]))
    train_df = trainval_df.iloc[train_idx].reset_index(drop=True)
    val_df = trainval_df.iloc[val_idx].reset_index(drop=True)

    return train_df, val_df, test_df


def assert_no_group_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    group_col: str,
) -> None:
    """Fail loudly if any group appears in more than one split.

    This is a defensive check: if it ever trips, the split logic is broken
    and any downstream metrics are meaningless.
    """
    train_g = set(train_df[group_col])
    val_g = set(val_df[group_col])
    test_g = set(test_df[group_col])

    overlaps = {
        "train ∩ val": train_g & val_g,
        "train ∩ test": train_g & test_g,
        "val ∩ test": val_g & test_g,
    }
    for name, shared in overlaps.items():
        if shared:
            raise AssertionError(
                f"Group leakage detected in {name}: {len(shared)} shared {group_col}s. "
                "Split is broken."
            )


def report(df: pd.DataFrame, name: str, classes: list[str]) -> None:
    """Print sanity numbers about a split: image count, lesion count, class dist."""
    print(f"\n[{name}] {len(df):>5} images, {df['lesion_id'].nunique():>5} lesions")
    counts = df["dx"].value_counts().reindex(classes, fill_value=0)
    pct = (counts / len(df) * 100).round(1)
    for cls in classes:
        print(f"    {cls:<6} {counts[cls]:>5}  ({pct[cls]:>5}%)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    args = parser.parse_args()

    cfg = load_config(args.config)

    metadata_path = Path(cfg["data"]["root"]) / cfg["data"]["metadata_csv"]
    df = pd.read_csv(metadata_path)
    print(f"Loaded {len(df)} rows from {metadata_path}")

    df = add_label_column(df, cfg["data"]["classes"])

    train_df, val_df, test_df = grouped_three_way_split(
        df,
        group_col="lesion_id",
        train_frac=cfg["data"]["split"]["train"],
        val_frac=cfg["data"]["split"]["val"],
        test_frac=cfg["data"]["split"]["test"],
        seed=cfg["seed"],
    )

    assert_no_group_leakage(train_df, val_df, test_df, group_col="lesion_id")
    print("Leakage check passed: no lesion_id appears in more than one split.")

    for name, split in [("train", train_df), ("val", val_df), ("test", test_df)]:
        report(split, name, cfg["data"]["classes"])

    splits_dir = Path(cfg["data"]["splits_dir"])
    splits_dir.mkdir(parents=True, exist_ok=True)

    for name, split in [("train", train_df), ("val", val_df), ("test", test_df)]:
        out = splits_dir / f"{name}.csv"
        split.to_csv(out, index=False)
        print(f"Wrote {out} ({len(split)} rows)")


if __name__ == "__main__":
    main()