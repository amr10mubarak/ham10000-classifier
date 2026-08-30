"""Evaluate the best checkpoint on the held-out test set.

This is the one and only look at test data. Produces:
- Printed metrics: macro F1, per-class recall & precision
- results/test_metrics.json: same metrics, machine-readable
- results/confusion_matrix.png: visual error analysis
"""
import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

from src.data import HAM10000Dataset
from src.train import build_model, get_device, load_config
from src.transforms import build_eval_transforms


def load_checkpoint(model: torch.nn.Module, path: str, device: torch.device) -> dict:
    """Load model weights from a checkpoint file. Returns the full checkpoint dict."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    return ckpt


def build_test_loader(cfg: dict) -> DataLoader:
    """Build the test dataloader — eval transforms only, no shuffle."""
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    norm_cfg = cfg["normalize"]

    eval_tf = build_eval_transforms(
        image_size=train_cfg["image_size"],
        mean=norm_cfg["mean"],
        std=norm_cfg["std"],
    )
    test_ds = HAM10000Dataset(
        split_csv=Path(data_cfg["splits_dir"]) / "test.csv",
        data_root=data_cfg["root"],
        image_dirs=data_cfg["image_dirs"],
        transform=eval_tf,
    )
    return DataLoader(
        test_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
        pin_memory=True,
    )


@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the model over the loader. Returns (labels, predictions) as numpy arrays."""
    model.eval()
    all_preds, all_labels = [], []
    for images, labels in loader:
        images = images.to(device)
        preds = model(images).argmax(dim=1).cpu()
        all_preds.append(preds)
        all_labels.append(labels)
    return (
        torch.cat(all_labels).numpy(),
        torch.cat(all_preds).numpy(),
    )


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], out_path: str) -> None:
    """Save a labeled confusion matrix as a PNG. Rows = true, cols = predicted."""
    fig, ax = plt.subplots(figsize=(8, 7))
    # Row-normalize so each row sums to 1 — recall interpretation
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Test set confusion matrix (row-normalized)")

    # Annotate each cell with count and percentage
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            count = cm[i, j]
            pct = cm_norm[i, j]
            color = "white" if pct > 0.5 else "black"
            ax.text(j, i, f"{count}\n{pct:.0%}",
                    ha="center", va="center", color=color, fontsize=9)

    fig.colorbar(im, ax=ax, label="Row-normalized (recall)")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"Confusion matrix saved to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device()
    class_names = cfg["data"]["classes"]
    num_classes = cfg["model"]["num_classes"]

    print(f"Device: {device}")
    print(f"Loading checkpoint: {args.checkpoint}")
    model = build_model(cfg["model"]).to(device)
    ckpt = load_checkpoint(model, args.checkpoint, device)
    print(f"Checkpoint from epoch {ckpt['epoch']}, "
          f"val F1={ckpt['val_metrics']['macro_f1']:.4f}\n")

    test_loader = build_test_loader(cfg)
    print(f"Test set: {len(test_loader.dataset)} images, "
          f"{len(test_loader)} batches\n")

    labels, preds = collect_predictions(model, test_loader, device)

    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    per_class_recall = recall_score(
        labels, preds, average=None, zero_division=0,
        labels=list(range(num_classes)),
    )
    per_class_precision = precision_score(
        labels, preds, average=None, zero_division=0,
        labels=list(range(num_classes)),
    )

    print("=" * 60)
    print(f"TEST SET RESULTS")
    print("=" * 60)
    print(f"Macro F1: {macro_f1:.4f}\n")
    print(f"{'Class':<8}{'Recall':>10}{'Precision':>12}{'Support':>10}")
    print("-" * 40)
    for i, name in enumerate(class_names):
        support = int((labels == i).sum())
        print(f"{name:<8}{per_class_recall[i]:>10.4f}"
              f"{per_class_precision[i]:>12.4f}{support:>10}")

    # Save machine-readable metrics
    os.makedirs("results", exist_ok=True)
    metrics = {
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": ckpt["epoch"],
        "checkpoint_val_f1": ckpt["val_metrics"]["macro_f1"],
        "test_macro_f1": float(macro_f1),
        "test_per_class_recall": {
            name: float(r) for name, r in zip(class_names, per_class_recall)
        },
        "test_per_class_precision": {
            name: float(p) for name, p in zip(class_names, per_class_precision)
        },
        "test_support": {
            name: int((labels == i).sum())
            for i, name in enumerate(class_names)
        },
    }
    with open("results/test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to results/test_metrics.json")

    # Confusion matrix
    cm = confusion_matrix(labels, preds, labels=list(range(num_classes)))
    plot_confusion_matrix(cm, class_names, "results/confusion_matrix.png")


if __name__ == "__main__":
    main()