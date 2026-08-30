"""Training script for HAM10000 classifier.

Reads all hyperparameters from config.yaml. Nothing hardcoded.
"""
import argparse
from pathlib import Path

import csv
import os
import timm
import torch
import yaml

from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, recall_score
from src.data import HAM10000Dataset
from src.transforms import build_train_transforms, build_eval_transforms


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_device() -> torch.device:
    """Pick the best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(model_cfg: dict) -> torch.nn.Module:
    """Build a timm model with the right number of output classes."""
    model = timm.create_model(
        model_cfg["name"],
        pretrained=model_cfg["pretrained"],
        num_classes=model_cfg["num_classes"],
    )
    return model


def build_dataloaders(cfg: dict) -> tuple[DataLoader, DataLoader]:
    """Build train and val dataloaders from config."""
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    norm_cfg = cfg["normalize"]

    train_tf = build_train_transforms(
        image_size=train_cfg["image_size"],
        mean=norm_cfg["mean"],
        std=norm_cfg["std"],
    )
    eval_tf = build_eval_transforms(
        image_size=train_cfg["image_size"],
        mean=norm_cfg["mean"],
        std=norm_cfg["std"],
    )

    splits_dir = Path(data_cfg["splits_dir"])
    train_ds = HAM10000Dataset(
        split_csv=splits_dir / "train.csv",
        data_root=data_cfg["root"],
        image_dirs=data_cfg["image_dirs"],
        transform=train_tf,
    )
    val_ds = HAM10000Dataset(
        split_csv=splits_dir / "val.csv",
        data_root=data_cfg["root"],
        image_dirs=data_cfg["image_dirs"],
        transform=eval_tf,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
        pin_memory=True,
    )

    return train_loader, val_loader

def build_loss(cfg: dict, train_loader: DataLoader, device: torch.device) -> torch.nn.Module:
    """Build the loss function, weighted by inverse class frequency if configured."""
    strategy = cfg["training"]["imbalance_strategy"]

    if strategy == "weighted_loss":
        # Count class occurrences in the training set
        labels = [label for _, label in train_loader.dataset]
        counts = torch.zeros(cfg["model"]["num_classes"])
        for lbl in labels:
            counts[lbl] += 1
        # Inverse frequency, normalized so weights average to 1
        weights = counts.sum() / (counts * len(counts))
        weights = weights.to(device)
        print(f"Class weights: {weights.cpu().numpy().round(3)}")
        return torch.nn.CrossEntropyLoss(weight=weights)

    if strategy == "none":
        return torch.nn.CrossEntropyLoss()

    raise ValueError(f"Unknown imbalance_strategy: {strategy}")


def build_optimizer(cfg: dict, model: torch.nn.Module) -> torch.optim.Optimizer:
    """Build the optimizer from config."""
    train_cfg = cfg["training"]
    name = train_cfg["optimizer"].lower()

    if name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=train_cfg["learning_rate"],
            weight_decay=train_cfg["weight_decay"],
        )

    raise ValueError(f"Unknown optimizer: {name}")

def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Run one epoch of training. Returns average loss over the epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        # The five-step dance
        logits = model(images)              # 1. forward
        loss = loss_fn(logits, labels)      # 2. loss
        optimizer.zero_grad()               # 3. zero grads
        loss.backward()                     # 4. backward
        optimizer.step()                    # 5. update

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches

@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
    num_classes: int,
) -> dict:
    """Evaluate the model on a loader. Returns loss, macro F1, per-class recall."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = loss_fn(logits, labels)
        preds = logits.argmax(dim=1)

        total_loss += loss.item()
        n_batches += 1
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    return {
        "loss": total_loss / n_batches,
        "macro_f1": f1_score(all_labels, all_preds, average="macro", zero_division=0),
        "per_class_recall": recall_score(
            all_labels, all_preds, average=None, zero_division=0,
            labels=list(range(num_classes)),
        ),
    }

def log_epoch(
    csv_path: str,
    epoch: int,
    train_loss: float,
    val_metrics: dict,
    class_names: list[str],
) -> None:
    """Append one row of per-epoch metrics to a CSV. Creates header on first call."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    is_new = not os.path.exists(csv_path)

    row = {
        "epoch": epoch,
        "train_loss": round(train_loss, 4),
        "val_loss": round(val_metrics["loss"], 4),
        "val_macro_f1": round(val_metrics["macro_f1"], 4),
    }
    for name, recall in zip(class_names, val_metrics["per_class_recall"]):
        row[f"recall_{name}"] = round(recall, 4)

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if is_new:
            writer.writeheader()
        writer.writerow(row)

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_metrics: dict,
    path: str,
) -> None:
    """Save model weights + optimizer state + metadata for resuming."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_metrics": val_metrics,
    }, path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device()
    print(f"Device: {device}")

    model = build_model(cfg["model"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {cfg['model']['name']}, {n_params:,} parameters")

    train_loader, val_loader = build_dataloaders(cfg)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    loss_fn = build_loss(cfg, train_loader, device)
    optimizer = build_optimizer(cfg, model)
    print(f"Optimizer: {type(optimizer).__name__}, "
          f"lr={cfg['training']['learning_rate']}\n")

    num_epochs = cfg["training"]["epochs"]
    num_classes = cfg["model"]["num_classes"]
    class_names = cfg["data"]["classes"]
    ckpt_path = os.path.join(cfg["training"]["checkpoint_dir"], "best.pt")
    csv_path = "results/metrics.csv"

    best_f1 = -1.0
    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_metrics = evaluate(model, val_loader, loss_fn, device, num_classes)

        print(
            f"Epoch {epoch:2d}/{num_epochs}  "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_metrics['loss']:.4f}  "
            f"val_f1={val_metrics['macro_f1']:.4f}"
        )
        log_epoch(csv_path, epoch, train_loss, val_metrics, class_names)

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            save_checkpoint(model, optimizer, epoch, val_metrics, ckpt_path)
            print(f"  ↑ new best val F1: {best_f1:.4f} — checkpoint saved")

    print(f"\nTraining complete. Best val F1: {best_f1:.4f}")

if __name__ == "__main__":
    main()