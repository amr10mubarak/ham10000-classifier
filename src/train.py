"""Training script for HAM10000 classifier.

Reads all hyperparameters from config.yaml. Nothing hardcoded.
"""
import argparse
from pathlib import Path


import timm
import torch
import yaml

from torch.utils.data import DataLoader

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device()
    print(f"Device: {device}")

    model = build_model(cfg["model"])
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {cfg['model']['name']}, {n_params:,} parameters")

    train_loader, val_loader = build_dataloaders(cfg)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    loss_fn = build_loss(cfg, train_loader, device)

    optimizer = build_optimizer(cfg, model)
    print(f"Optimizer: {type(optimizer).__name__}, lr={cfg['training']['learning_rate']}")

    # One quick epoch to prove the loop runs
    print("\nRunning one training epoch...")
    avg_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
    print(f"Epoch 1 average loss: {avg_loss:.4f}")

if __name__ == "__main__":
    main()