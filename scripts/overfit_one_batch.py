"""Sanity check: can the model overfit a single batch?

If loss doesn't crash toward zero in ~100 steps, something is wrong
with the pipeline. Run this before any real training.
"""
import argparse

import torch

from src.train import (
    build_dataloaders,
    build_loss,
    build_model,
    build_optimizer,
    get_device,
    load_config,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device()
    print(f"Device: {device}")

    model = build_model(cfg["model"]).to(device)
    train_loader, _ = build_dataloaders(cfg)
    loss_fn = build_loss(cfg, train_loader, device)
    optimizer = build_optimizer(cfg, model)

    # Grab exactly one batch and reuse it forever
    images, labels = next(iter(train_loader))
    images = images.to(device)
    labels = labels.to(device)
    print(f"Overfitting one batch: {images.shape[0]} images, "
          f"labels={labels.cpu().tolist()}\n")

    model.train()
    for step in range(1, args.steps + 1):
        logits = model(images)
        loss = loss_fn(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step == 1 or step % 10 == 0:
            preds = logits.argmax(dim=1)
            acc = (preds == labels).float().mean().item()
            print(f"Step {step:3d}  loss={loss.item():.4f}  acc={acc:.2%}")


if __name__ == "__main__":
    main()