"""
Image transforms for HAM10000.

Two pipelines:
  - build_train_transforms: augmentation + resize + normalize (used at training time)
  - build_eval_transforms:  resize + normalize only (used for val and test)

Augmentation choices are deliberately conservative. Every transform is a
claim that the label is invariant to it. For dermatoscopy:
  - Flips and rotations are safe (no canonical orientation for a dermatoscope).
  - Small color jitter is defensible (imaging conditions vary), but pigmentation
    is diagnostic so we keep the magnitude low.
  - No random crops: lesions are centered in HAM10000 and cropping can remove
    diagnostic tissue.
  - No cutout/erasing/color inversion: these break the clinical signal.
"""

from __future__ import annotations

from torchvision import transforms


def build_train_transforms(
    image_size: int,
    mean: list[float],
    std: list[float],
) -> transforms.Compose:
    """Training-time pipeline: augment, resize, tensor, normalize."""
    return transforms.Compose(
        [
            # Resize the shorter side so the whole lesion fits, then take a
            # deterministic center crop to image_size. This avoids stretching
            # the aspect ratio while producing a fixed shape for the model.
            transforms.Resize(image_size + 32),
            transforms.CenterCrop(image_size),

            # Geometric augmentations: safe for dermatoscopy.
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=20),

            # Color augmentation: conservative. Pigmentation and color
            # asymmetry are diagnostic features, so we perturb gently.
            transforms.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1,
                hue=0.02,
            ),

            # Convert PIL image [0, 255] uint8 -> float tensor in [0, 1].
            transforms.ToTensor(),

            # Normalize to the distribution the pretrained backbone expects.
            # For an ImageNet-pretrained model, these are ImageNet mean/std.
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def build_eval_transforms(
    image_size: int,
    mean: list[float],
    std: list[float],
) -> transforms.Compose:
    """Validation/test pipeline: deterministic resize + normalize only.

    No augmentation, ever. Evaluation must be reproducible.
    """
    return transforms.Compose(
        [
            transforms.Resize(image_size + 32),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )