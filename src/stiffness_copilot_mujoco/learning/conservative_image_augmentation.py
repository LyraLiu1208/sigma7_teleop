from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


LIGHT_CONSERVATIVE_AUGMENTATION_SPEC_VERSION = "light_conservative_v1"
LIGHT_CONSERVATIVE_AUGMENTATION_MODE = "light"
NO_AUGMENTATION_MODE = "none"
LIGHT_CONSERVATIVE_AUGMENTATION_SCOPE = "training_only"
LIGHT_CONSERVATIVE_AUGMENTATION_WHITELIST = (
    {
        "name": "random_translation",
        "probability": 0.50,
        "max_shift_px": 4,
        "padding_mode": "edge",
        "wraparound": False,
    },
    {
        "name": "brightness_jitter",
        "probability": 0.20,
        "factor_range": [0.90, 1.10],
    },
    {
        "name": "contrast_jitter",
        "probability": 0.20,
        "factor_range": [0.90, 1.10],
    },
    {
        "name": "gaussian_noise",
        "probability": 0.10,
        "std_range": [0.0, 2.0],
        "value_range": [0, 255],
    },
    {
        "name": "very_light_blur",
        "probability": 0.05,
        "radius_max": 0.35,
    },
)
LIGHT_CONSERVATIVE_AUGMENTATION_BLACKLIST = (
    "large rotation",
    "perspective transform",
    "cutout / occlusion",
    "strong color distortion",
    "hue/saturation shift",
    "horizontal/vertical flip",
    "large crop",
    "any transform that can remove peg/hole or alter peg-hole spatial semantics",
)


@dataclass(frozen=True)
class LightConservativeAugmentationConfig:
    translation_probability: float = 0.50
    max_shift_px: int = 4
    brightness_probability: float = 0.20
    brightness_min_factor: float = 0.90
    brightness_max_factor: float = 1.10
    contrast_probability: float = 0.20
    contrast_min_factor: float = 0.90
    contrast_max_factor: float = 1.10
    noise_probability: float = 0.10
    noise_max_std: float = 2.0
    blur_probability: float = 0.05
    blur_max_radius: float = 0.35

    def to_metadata(self) -> dict[str, Any]:
        return {
            "spec_version": LIGHT_CONSERVATIVE_AUGMENTATION_SPEC_VERSION,
            "mode": LIGHT_CONSERVATIVE_AUGMENTATION_MODE,
            "scope": LIGHT_CONSERVATIVE_AUGMENTATION_SCOPE,
            "whitelist": [dict(entry) for entry in LIGHT_CONSERVATIVE_AUGMENTATION_WHITELIST],
            "blacklist": list(LIGHT_CONSERVATIVE_AUGMENTATION_BLACKLIST),
            "parameters": {
                "translation_probability": float(self.translation_probability),
                "max_shift_px": int(self.max_shift_px),
                "brightness_probability": float(self.brightness_probability),
                "brightness_factor_range": [float(self.brightness_min_factor), float(self.brightness_max_factor)],
                "contrast_probability": float(self.contrast_probability),
                "contrast_factor_range": [float(self.contrast_min_factor), float(self.contrast_max_factor)],
                "noise_probability": float(self.noise_probability),
                "noise_max_std": float(self.noise_max_std),
                "blur_probability": float(self.blur_probability),
                "blur_max_radius": float(self.blur_max_radius),
            },
            "reproducibility": "seeded_per_sample_from_train_seed",
        }


DEFAULT_LIGHT_CONSERVATIVE_AUGMENTATION_CONFIG = LightConservativeAugmentationConfig()


def _validate_rgb_image(image: np.ndarray) -> np.ndarray:
    rgb = np.asarray(image, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"RGB image must have shape [H, W, 3], observed {rgb.shape}.")
    return rgb


def _translate_edge_padded(image: np.ndarray, *, shift_y: int, shift_x: int, max_shift_px: int) -> np.ndarray:
    if max_shift_px < 0:
        raise ValueError("max_shift_px must be non-negative.")
    if shift_x == 0 and shift_y == 0:
        return image
    h, w, _ = image.shape
    padded = np.pad(image, ((max_shift_px, max_shift_px), (max_shift_px, max_shift_px), (0, 0)), mode="edge")
    start_y = max_shift_px - int(shift_y)
    start_x = max_shift_px - int(shift_x)
    return padded[start_y : start_y + h, start_x : start_x + w, :]


def _adjust_brightness(image: np.ndarray, *, factor: float) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32) * float(factor)
    return np.clip(arr, 0.0, 255.0)


def _adjust_contrast(image: np.ndarray, *, factor: float) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    mean = arr.mean(axis=(0, 1), keepdims=True)
    arr = (arr - mean) * float(factor) + mean
    return np.clip(arr, 0.0, 255.0)


def _gaussian_kernel_1d(sigma: float) -> np.ndarray:
    sigma = float(sigma)
    if sigma <= 0.0:
        return np.asarray([1.0], dtype=np.float32)
    radius = max(1, int(np.ceil(3.0 * sigma)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(offsets**2) / (2.0 * sigma**2))
    kernel /= float(kernel.sum())
    return kernel.astype(np.float32, copy=False)


def _convolve_along_width(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    pad = int(kernel.size // 2)
    padded = np.pad(image, ((0, 0), (pad, pad), (0, 0)), mode="edge")
    out = np.empty_like(image, dtype=np.float32)
    for x in range(image.shape[1]):
        window = padded[:, x : x + kernel.size, :]
        out[:, x, :] = np.tensordot(window, kernel, axes=([1], [0]))
    return out


def _convolve_along_height(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    pad = int(kernel.size // 2)
    padded = np.pad(image, ((pad, pad), (0, 0), (0, 0)), mode="edge")
    out = np.empty_like(image, dtype=np.float32)
    for y in range(image.shape[0]):
        window = padded[y : y + kernel.size, :, :]
        out[y, :, :] = np.tensordot(window, kernel, axes=([0], [0]))
    return out


def _apply_gaussian_blur(image: np.ndarray, *, radius: float) -> np.ndarray:
    radius = float(radius)
    if radius <= 0.0:
        return np.asarray(image, dtype=np.float32)
    kernel = _gaussian_kernel_1d(radius)
    arr = np.asarray(image, dtype=np.float32)
    blurred = _convolve_along_width(arr, kernel)
    blurred = _convolve_along_height(blurred, kernel)
    return np.clip(blurred, 0.0, 255.0)


def augment_light_conservative_rgb_image(
    image: np.ndarray,
    *,
    train_seed: int,
    sample_id: int,
    config: LightConservativeAugmentationConfig = DEFAULT_LIGHT_CONSERVATIVE_AUGMENTATION_CONFIG,
) -> np.ndarray:
    rgb = _validate_rgb_image(image)
    seed_sequence = np.random.SeedSequence([int(train_seed), int(sample_id)])
    rng = np.random.default_rng(seed_sequence)

    augmented = np.asarray(rgb, dtype=np.float32)

    if rng.random() < float(config.translation_probability):
        shift_y = int(rng.integers(-int(config.max_shift_px), int(config.max_shift_px) + 1))
        shift_x = int(rng.integers(-int(config.max_shift_px), int(config.max_shift_px) + 1))
        augmented = _translate_edge_padded(
            np.asarray(np.clip(augmented, 0.0, 255.0), dtype=np.uint8),
            shift_y=shift_y,
            shift_x=shift_x,
            max_shift_px=int(config.max_shift_px),
        ).astype(np.float32)

    if rng.random() < float(config.brightness_probability):
        factor = float(rng.uniform(float(config.brightness_min_factor), float(config.brightness_max_factor)))
        augmented = _adjust_brightness(augmented, factor=factor)

    if rng.random() < float(config.contrast_probability):
        factor = float(rng.uniform(float(config.contrast_min_factor), float(config.contrast_max_factor)))
        augmented = _adjust_contrast(augmented, factor=factor)

    if rng.random() < float(config.noise_probability):
        sigma = float(rng.uniform(0.0, float(config.noise_max_std)))
        if sigma > 0.0:
            augmented = np.clip(augmented + rng.normal(0.0, sigma, size=augmented.shape), 0.0, 255.0)

    if rng.random() < float(config.blur_probability):
        radius = float(rng.uniform(0.0, float(config.blur_max_radius)))
        if radius > 0.0:
            augmented = _apply_gaussian_blur(augmented, radius=radius)

    return np.clip(np.rint(augmented), 0, 255).astype(np.uint8, copy=False)


def augment_light_conservative_rgb_batch(
    rgb_images: np.ndarray,
    *,
    train_seed: int,
    sample_ids: np.ndarray,
    config: LightConservativeAugmentationConfig = DEFAULT_LIGHT_CONSERVATIVE_AUGMENTATION_CONFIG,
) -> np.ndarray:
    images = np.asarray(rgb_images, dtype=np.uint8)
    if images.ndim == 3:
        images = images[None, ...]
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(f"RGB batch must have shape [N, H, W, 3], observed {images.shape}.")
    ids = np.asarray(sample_ids, dtype=np.int64)
    if ids.ndim != 1 or ids.shape[0] != images.shape[0]:
        raise ValueError("sample_ids must be one-dimensional and match the batch length.")
    augmented = np.empty_like(images)
    for index, (image, sample_id) in enumerate(zip(images, ids, strict=True)):
        augmented[index] = augment_light_conservative_rgb_image(
            image,
            train_seed=train_seed,
            sample_id=int(sample_id),
            config=config,
        )
    return augmented


def describe_light_conservative_augmentation() -> dict[str, Any]:
    return DEFAULT_LIGHT_CONSERVATIVE_AUGMENTATION_CONFIG.to_metadata()


__all__ = [
    "DEFAULT_LIGHT_CONSERVATIVE_AUGMENTATION_CONFIG",
    "LIGHT_CONSERVATIVE_AUGMENTATION_BLACKLIST",
    "LIGHT_CONSERVATIVE_AUGMENTATION_MODE",
    "LIGHT_CONSERVATIVE_AUGMENTATION_SCOPE",
    "LIGHT_CONSERVATIVE_AUGMENTATION_SPEC_VERSION",
    "LIGHT_CONSERVATIVE_AUGMENTATION_WHITELIST",
    "NO_AUGMENTATION_MODE",
    "LightConservativeAugmentationConfig",
    "augment_light_conservative_rgb_batch",
    "augment_light_conservative_rgb_image",
    "describe_light_conservative_augmentation",
]
