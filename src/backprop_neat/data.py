"""Synthetic, independent train/validation/test classification samples."""

from dataclasses import dataclass

import numpy as np

from .config import Config


@dataclass
class Dataset:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


def sample(task: str, n: int, rng: np.random.Generator, noise: float = 0.05):
    if task in {"xor", "circle"}:
        x = rng.uniform(-1.0, 1.0, (n, 2))
        y = (
            ((x[:, 0] > 0) != (x[:, 1] > 0))
            if task == "xor"
            else (np.sum(x**2, axis=1) < 2 / np.pi)
        )
        # Labels refer to the clean coordinates; observation noise can obscure the boundary.
        x += rng.normal(0, noise, x.shape)
    elif task == "spiral":
        y = np.arange(n) % 2
        rng.shuffle(y)
        radius = np.sqrt(rng.uniform(0.01, 1, n))
        angle = 3 * np.pi * radius + np.pi * y
        x = np.column_stack((radius * np.cos(angle), radius * np.sin(angle)))
        x += rng.normal(0, noise, x.shape)
    else:
        raise ValueError(f"Unknown task: {task}")
    return x.astype(np.float32), y.astype(np.float32)


def make_dataset(config: Config):
    streams = np.random.SeedSequence(config.seed).spawn(3)
    splits = [
        sample(config.task, n, np.random.default_rng(seed), config.noise)
        for n, seed in zip(
            (config.train_size, config.val_size, config.test_size), streams, strict=True
        )
    ]
    return Dataset(*(array for split in splits for array in split))
