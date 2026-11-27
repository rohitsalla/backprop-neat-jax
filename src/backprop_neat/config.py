"""Validated, JSON-serializable experiment settings."""

from dataclasses import asdict, dataclass, fields
from math import isfinite
from numbers import Integral, Real


@dataclass(frozen=True)
class Config:
    task: str = "xor"
    mode: str = "hybrid"
    seed: int = 42
    population_size: int = 32
    generations: int = 40
    train_steps: int = 80
    learning_rate: float = 0.03
    train_size: int = 256
    val_size: int = 128
    test_size: int = 512
    noise: float = 0.05
    max_nodes: int = 32
    fixed_hidden: int = 12
    add_node_rate: float = 0.35
    add_connection_rate: float = 0.6
    weight_mutation_rate: float = 0.8
    weight_sigma: float = 0.25
    crossover_rate: float = 0.75
    compatibility_threshold: float = 3.0
    excess_coefficient: float = 1.0
    disjoint_coefficient: float = 1.0
    weight_coefficient: float = 0.4
    survival_fraction: float = 0.5
    stagnation_generations: int = 15
    selection_temperature: float = 0.3
    complexity_penalty: float = 0.0001
    l2: float = 0.00001

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(field.default, int):
                if isinstance(value, bool) or not isinstance(value, Integral):
                    raise ValueError(f"{field.name} must be an integer")
            elif isinstance(field.default, float):
                if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
                    raise ValueError(f"{field.name} must be a finite number")
        if self.task not in {"xor", "circle", "spiral"}:
            raise ValueError("task must be xor, circle, or spiral")
        if self.mode not in {"hybrid", "evolution", "fixed"}:
            raise ValueError("mode must be hybrid, evolution, or fixed")
        for name in ("population_size", "generations", "train_size", "val_size", "test_size"):
            if getattr(self, name) < (
                2 if name == "population_size" else 4 if "size" in name else 1
            ):
                raise ValueError(f"{name} is too small")
        if self.train_steps < 0 or self.max_nodes < 3 or self.fixed_hidden < 0:
            raise ValueError("invalid training steps or node limits")
        if self.mode == "fixed" and self.fixed_hidden + 3 > self.max_nodes:
            raise ValueError("fixed_hidden + 3 exceeds max_nodes")
        for name in (
            "add_node_rate",
            "add_connection_rate",
            "weight_mutation_rate",
            "crossover_rate",
            "survival_fraction",
        ):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.survival_fraction == 0:
            raise ValueError("survival_fraction must be positive")
        for name in ("learning_rate", "compatibility_threshold", "selection_temperature"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in (
            "noise",
            "weight_sigma",
            "complexity_penalty",
            "l2",
            "excess_coefficient",
            "disjoint_coefficient",
            "weight_coefficient",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        if self.stagnation_generations < 1 or self.seed < 0:
            raise ValueError("invalid stagnation limit or seed")

    def to_dict(self):
        return asdict(self)
