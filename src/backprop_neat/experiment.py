"""Experiment orchestration with validation-only selection and saved artifacts."""

import csv
import json
import platform
import time
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .config import Config
from .data import Dataset, make_dataset
from .evolution import SpeciesSet
from .genome import Genome, InnovationTracker, fixed_genome, minimal_genome
from .network import evaluate_population, pack_population, train_population, unpack_population


@dataclass
class Result:
    config: Config
    champion: Genome
    history: list[dict]
    metrics: dict
    dataset: Dataset


def write_json(path: Path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def run(config: Config, output: str | Path | None = None, *, verbose: bool = True):
    if output is not None and Path(output).exists() and any(Path(output).iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    started = time.perf_counter()
    dataset = make_dataset(config)
    # A separate stream keeps evolution randomness independent of data generation.
    rng = np.random.default_rng(np.random.SeedSequence([config.seed, 947]))
    tracker = InnovationTracker()
    population = [
        (
            fixed_genome(tracker, rng, config.fixed_hidden)
            if config.mode == "fixed"
            else minimal_genome(tracker, rng)
        )
        for _ in range(config.population_size)
    ]
    species_set = SpeciesSet()
    history = []
    champion = None
    champion_generation = 0
    train_x, train_y = jnp.asarray(dataset.x_train), jnp.asarray(dataset.y_train)
    val_x, val_y = jnp.asarray(dataset.x_val), jnp.asarray(dataset.y_val)
    for generation in range(config.generations):
        params, masks = pack_population(population, config.max_nodes)
        if config.mode != "evolution" and config.train_steps:
            params = train_population(
                params,
                masks,
                train_x,
                train_y,
                steps=config.train_steps,
                learning_rate=config.learning_rate,
                l2=config.l2,
            )
        train_losses, train_accuracies = map(
            np.asarray, evaluate_population(params, masks, train_x, train_y)
        )
        val_losses, val_accuracies = map(
            np.asarray, evaluate_population(params, masks, val_x, val_y)
        )
        if not np.isfinite(train_losses).all() or not np.isfinite(val_losses).all():
            raise FloatingPointError(
                "Non-finite network loss; reduce learning rate or weight scale"
            )
        # Lamarckian inheritance: trained parameters are written into historical genes.
        unpack_population(population, params)
        for genome, loss in zip(population, val_losses, strict=True):
            genome.fitness = -float(loss) - config.complexity_penalty * genome.enabled_connections
        best_index = max(range(len(population)), key=lambda i: population[i].fitness)
        best = population[best_index]
        if champion is None or best.fitness > champion.fitness:
            champion = best.clone()
            champion_generation = generation
        species_set.assign(population, generation, config)
        record = {
            "generation": generation,
            "train_loss": float(train_losses[best_index]),
            "val_loss": float(val_losses[best_index]),
            "train_accuracy": float(train_accuracies[best_index]),
            "val_accuracy": float(val_accuracies[best_index]),
            "best_so_far_fitness": champion.fitness,
            "mean_val_accuracy": float(val_accuracies.mean()),
            "species": len(species_set.species),
            "best_hidden_nodes": best.hidden_nodes,
            "best_connections": best.enabled_connections,
            "mean_hidden_nodes": float(np.mean([g.hidden_nodes for g in population])),
            "max_hidden_nodes": max(g.hidden_nodes for g in population),
            "mean_connections": float(np.mean([g.enabled_connections for g in population])),
        }
        history.append(record)
        if verbose:
            print(
                f"{config.task:6} {config.mode:9} gen={generation:03d} "
                f"train={record['train_accuracy']:.3f} val={record['val_accuracy']:.3f} "
                f"loss={record['val_loss']:.4f} hidden={best.hidden_nodes} "
                f"edges={best.enabled_connections} species={record['species']}",
                flush=True,
            )
        if generation + 1 < config.generations:
            population = species_set.reproduce(tracker, rng, config, generation)
    assert champion is not None
    params, masks = pack_population([champion], config.max_nodes)
    metrics = {}
    # The test split is evaluated only after architecture/weight selection finishes.
    for split in ("train", "val", "test"):
        losses, accuracies = evaluate_population(
            params,
            masks,
            jnp.asarray(getattr(dataset, f"x_{split}")),
            jnp.asarray(getattr(dataset, f"y_{split}")),
        )
        metrics[f"{split}_loss"] = float(losses[0])
        metrics[f"{split}_accuracy"] = float(accuracies[0])
    metrics.update(
        {
            "selected_generation": champion_generation,
            "hidden_nodes": champion.hidden_nodes,
            "enabled_connections": champion.enabled_connections,
            "total_connection_genes": len(champion.connections),
            "elapsed_seconds": time.perf_counter() - started,
        }
    )
    result = Result(config, champion, history, metrics, dataset)
    if output is not None:
        save_result(result, Path(output))
    return result


def save_result(result: Result, output: Path):
    # Prevent silently mixing outputs from different experiments.
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    write_json(output / "config.json", result.config.to_dict())
    write_json(output / "metrics.json", result.metrics)
    write_json(output / "champion.json", result.champion.to_dict())
    write_json(
        output / "environment.json",
        {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                name: version(name) for name in ("jax", "jaxlib", "optax", "numpy", "matplotlib")
            },
            "devices": [str(device) for device in jax.devices()],
        },
    )
    with (output / "history.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result.history[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(result.history)
    np.savez_compressed(output / "dataset.npz", **vars(result.dataset))
    from .plots import plot_result

    plot_result(result, output)
