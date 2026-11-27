"""Command-line experiment runner, multi-seed benchmarks, and saved-model inference."""

import argparse
import csv
import json
from dataclasses import fields
from pathlib import Path

import numpy as np

from .config import Config
from .experiment import run, write_json
from .genome import Genome
from .network import predict


def config_arguments(parser):
    parser.add_argument("--config", type=Path, help="JSON config; explicit flags override it")
    for field in fields(Config):
        flag = "--" + field.name.replace("_", "-")
        parser.add_argument(flag, type=type(field.default), default=None)


def build_config(args):
    payload = json.loads(args.config.read_text()) if args.config else {}
    payload.update(
        {
            f.name: getattr(args, f.name)
            for f in fields(Config)
            if getattr(args, f.name, None) is not None
        }
    )
    return Config(**payload)


def main(argv=None):
    parser = argparse.ArgumentParser(description="NEAT + JAX backpropagation")
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train", help="Run a single experiment")
    config_arguments(train)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--quiet", action="store_true")
    benchmark = subparsers.add_parser("benchmark", help="Compare methods across tasks and seeds")
    config_arguments(benchmark)
    benchmark.add_argument(
        "--tasks",
        nargs="+",
        choices=["xor", "circle", "spiral"],
        default=["xor", "circle", "spiral"],
    )
    benchmark.add_argument(
        "--modes",
        nargs="+",
        choices=["hybrid", "evolution", "fixed"],
        default=["hybrid", "evolution", "fixed"],
    )
    benchmark.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    benchmark.add_argument("--output", type=Path, required=True)
    infer = subparsers.add_parser("predict", help="Predict from a saved champion JSON")
    infer.add_argument("--model", type=Path, required=True)
    infer.add_argument("--input", type=Path, required=True, help="CSV with header x1,x2")
    infer.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "predict":
        genome = Genome.from_dict(json.loads(args.model.read_text()))
        data = np.genfromtxt(args.input, delimiter=",", names=True, dtype=np.float32)
        x = np.column_stack((np.atleast_1d(data["x1"]), np.atleast_1d(data["x2"])))
        if not np.isfinite(x).all():
            parser.error("input must contain finite numeric x1,x2 values")
        probabilities = predict(genome, x)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(
            args.output,
            np.column_stack((x, probabilities, probabilities >= 0.5)),
            delimiter=",",
            header="x1,x2,probability,label",
            comments="",
            fmt="%.7g",
        )
        return
    config = build_config(args)
    if args.output.exists() and any(args.output.iterdir()):
        parser.error(f"output directory must be empty: {args.output}")
    if args.command == "train":
        result = run(config, args.output, verbose=not args.quiet)
        print(json.dumps(result.metrics, indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    if (
        len(set(args.seeds)) != len(args.seeds)
        or len(set(args.tasks)) != len(args.tasks)
        or len(set(args.modes)) != len(args.modes)
    ):
        parser.error("tasks, modes, and seeds must not contain duplicates")
    for task in args.tasks:
        for mode in args.modes:
            for seed in args.seeds:
                settings = Config(**(config.to_dict() | {"task": task, "mode": mode, "seed": seed}))
                result = run(settings, args.output / task / mode / f"seed_{seed}")
                records.append({"task": task, "mode": mode, "seed": seed, **result.metrics})
                # Save after each completed run so partial benchmark progress remains usable.
                with (args.output / "results.csv").open("w", newline="") as stream:
                    writer = csv.DictWriter(
                        stream, fieldnames=list(records[0]), lineterminator="\n"
                    )
                    writer.writeheader()
                    writer.writerows(records)
    summary = []
    for task in args.tasks:
        for mode in args.modes:
            matching = [r for r in records if r["task"] == task and r["mode"] == mode]
            scores = np.array([r["test_accuracy"] for r in matching])
            summary.append(
                {
                    "task": task,
                    "mode": mode,
                    "runs": len(scores),
                    "test_accuracy_mean": float(scores.mean()),
                    "test_accuracy_std": float(scores.std(ddof=1)) if len(scores) > 1 else None,
                    "hidden_nodes_mean": float(np.mean([r["hidden_nodes"] for r in matching])),
                    "elapsed_seconds_mean": float(
                        np.mean([r["elapsed_seconds"] for r in matching])
                    ),
                }
            )
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
