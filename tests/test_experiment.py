import json
from dataclasses import replace

import numpy as np
import pytest

from backprop_neat.cli import main
from backprop_neat.config import Config
from backprop_neat.data import make_dataset, sample
from backprop_neat.experiment import run
from backprop_neat.genome import Genome
from backprop_neat.network import predict


def small_config(**kwargs):
    return replace(
        Config(
            population_size=6,
            generations=4,
            train_steps=6,
            train_size=24,
            val_size=16,
            test_size=20,
            max_nodes=9,
            fixed_hidden=3,
            add_node_rate=1.0,
            add_connection_rate=1.0,
        ),
        **kwargs,
    )


@pytest.mark.parametrize("task", ["xor", "circle", "spiral"])
def test_data_deterministic_with_independent_splits(task):
    config = small_config(task=task)
    a, b = make_dataset(config), make_dataset(config)
    np.testing.assert_array_equal(a.x_train, b.x_train)
    assert not np.array_equal(a.x_train[:16], a.x_val)
    assert a.x_train.dtype == np.float32
    assert set(np.unique(a.y_train)) == {0, 1}
    x, y = sample(task, 17, np.random.default_rng(2), noise=0)
    assert x.shape == (17, 2) and y.shape == (17,)


@pytest.mark.parametrize(
    "overrides",
    [
        {"max_nodes": 2},
        {"population_size": 1},
        {"train_steps": -1},
        {"mode": "invalid"},
        {"add_node_rate": 1.2},
        {"learning_rate": 0},
        {"mode": "fixed", "max_nodes": 4, "fixed_hidden": 4},
        {"noise": -0.1},
    ],
)
def test_invalid_config_rejected(overrides):
    with pytest.raises(ValueError):
        Config(**overrides)


@pytest.mark.parametrize(
    "overrides",
    [
        {"learning_rate": float("nan")},
        {"l2": float("inf")},
        {"generations": 2.5},
        {"max_nodes": True},
        {"excess_coefficient": -1},
    ],
)
def test_nonfinite_or_mistyped_config_rejected(overrides):
    with pytest.raises(ValueError):
        Config(**overrides)


def test_hybrid_run_deterministic_and_grows():
    config = small_config()
    a, b = run(config, verbose=False), run(config, verbose=False)
    assert a.history == b.history
    assert a.champion.to_dict() == b.champion.to_dict()
    assert a.history[0]["max_hidden_nodes"] == 0
    assert a.history[-1]["max_hidden_nodes"] > 0
    assert 0 <= a.metrics["test_accuracy"] <= 1
    scores = [row["best_so_far_fitness"] for row in a.history]
    assert scores == sorted(scores)


def test_test_labels_cannot_affect_selected_model(monkeypatch):
    import backprop_neat.experiment as experiment

    config = small_config()
    normal = run(config, verbose=False)
    data = make_dataset(config)
    data.y_test = 1 - data.y_test
    monkeypatch.setattr(experiment, "make_dataset", lambda _: data)
    flipped = run(config, verbose=False)
    assert normal.history == flipped.history
    assert normal.champion.to_dict() == flipped.champion.to_dict()
    assert normal.metrics["test_accuracy"] + flipped.metrics["test_accuracy"] == pytest.approx(1)


def test_evolution_mode_does_not_call_backprop(monkeypatch):
    import backprop_neat.experiment as experiment

    def fail(*args, **kwargs):
        raise AssertionError("evolution mode must not train by gradient")

    monkeypatch.setattr(experiment, "train_population", fail)
    result = run(small_config(mode="evolution"), verbose=False)
    assert len(result.history) == 4
    assert result.history[-1]["max_hidden_nodes"] > 0


def test_fixed_mode_keeps_topology_constant():
    result = run(small_config(mode="fixed"), verbose=False)
    assert all(row["max_hidden_nodes"] == 3 for row in result.history)
    assert all(row["mean_connections"] == 9 for row in result.history)


def test_output_artifacts_and_cli_inference(tmp_path):
    output = tmp_path / "experiment"
    result = run(small_config(generations=2), output, verbose=False)
    expected = {
        "config.json",
        "metrics.json",
        "champion.json",
        "environment.json",
        "history.csv",
        "dataset.npz",
        "decision_boundary.png",
        "learning_curves.png",
        "topology.png",
    }
    assert {p.name for p in output.iterdir()} == expected
    assert all((output / name).stat().st_size > 0 for name in expected)
    with pytest.raises(FileExistsError):
        run(small_config(), output, verbose=False)
    restored = Genome.from_dict(json.loads((output / "champion.json").read_text()))
    np.testing.assert_allclose(
        predict(restored, result.dataset.x_test), predict(result.champion, result.dataset.x_test)
    )
    points = tmp_path / "input.csv"
    points.write_text("x1,x2\n-1,1\n")
    predictions = tmp_path / "predictions.csv"
    main(
        [
            "predict",
            "--model",
            str(output / "champion.json"),
            "--input",
            str(points),
            "--output",
            str(predictions),
        ]
    )
    assert predictions.read_text().startswith("x1,x2,probability,label\n")
    assert len(predictions.read_text().splitlines()) == 2


def test_cli_overrides_and_benchmark_summary(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps(small_config(generations=1).to_dict()))
    output = tmp_path / "benchmark"
    main(
        [
            "benchmark",
            "--config",
            str(config),
            "--generations",
            "2",
            "--tasks",
            "xor",
            "--modes",
            "evolution",
            "--seeds",
            "0",
            "--output",
            str(output),
        ]
    )
    saved_config = json.loads((output / "xor/evolution/seed_0/config.json").read_text())
    assert saved_config["generations"] == 2
    summary = json.loads((output / "summary.json").read_text())
    assert summary[0]["runs"] == 1
    assert summary[0]["test_accuracy_std"] is None
    assert (output / "results.csv").is_file()
