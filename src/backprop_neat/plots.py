"""Publication-friendly decision boundaries, learning curves, and graph diagrams."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

from .genome import Genome
from .network import predict


def plot_topology(genome: Genome, path: Path):
    fig, ax = plt.subplots(figsize=(11, 6))
    levels = sorted({node.depth for node in genome.nodes.values()})
    positions = {}
    for depth in levels:
        nodes = [n for n in genome.ordered_nodes() if n.depth == depth]
        for y, node in zip(
            np.linspace(-1, 1, len(nodes)) if len(nodes) > 1 else [0], nodes, strict=True
        ):
            positions[node.key] = (depth * 10, y * 2)
    for edge in genome.connections.values():
        if edge.enabled:
            arrow = FancyArrowPatch(
                positions[edge.source],
                positions[edge.target],
                arrowstyle="-|>",
                mutation_scale=12,
                shrinkA=17,
                shrinkB=17,
                linewidth=0.5 + min(abs(edge.weight), 4) * 0.5,
                color="#2563eb" if edge.weight >= 0 else "#e16842",
                alpha=0.6,
                connectionstyle="arc3,rad=0.08",
            )
            ax.add_patch(arrow)
    colors = {"input": "#c5e5fa", "hidden": "#dbd4fa", "output": "#f8d9b3"}
    for node in genome.ordered_nodes():
        x, y = positions[node.key]
        ax.scatter([x], [y], s=700, color=colors[node.kind], edgecolor="#25324a", zorder=3)
        ax.text(x, y, str(node.key), ha="center", va="center", zorder=4)
    ax.set(
        xlim=(-1, 11),
        ylim=(-3, 3),
        title=(
            f"Selected topology · {genome.hidden_nodes} hidden nodes · "
            f"{genome.enabled_connections} connections"
        ),
    )
    ax.text(
        0.5,
        -0.02,
        "Blue: positive weight    Orange: negative weight    Width: weight magnitude",
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_result(result, output: Path):
    all_x = np.concatenate([result.dataset.x_train, result.dataset.x_val, result.dataset.x_test])
    lo, hi = all_x.min(axis=0) - 0.15, all_x.max(axis=0) + 0.15
    xx, yy = np.meshgrid(np.linspace(lo[0], hi[0], 220), np.linspace(lo[1], hi[1], 220))
    grid = np.column_stack((xx.ravel(), yy.ravel())).astype(np.float32)
    # Chunk prediction to keep graph activation memory bounded for large capacities.
    probabilities = np.concatenate(
        [
            predict(result.champion, chunk, result.config.max_nodes)
            for chunk in np.array_split(grid, 8)
        ]
    ).reshape(xx.shape)
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.contourf(
        xx, yy, probabilities, levels=np.linspace(0, 1, 21), cmap="RdBu_r", alpha=0.8
    )
    if probabilities.min() < 0.5 < probabilities.max():
        ax.contour(xx, yy, probabilities, levels=[0.5], colors="#202020", linewidths=1)
    ax.scatter(
        result.dataset.x_test[:, 0],
        result.dataset.x_test[:, 1],
        c=result.dataset.y_test,
        cmap="RdBu_r",
        vmin=0,
        vmax=1,
        s=13,
        edgecolor="white",
        linewidth=0.3,
        alpha=0.8,
    )
    ax.set(
        xlabel="x₁",
        ylabel="x₂",
        aspect="equal",
        title=f"{result.config.task.title()} · {result.config.mode} · "
        f"test accuracy {result.metrics['test_accuracy']:.1%}",
    )
    fig.colorbar(image, ax=ax, label="P(class 1)")
    fig.tight_layout()
    fig.savefig(output / "decision_boundary.png", dpi=160)
    plt.close(fig)
    history = result.history
    generations = [row["generation"] for row in history]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for split in ("train", "val"):
        axes[0, 0].plot(generations, [r[f"{split}_accuracy"] for r in history], label=split)
        axes[0, 1].plot(generations, [r[f"{split}_loss"] for r in history], label=split)
    axes[0, 0].set(title="Generation champion accuracy", ylim=(0, 1.03), ylabel="Accuracy")
    axes[0, 1].set(title="Generation champion cross entropy", ylabel="BCE")
    for key, label in (
        ("mean_hidden_nodes", "Population mean"),
        ("best_hidden_nodes", "Champion"),
        ("max_hidden_nodes", "Population max"),
    ):
        axes[1, 0].plot(generations, [r[key] for r in history], label=label)
    axes[1, 0].set(title="Architectural growth", ylabel="Hidden nodes")
    axes[1, 1].plot(generations, [r["mean_connections"] for r in history], label="Mean edges")
    axes[1, 1].plot(generations, [r["species"] for r in history], label="Species")
    axes[1, 1].set(title="Connectivity and diversity", ylabel="Count")
    for ax in axes.flat:
        ax.set_xlabel("Generation")
        ax.grid(alpha=0.2)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output / "learning_curves.png", dpi=160)
    plt.close(fig)
    plot_topology(result.champion, output / "topology.png")
