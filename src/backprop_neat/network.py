"""Static-capacity, differentiable feedforward graphs batched with JAX.

Each genome is packed in topological order. Capacity is a storage bound, not a
fixed architecture: edge and node masks determine the actual network. Packing
avoids recompiling for every newly evolved graph.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .genome import Genome


def pack(genome: Genome, capacity: int):
    genome.validate(capacity)
    ordered = genome.ordered_nodes()
    slots = {node.key: index for index, node in enumerate(ordered)}
    weights = np.zeros((capacity, capacity), dtype=np.float32)
    edge_mask = np.zeros_like(weights)
    biases = np.zeros(capacity, dtype=np.float32)
    node_mask = np.zeros(capacity, dtype=np.float32)
    for node in ordered:
        biases[slots[node.key]] = node.bias
        node_mask[slots[node.key]] = node.kind != "input"
    for gene in genome.connections.values():
        if gene.enabled:
            i, j = slots[gene.source], slots[gene.target]
            weights[i, j] = gene.weight
            edge_mask[i, j] = 1
    return (
        {"weights": weights, "biases": biases},
        {"edges": edge_mask, "nodes": node_mask, "output": np.int32(slots[2])},
    )


def pack_population(population: list[Genome], capacity: int):
    pairs = [pack(genome, capacity) for genome in population]
    params = jax.tree.map(lambda *leaves: jnp.asarray(np.stack(leaves)), *(p[0] for p in pairs))
    masks = jax.tree.map(lambda *leaves: jnp.asarray(np.stack(leaves)), *(p[1] for p in pairs))
    return params, masks


def unpack_population(population: list[Genome], params):
    host = jax.device_get(params)
    for index, genome in enumerate(population):
        slots = {node.key: i for i, node in enumerate(genome.ordered_nodes())}
        for node in genome.nodes.values():
            if node.kind != "input":
                node.bias = float(host["biases"][index, slots[node.key]])
        for gene in genome.connections.values():
            if gene.enabled:
                gene.weight = float(host["weights"][index, slots[gene.source], slots[gene.target]])


@jax.jit
def forward(params, masks, x):
    """Return raw output logits for one genome and a batch of input samples."""
    capacity = params["biases"].shape[0]
    values = jnp.zeros((x.shape[0], capacity), dtype=x.dtype).at[:, :2].set(x)
    weights = params["weights"] * masks["edges"]

    def evaluate_node(index, activations):
        z = activations @ weights[:, index] + params["biases"][index]
        # Output remains a logit; hidden nodes use tanh.
        value = jnp.where(index == masks["output"], z, jnp.tanh(z))
        return activations.at[:, index].set(value * masks["nodes"][index])

    values = jax.lax.fori_loop(2, capacity, evaluate_node, values)
    return values[:, masks["output"]]


def objective(params, masks, x, y, l2):
    logits = forward(params, masks, x)
    bce = jnp.mean(optax.sigmoid_binary_cross_entropy(logits, y))
    count = jnp.maximum(1, masks["edges"].sum() + masks["nodes"].sum())
    penalty = (
        jnp.sum((params["weights"] * masks["edges"]) ** 2)
        + jnp.sum((params["biases"] * masks["nodes"]) ** 2)
    ) / count
    return bce + l2 * penalty


@partial(jax.jit, static_argnames=("steps", "learning_rate"))
def train_population(params, masks, x, y, *, steps: int, learning_rate: float, l2: float):
    """Independent full-batch Adam updates for each genome; moments reset per generation."""
    optimizer = optax.chain(optax.clip_by_global_norm(5.0), optax.adam(learning_rate))

    def train_one(initial, graph):
        def step(carry, _):
            weights, state = carry
            loss, gradients = jax.value_and_grad(objective)(weights, graph, x, y, l2)
            updates, state = optimizer.update(gradients, state, weights)
            return (optax.apply_updates(weights, updates), state), loss

        (trained, _), _ = jax.lax.scan(step, (initial, optimizer.init(initial)), None, length=steps)
        return trained

    return jax.vmap(train_one)(params, masks)


@jax.jit
def evaluate_population(params, masks, x, y):
    def evaluate_one(weights, graph):
        logits = forward(weights, graph, x)
        loss = jnp.mean(optax.sigmoid_binary_cross_entropy(logits, y))
        accuracy = jnp.mean((logits >= 0) == (y >= 0.5))
        return loss, accuracy

    return jax.vmap(evaluate_one)(params, masks)


@jax.jit
def predict_packed(params, masks, x):
    return jax.nn.sigmoid(forward(params, masks, x))


def predict(genome: Genome, x: np.ndarray, capacity: int | None = None):
    params, masks = pack(genome, capacity or len(genome.nodes))
    return np.asarray(predict_packed(params, masks, jnp.asarray(x, dtype=jnp.float32)))
