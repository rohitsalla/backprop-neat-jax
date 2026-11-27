import jax
import jax.numpy as jnp
import numpy as np

from backprop_neat.genome import InnovationTracker, add_node, fixed_genome, minimal_genome
from backprop_neat.network import (
    evaluate_population,
    forward,
    objective,
    pack,
    pack_population,
    predict,
    train_population,
    unpack_population,
)


def test_forward_matches_numpy_reference():
    rng, tracker = np.random.default_rng(5), InnovationTracker()
    genome = minimal_genome(tracker, rng)
    for _ in range(4):
        add_node(genome, tracker, rng, 10)
    genome.nodes[2].bias = 0.3
    x = rng.normal(size=(9, 2)).astype(np.float32)
    activations = {0: x[:, 0], 1: x[:, 1]}
    for node in genome.ordered_nodes()[2:]:
        z = np.full(len(x), node.bias)
        for gene in genome.connections.values():
            if gene.enabled and gene.target == node.key:
                z += activations[gene.source] * gene.weight
        activations[node.key] = z if node.kind == "output" else np.tanh(z)
    params, masks = pack(genome, 12)
    actual = forward(params, masks, jnp.asarray(x))
    np.testing.assert_allclose(actual, activations[2], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(predict(genome, x), predict(genome, x, 16), atol=1e-6)


def test_disabled_and_padded_parameters_have_zero_gradient():
    rng, tracker = np.random.default_rng(5), InnovationTracker()
    genome = minimal_genome(tracker, rng)
    add_node(genome, tracker, rng, 8)
    params, masks = pack(genome, 8)
    x = jnp.array([[-1, 1], [1, 1]], dtype=jnp.float32)
    grads = jax.grad(objective)(params, masks, x, jnp.array([1.0, 0.0]), 0.01)
    assert np.any(np.abs(np.asarray(grads["weights"])) > 0)
    np.testing.assert_array_equal(np.asarray(grads["weights"]) * (1 - masks["edges"]), 0)
    np.testing.assert_array_equal(np.asarray(grads["biases"]) * (1 - masks["nodes"]), 0)


def test_adam_learns_xor_and_trained_weights_are_inherited():
    rng, tracker = np.random.default_rng(42), InnovationTracker()
    genome = fixed_genome(tracker, rng, 6)
    x = jnp.array([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=jnp.float32)
    y = jnp.array([0, 1, 1, 0], dtype=jnp.float32)
    params, masks = pack_population([genome], 10)
    before, _ = evaluate_population(params, masks, x, y)
    trained = train_population(params, masks, x, y, steps=150, learning_rate=0.05, l2=0.0)
    after, accuracy = evaluate_population(trained, masks, x, y)
    assert float(after[0]) < float(before[0]) * 0.1
    assert float(accuracy[0]) == 1.0
    unpack_population([genome], trained)
    restored, _ = pack_population([genome], 10)
    np.testing.assert_allclose(restored["weights"], trained["weights"], atol=1e-7)
    np.testing.assert_allclose(restored["biases"], trained["biases"], atol=1e-7)
    clone = genome.clone()
    np.testing.assert_allclose(predict(clone, np.asarray(x)), predict(genome, np.asarray(x)))


def test_population_training_is_independent_and_zero_steps_is_identity():
    rng, tracker = np.random.default_rng(9), InnovationTracker()
    a, b = minimal_genome(tracker, rng), fixed_genome(tracker, rng, 3)
    x = jnp.array([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=jnp.float32)
    y = jnp.array([0, 1, 1, 0], dtype=jnp.float32)
    params, masks = pack_population([a, b], 8)
    unchanged = train_population(params, masks, x, y, steps=0, learning_rate=0.03, l2=0.0)
    np.testing.assert_array_equal(unchanged["weights"], params["weights"])
    trained = train_population(params, masks, x, y, steps=5, learning_rate=0.03, l2=0.0)
    single_params, single_masks = pack_population([a], 8)
    single = train_population(
        single_params, single_masks, x, y, steps=5, learning_rate=0.03, l2=0.0
    )
    np.testing.assert_allclose(trained["weights"][0], single["weights"][0], atol=1e-6)


def test_autodiff_agrees_with_finite_difference():
    rng, tracker = np.random.default_rng(14), InnovationTracker()
    params, masks = pack(fixed_genome(tracker, rng, 2), 7)
    params = jax.tree.map(jnp.asarray, params)
    x = jnp.array([[-0.5, 0.8], [0.4, 0.1]], dtype=jnp.float32)
    y = jnp.array([1, 0], dtype=jnp.float32)
    source, target = np.argwhere(masks["edges"])[0]
    gradients = jax.grad(objective)(params, masks, x, y, 0.0)
    epsilon = 0.001
    high = params | {"weights": params["weights"].at[source, target].add(epsilon)}
    low = params | {"weights": params["weights"].at[source, target].add(-epsilon)}
    numeric = (objective(high, masks, x, y, 0.0) - objective(low, masks, x, y, 0.0)) / (2 * epsilon)
    np.testing.assert_allclose(gradients["weights"][source, target], numeric, rtol=0.02, atol=1e-4)
