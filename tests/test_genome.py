import json

import numpy as np
import pytest

from backprop_neat.config import Config
from backprop_neat.evolution import SpeciesSet
from backprop_neat.genome import (
    Genome,
    InnovationTracker,
    add_connection,
    add_node,
    compatibility,
    crossover,
    minimal_genome,
    mutate,
)


def test_historical_markings_reused():
    tracker = InnovationTracker()
    a = minimal_genome(tracker, np.random.default_rng(1))
    b = a.clone()
    assert tracker.connection(0, 2) == tracker.connection(0, 2)
    assert add_node(a, tracker, np.random.default_rng(3), 8)
    assert add_node(b, tracker, np.random.default_rng(3), 8)
    assert set(a.nodes) == set(b.nodes)
    assert set(a.connections) == set(b.connections)
    assert a.hidden_nodes == 1
    assert a.enabled_connections == 3
    assert sum(not g.enabled for g in a.connections.values()) == 1


def test_capacity_and_connection_saturation():
    tracker, rng = InnovationTracker(), np.random.default_rng(5)
    genome = minimal_genome(tracker, rng)
    assert not add_node(genome, tracker, rng, 3)
    assert not add_connection(genome, tracker, rng)
    assert add_node(genome, tracker, rng, 4)
    assert not add_node(genome, tracker, rng, 4)
    while add_connection(genome, tracker, rng):
        genome.validate(4)
    assert genome.enabled_connections == 5


def test_disabled_connection_reenabled_without_new_innovation():
    tracker, rng = InnovationTracker(), np.random.default_rng(1)
    genome = minimal_genome(tracker, rng)
    genome.connections[0].enabled = False
    assert add_connection(genome, tracker, rng)
    assert genome.connections[0].enabled
    assert len(genome.connections) == 2


def test_random_mutation_and_crossover_preserve_dag():
    rng, tracker = np.random.default_rng(123), InnovationTracker()
    config = Config(max_nodes=14, add_node_rate=0.9, add_connection_rate=0.9)
    population = [minimal_genome(tracker, rng) for _ in range(10)]
    for _ in range(150):
        a, b = (population[int(rng.integers(len(population)))] for _ in range(2))
        a.fitness, b.fitness = float(rng.normal()), float(rng.normal())
        child = crossover(a, b, rng, config.max_nodes)
        mutate(child, tracker, rng, config)
        child.validate(config.max_nodes)
        population[int(rng.integers(len(population)))] = child
    assert max(g.hidden_nodes for g in population) > 0


def test_distance_symmetric_and_zero_on_clone():
    tracker, rng = InnovationTracker(), np.random.default_rng(1)
    a = minimal_genome(tracker, rng)
    b = a.clone()
    config = Config()
    assert compatibility(a, b, config) == 0
    add_node(b, tracker, rng, 10)
    assert compatibility(a, b, config) > 0
    assert compatibility(a, b, config) == compatibility(b, a, config)


def test_crossover_fitter_parent_controls_disjoint_genes():
    tracker, rng = InnovationTracker(), np.random.default_rng(2)
    a = minimal_genome(tracker, rng)
    b = a.clone()
    add_node(b, tracker, rng, 10)
    a.fitness, b.fitness = 1, 2
    child = crossover(a, b, rng, 10)
    assert set(child.connections) == set(b.connections)
    assert set(child.nodes) == set(b.nodes)
    child.nodes[2].bias = 123
    assert b.nodes[2].bias != 123


def test_serialization_roundtrip_and_invalid_edges():
    tracker, rng = InnovationTracker(), np.random.default_rng(4)
    genome = minimal_genome(tracker, rng)
    add_node(genome, tracker, rng, 10)
    genome.fitness = -0.2
    restored = Genome.from_dict(json.loads(json.dumps(genome.to_dict(), allow_nan=False)))
    assert restored.to_dict() == genome.to_dict()
    next(iter(restored.connections.values())).source = 2
    with pytest.raises(ValueError, match="feedforward"):
        restored.validate()


def test_exact_reproduction_and_champion_preservation():
    config = Config(population_size=12, max_nodes=8)
    tracker, rng = InnovationTracker(), np.random.default_rng(8)
    population = [minimal_genome(tracker, rng) for _ in range(config.population_size)]
    for i, genome in enumerate(population):
        genome.fitness = -i / 10
    species = SpeciesSet()
    species.assign(population, 0, config)
    offspring = species.reproduce(tracker, rng, config, 0)
    assert len(offspring) == config.population_size
    assert offspring[0].to_dict() == population[0].to_dict()
    for genome in offspring:
        genome.validate(config.max_nodes)
    species.assign(population, 1, config)
    assert species.species[0].last_improved == 0


def test_stagnant_species_removed_but_champion_species_protected():
    config = Config(population_size=6, compatibility_threshold=0.001, stagnation_generations=1)
    tracker, rng = InnovationTracker(), np.random.default_rng(21)
    population = [minimal_genome(tracker, rng) for _ in range(6)]
    for index, genome in enumerate(population):
        genome.fitness = -float(index)
    species = SpeciesSet()
    species.assign(population, 0, config)
    assert len(species.species) > 1
    offspring = species.reproduce(tracker, rng, config, 3)
    assert len(offspring) == 6
    assert offspring[0].fitness == 0
    assert all(g.hidden_nodes == 0 for g in offspring[:2])
