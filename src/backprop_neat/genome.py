"""Historical genes, acyclic structural mutations, and aligned crossover.

Node depths are immutable historical attributes. Only edges from lower to higher
depth are allowed, so independently evolved parents cannot create a cycle when
crossed. A persistent innovation registry aligns equivalent mutations.
"""

import copy
from dataclasses import asdict, dataclass, field

import numpy as np

from .config import Config


@dataclass
class NodeGene:
    key: int
    kind: str
    depth: float
    bias: float = 0.0


@dataclass
class ConnectionGene:
    innovation: int
    source: int
    target: int
    weight: float
    enabled: bool = True


@dataclass
class Genome:
    nodes: dict[int, NodeGene]
    connections: dict[int, ConnectionGene]
    fitness: float = -float("inf")

    def clone(self):
        return copy.deepcopy(self)

    @property
    def enabled_connections(self):
        return sum(g.enabled for g in self.connections.values())

    @property
    def hidden_nodes(self):
        return sum(n.kind == "hidden" for n in self.nodes.values())

    def ordered_nodes(self):
        return sorted(self.nodes.values(), key=lambda n: (n.depth, n.key))

    def validate(self, max_nodes: int = 10_000):
        if len(self.nodes) > max_nodes:
            raise ValueError("genome exceeds node capacity")
        if sorted(n.key for n in self.nodes.values() if n.kind == "input") != [0, 1]:
            raise ValueError("expected two input nodes with IDs 0 and 1")
        if [n.key for n in self.nodes.values() if n.kind == "output"] != [2]:
            raise ValueError("expected one output node with ID 2")
        pairs = set()
        for key, node in self.nodes.items():
            if key != node.key or node.kind not in {"input", "hidden", "output"}:
                raise ValueError("invalid node gene")
            if not np.isfinite([node.depth, node.bias]).all():
                raise ValueError("non-finite node attribute")
            if node.kind == "input" and node.depth != 0:
                raise ValueError("inputs must have depth zero")
            if node.kind == "output" and node.depth != 1:
                raise ValueError("output must have depth one")
            if node.kind == "hidden" and not 0 < node.depth < 1:
                raise ValueError("hidden node depth must be between zero and one")
        for key, gene in self.connections.items():
            if (
                key != gene.innovation
                or gene.source not in self.nodes
                or gene.target not in self.nodes
            ):
                raise ValueError("invalid connection endpoints or innovation")
            if self.nodes[gene.source].depth >= self.nodes[gene.target].depth:
                raise ValueError("connection violates feedforward order")
            if (gene.source, gene.target) in pairs or not np.isfinite(gene.weight):
                raise ValueError("duplicate edge or non-finite weight")
            pairs.add((gene.source, gene.target))

    def to_dict(self):
        return {
            "nodes": [asdict(n) for n in self.ordered_nodes()],
            "connections": [
                asdict(g) for g in sorted(self.connections.values(), key=lambda g: g.innovation)
            ],
            "fitness": self.fitness if np.isfinite(self.fitness) else None,
        }

    @classmethod
    def from_dict(cls, payload):
        genome = cls(
            {n["key"]: NodeGene(**n) for n in payload["nodes"]},
            {g["innovation"]: ConnectionGene(**g) for g in payload["connections"]},
            payload.get("fitness") if payload.get("fitness") is not None else -float("inf"),
        )
        genome.validate()
        return genome


@dataclass
class InnovationTracker:
    next_node: int = 3
    next_innovation: int = 0
    edges: dict[tuple[int, int], int] = field(default_factory=dict)
    splits: dict[int, int] = field(default_factory=dict)

    def connection(self, source: int, target: int):
        pair = (source, target)
        if pair not in self.edges:
            self.edges[pair] = self.next_innovation
            self.next_innovation += 1
        return self.edges[pair]

    def split(self, innovation: int):
        if innovation not in self.splits:
            self.splits[innovation] = self.next_node
            self.next_node += 1
        return self.splits[innovation]


def minimal_genome(tracker: InnovationTracker, rng: np.random.Generator):
    nodes = {0: NodeGene(0, "input", 0), 1: NodeGene(1, "input", 0), 2: NodeGene(2, "output", 1)}
    connections = {}
    for source in (0, 1):
        innovation = tracker.connection(source, 2)
        connections[innovation] = ConnectionGene(innovation, source, 2, float(rng.normal()))
    return Genome(nodes, connections)


def fixed_genome(tracker: InnovationTracker, rng: np.random.Generator, hidden: int):
    genome = minimal_genome(tracker, rng)
    genome.connections.clear()
    # Shared IDs and topology for every member of the fixed baseline.
    tracker.next_node = max(tracker.next_node, 3 + hidden)
    for key in range(3, 3 + hidden):
        genome.nodes[key] = NodeGene(key, "hidden", 0.5, float(rng.normal(0, 0.1)))
    pairs = (
        (
            [(i, h) for i in (0, 1) for h in range(3, 3 + hidden)]
            + [(h, 2) for h in range(3, 3 + hidden)]
        )
        if hidden
        else [(0, 2), (1, 2)]
    )
    for source, target in pairs:
        innovation = tracker.connection(source, target)
        scale = 1 / np.sqrt(max(1, hidden)) if target == 2 else 1.0
        genome.connections[innovation] = ConnectionGene(
            innovation, source, target, float(rng.normal(0, scale))
        )
    return genome


def add_node(genome: Genome, tracker: InnovationTracker, rng: np.random.Generator, max_nodes: int):
    if len(genome.nodes) >= max_nodes:
        return False
    candidates = [
        g
        for g in genome.connections.values()
        if g.enabled and tracker.splits.get(g.innovation) not in genome.nodes
    ]
    rng.shuffle(candidates)
    for gene in candidates:
        source_depth = genome.nodes[gene.source].depth
        target_depth = genome.nodes[gene.target].depth
        depth = (source_depth + target_depth) / 2
        if not source_depth < depth < target_depth:  # Floating point subdivision limit.
            continue
        key = tracker.split(gene.innovation)
        genome.nodes[key] = NodeGene(key, "hidden", depth)
        gene.enabled = False
        for source, target, weight in ((gene.source, key, 1.0), (key, gene.target, gene.weight)):
            innovation = tracker.connection(source, target)
            genome.connections[innovation] = ConnectionGene(innovation, source, target, weight)
        return True
    return False


def add_connection(genome: Genome, tracker: InnovationTracker, rng: np.random.Generator):
    existing = {(g.source, g.target): g for g in genome.connections.values()}
    candidates = [
        (a.key, b.key)
        for a in genome.nodes.values()
        for b in genome.nodes.values()
        if a.depth < b.depth
        and ((a.key, b.key) not in existing or not existing[a.key, b.key].enabled)
    ]
    if not candidates:
        return False
    source, target = candidates[int(rng.integers(len(candidates)))]
    if (source, target) in existing:
        existing[source, target].enabled = True
    else:
        innovation = tracker.connection(source, target)
        genome.connections[innovation] = ConnectionGene(
            innovation, source, target, float(rng.normal())
        )
    return True


def mutate(genome: Genome, tracker: InnovationTracker, rng: np.random.Generator, config: Config):
    if rng.random() < config.weight_mutation_rate:
        for gene in genome.connections.values():
            if rng.random() < 0.9:
                gene.weight += float(rng.normal(0, config.weight_sigma))
            else:
                gene.weight = float(rng.normal())
        for node in genome.nodes.values():
            if node.kind != "input":
                node.bias += float(rng.normal(0, config.weight_sigma))
    if config.mode != "fixed":
        if rng.random() < config.add_node_rate:
            add_node(genome, tracker, rng, config.max_nodes)
        if rng.random() < config.add_connection_rate:
            add_connection(genome, tracker, rng)
    genome.fitness = -float("inf")
    genome.validate(config.max_nodes)


def compatibility(a: Genome, b: Genome, config: Config):
    a_keys, b_keys = set(a.connections), set(b.connections)
    matching = a_keys & b_keys
    a_max, b_max = max(a_keys, default=-1), max(b_keys, default=-1)
    excess = sum(k > b_max for k in a_keys - b_keys) + sum(k > a_max for k in b_keys - a_keys)
    disjoint = len(a_keys ^ b_keys) - excess
    normalizer = max(len(a_keys), len(b_keys))
    normalizer = normalizer if normalizer >= 20 else 1
    weight_distance = (
        np.mean([abs(a.connections[k].weight - b.connections[k].weight) for k in matching])
        if matching
        else 0.0
    )
    return float(
        (config.excess_coefficient * excess + config.disjoint_coefficient * disjoint) / normalizer
        + config.weight_coefficient * weight_distance
    )


def crossover(a: Genome, b: Genome, rng: np.random.Generator, max_nodes: int):
    if b.fitness > a.fitness or (b.fitness == a.fitness and rng.random() < 0.5):
        a, b = b, a
    # Disjoint/excess genes come from the fitter parent. On ties, choose one
    # primary parent randomly; this bounded variant avoids unbounded unions.
    child = a.clone()
    for key, gene in list(child.connections.items()):
        other = b.connections.get(key)
        if other is not None:
            inherited = copy.deepcopy(gene if rng.random() < 0.5 else other)
            if not gene.enabled or not other.enabled:
                inherited.enabled = bool(rng.random() >= 0.75)
            child.connections[key] = inherited
    for key in child.nodes.keys() & b.nodes.keys():
        if rng.random() < 0.5:
            child.nodes[key].bias = b.nodes[key].bias
    child.fitness = -float("inf")
    child.validate(max_nodes)
    return child
