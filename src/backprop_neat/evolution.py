"""Persistent species, fitness sharing, stagnation, and bounded reproduction."""

from dataclasses import dataclass, field

import numpy as np

from .config import Config
from .genome import Genome, InnovationTracker, compatibility, crossover, mutate


@dataclass
class Species:
    key: int
    representative: Genome
    members: list[Genome] = field(default_factory=list)
    best_fitness: float = -float("inf")
    last_improved: int = 0


class SpeciesSet:
    def __init__(self):
        self.species: list[Species] = []
        self.next_key = 0

    def assign(self, population: list[Genome], generation: int, config: Config):
        for species in self.species:
            species.members = []
        for genome in population:
            for species in self.species:
                if (
                    compatibility(genome, species.representative, config)
                    < config.compatibility_threshold
                ):
                    species.members.append(genome)
                    break
            else:
                species = Species(self.next_key, genome.clone(), [genome], last_improved=generation)
                self.next_key += 1
                self.species.append(species)
        self.species = [species for species in self.species if species.members]
        for species in self.species:
            species.members.sort(key=lambda g: g.fitness, reverse=True)
            species.representative = species.members[0].clone()
            best = species.members[0].fitness
            if best > species.best_fitness + 1e-8:
                species.best_fitness = best
                species.last_improved = generation

    def reproduce(
        self, tracker: InnovationTracker, rng: np.random.Generator, config: Config, generation: int
    ):
        champion = max((g for s in self.species for g in s.members), key=lambda g: g.fitness)
        active = [
            s
            for s in self.species
            if any(g is champion for g in s.members)
            or generation - s.last_improved < config.stagnation_generations
        ]
        # Sum of fitness / species size equals its mean: explicit fitness sharing.
        # A common exponential transform makes negative loss fitness nonnegative.
        best = champion.fitness
        strengths = np.array(
            [
                np.mean(
                    [
                        np.exp(np.clip((g.fitness - best) / config.selection_temperature, -50, 0))
                        for g in s.members
                    ]
                )
                for s in active
            ]
        )
        probabilities = strengths / strengths.sum()
        raw = probabilities * (config.population_size - 1)
        counts = np.floor(raw).astype(int)
        remaining = config.population_size - 1 - int(counts.sum())
        counts[np.argsort(-(raw - counts), kind="stable")[:remaining]] += 1
        offspring = [champion.clone()]
        for species, count in zip(active, counts, strict=True):
            if count == 0:
                continue
            survivors = species.members[
                : max(1, int(np.ceil(len(species.members) * config.survival_fraction)))
            ]
            # Protect the best member in each reproducing species.
            elite_count = int(count >= 2)
            if elite_count:
                offspring.append(survivors[0].clone())
            for _ in range(int(count) - elite_count):
                a = survivors[int(rng.integers(len(survivors)))]
                if len(survivors) > 1 and rng.random() < config.crossover_rate:
                    b = survivors[int(rng.integers(len(survivors)))]
                    child = crossover(a, b, rng, config.max_nodes)
                else:
                    child = a.clone()
                mutate(child, tracker, rng, config)
                offspring.append(child)
        assert len(offspring) == config.population_size
        return offspring
