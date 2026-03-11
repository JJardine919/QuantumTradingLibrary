"""
Nature/ecology-inspired metaheuristic optimizers (pseudo-implementations).
GO, BWO, TSA, STSA, FPO, SMO, TGO, IAEO
"""

import math
import numpy as np
from .base import BaseOptimizer


class GrasshopperOptimizer(BaseOptimizer):
    """GO — Grasshopper Optimizer.
    Attraction/repulsion forces between grasshoppers.
    Decreasing comfort zone pushes toward exploitation.
    """

    def _iterate(self, t):
        c_max, c_min = 1.0, 0.00001
        c = c_max - t * (c_max - c_min) / self.max_iter  # decreasing coefficient

        for i in range(self.pop_size):
            S_i = np.zeros(self.dim)
            for j in range(self.pop_size):
                if i == j:
                    continue
                dist = np.linalg.norm(self.population[j] - self.population[i])
                dist = max(dist, 1e-10)
                # s(r) = f * exp(-r/l) - exp(-r)  [attraction-repulsion]
                f, l_param = 0.5, 1.5
                s_val = f * np.exp(-dist / l_param) - np.exp(-dist)
                d_ij = (self.population[j] - self.population[i]) / dist
                S_i += c * s_val * d_ij

            self.population[i] = c * S_i + self.best_pos
            self.population[i] = self.clip(self.population[i])
            self.fitness[i] = self.evaluate(self.population[i])


class BlackWidowOptimizer(BaseOptimizer):
    """BWO — Black Widow Optimizer.
    Mating (crossover) + cannibalism + mutation.
    """

    def _iterate(self, t):
        # Sort population
        sorted_idx = np.argsort(self.fitness) if self.minimize else np.argsort(-self.fitness)

        new_pop = self.population.copy()
        new_fit = self.fitness.copy()

        for i in range(0, self.pop_size - 1, 2):
            p1 = self.population[sorted_idx[i]]
            p2 = self.population[sorted_idx[i + 1]]

            # Mating (array crossover)
            alpha_m = np.random.rand(self.dim)
            child1 = alpha_m * p1 + (1 - alpha_m) * p2
            child2 = (1 - alpha_m) * p1 + alpha_m * p2

            child1 = self.clip(child1)
            child2 = self.clip(child2)
            f1 = self.evaluate(child1)
            f2 = self.evaluate(child2)

            # Cannibalism: keep best of parent+child
            candidates = [(p1, self.fitness[sorted_idx[i]]),
                          (p2, self.fitness[sorted_idx[i + 1]]),
                          (child1, f1), (child2, f2)]
            candidates.sort(key=lambda x: x[1], reverse=not self.minimize)
            new_pop[sorted_idx[i]] = candidates[0][0]
            new_fit[sorted_idx[i]] = candidates[0][1]
            new_pop[sorted_idx[i + 1]] = candidates[1][0]
            new_fit[sorted_idx[i + 1]] = candidates[1][1]

        # Mutation on worst members
        n_mutate = max(1, self.pop_size // 10)
        worst_idx = sorted_idx[-n_mutate:]
        for idx in worst_idx:
            mut_dim = np.random.randint(self.dim)
            new_pop[idx][mut_dim] = self.lb[mut_dim] + np.random.rand() * (
                self.ub[mut_dim] - self.lb[mut_dim]
            )
            new_fit[idx] = self.evaluate(new_pop[idx])

        self.population = new_pop
        self.fitness = new_fit


class TreeSeedAlgorithm(BaseOptimizer):
    """TSA — Tree-Seed Algorithm.
    Trees produce seeds; seeds grow near trees.
    Search transfer parameter controls exploration/exploitation.
    """

    def __init__(self, *args, ST=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.ST = ST  # search tendency

    def _iterate(self, t):
        for i in range(self.pop_size):
            # Generate seeds
            n_seeds = np.random.randint(1, max(2, self.pop_size // 5))

            for _ in range(n_seeds):
                seed = self.population[i].copy()
                j = np.random.randint(self.pop_size)
                while j == i:
                    j = np.random.randint(self.pop_size)

                for d in range(self.dim):
                    if np.random.rand() < self.ST:
                        # Exploit: seed near best tree
                        seed[d] = self.best_pos[d] + np.random.uniform(-1, 1) * (
                            self.best_pos[d] - self.population[j][d]
                        )
                    else:
                        # Explore: seed between trees
                        seed[d] = self.population[i][d] + np.random.uniform(-1, 1) * (
                            self.population[i][d] - self.population[j][d]
                        )

                seed = self.clip(seed)
                seed_fit = self.evaluate(seed)
                if self.is_better(seed_fit, self.fitness[i]):
                    self.population[i] = seed
                    self.fitness[i] = seed_fit


class SineTreeSeedAlgorithm(TreeSeedAlgorithm):
    """STSA — Sine Tree-Seed Algorithm.
    TSA enhanced with sine function for improved exploration.
    """

    def _iterate(self, t):
        for i in range(self.pop_size):
            n_seeds = np.random.randint(1, max(2, self.pop_size // 5))

            for _ in range(n_seeds):
                seed = self.population[i].copy()
                j = np.random.randint(self.pop_size)
                while j == i:
                    j = np.random.randint(self.pop_size)

                # Sine modulation
                theta = np.pi * t / self.max_iter
                sine_factor = np.sin(theta)

                for d in range(self.dim):
                    if np.random.rand() < self.ST:
                        seed[d] = self.best_pos[d] + sine_factor * np.random.uniform(-1, 1) * (
                            self.best_pos[d] - self.population[j][d]
                        )
                    else:
                        seed[d] = self.population[i][d] + sine_factor * np.random.uniform(-1, 1) * (
                            self.population[i][d] - self.population[j][d]
                        )

                seed = self.clip(seed)
                seed_fit = self.evaluate(seed)
                if self.is_better(seed_fit, self.fitness[i]):
                    self.population[i] = seed
                    self.fitness[i] = seed_fit


class FlowerPollinationOptimizer(BaseOptimizer):
    """FPO — Flower Pollination Optimizer.
    Global pollination via Levy flight.
    Local pollination via neighbor interaction.
    Switch probability p controls balance.
    """

    def _levy_flight(self):
        beta = 1.5
        sigma = (
            math.gamma(1 + beta) * np.sin(np.pi * beta / 2)
            / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
        ) ** (1 / beta)
        u = np.random.randn(self.dim) * sigma
        v = np.random.randn(self.dim)
        return u / (np.abs(v) ** (1 / beta))

    def _iterate(self, t):
        p = 0.8  # switch probability

        for i in range(self.pop_size):
            if np.random.rand() < p:
                # Global pollination (Levy flight)
                L = self._levy_flight()
                new_pos = self.population[i] + L * (self.best_pos - self.population[i])
            else:
                # Local pollination
                j, k = np.random.choice(self.pop_size, 2, replace=False)
                epsilon = np.random.rand()
                new_pos = self.population[i] + epsilon * (
                    self.population[j] - self.population[k]
                )

            new_pos = self.clip(new_pos)
            new_fit = self.evaluate(new_pos)
            if self.is_better(new_fit, self.fitness[i]):
                self.population[i] = new_pos
                self.fitness[i] = new_fit


class SlimeMouldOptimizer(BaseOptimizer):
    """SMO — Slime Mould Algorithm.
    Wrapping food behavior with oscillation.
    Approaching food + random search.
    """

    def _iterate(self, t):
        a = np.arctanh(np.clip(1 - t / self.max_iter, -0.999, 0.999))  # decreasing
        b = 1 - t / self.max_iter

        # Sort by fitness and get weight
        sorted_idx = np.argsort(self.fitness) if self.minimize else np.argsort(-self.fitness)
        worst_fit = self.fitness[sorted_idx[-1]]
        best_fit_local = self.fitness[sorted_idx[0]]

        W = np.zeros(self.pop_size)
        half = self.pop_size // 2
        for i in range(self.pop_size):
            rank = np.where(sorted_idx == i)[0][0]
            if rank < half:
                W[i] = 1 + np.random.rand() * np.log10(
                    (best_fit_local - self.fitness[i]) / (best_fit_local - worst_fit + 1e-10) + 1
                )
            else:
                W[i] = 1 - np.random.rand() * np.log10(
                    (best_fit_local - self.fitness[i]) / (best_fit_local - worst_fit + 1e-10) + 1
                )

        for i in range(self.pop_size):
            r = np.random.rand()
            p_val = np.tanh(abs(self.fitness[i] - best_fit_local))

            if r < p_val:
                # Wrapping food
                j, k = np.random.choice(self.pop_size, 2, replace=False)
                vc = np.random.uniform(-a, a, self.dim)
                self.population[i] = self.best_pos + vc * (
                    W[i] * self.population[j] - self.population[k]
                )
            else:
                if np.random.rand() < 0.5:
                    # Approach food
                    self.population[i] = b * self.population[i] + (1 - b) * self.best_pos
                else:
                    # Random search
                    self.population[i] = self.lb + np.random.rand(self.dim) * (self.ub - self.lb)

            self.population[i] = self.clip(self.population[i])
            self.fitness[i] = self.evaluate(self.population[i])


class TreeGrowthOptimizer(BaseOptimizer):
    """TGO — Tree Growth Optimizer.
    Best trees grow toward light (best solution).
    Competition trees grow toward better neighbors.
    Removal and replanting of worst trees.
    """

    def _iterate(self, t):
        sorted_idx = np.argsort(self.fitness) if self.minimize else np.argsort(-self.fitness)

        n_best = max(2, self.pop_size // 4)  # top trees
        n_comp = self.pop_size - n_best  # competition trees

        # Best trees: grow toward global best
        for rank in range(n_best):
            i = sorted_idx[rank]
            lambda_val = np.random.rand()
            theta = 2 * np.pi * np.random.rand()
            self.population[i] += lambda_val * (self.best_pos - self.population[i]) * np.cos(theta)
            self.population[i] = self.clip(self.population[i])
            self.fitness[i] = self.evaluate(self.population[i])

        # Competition trees: grow toward better neighbor or random
        for rank in range(n_best, self.pop_size):
            i = sorted_idx[rank]
            j = sorted_idx[np.random.randint(n_best)]  # random good tree
            r = np.random.rand()
            self.population[i] += r * (self.population[j] - self.population[i])
            self.population[i] = self.clip(self.population[i])
            self.fitness[i] = self.evaluate(self.population[i])

        # Remove and replant worst tree
        worst = sorted_idx[-1]
        self.population[worst] = self.lb + np.random.rand(self.dim) * (self.ub - self.lb)
        self.fitness[worst] = self.evaluate(self.population[worst])


class ImprovedArtificialEcosystemOptimizer(BaseOptimizer):
    """IAEO — Improved Artificial Ecosystem Optimizer.
    Three levels: production (herbivore), consumption (carnivore), decomposition.
    Energy-based transitions.
    """

    def _iterate(self, t):
        sorted_idx = np.argsort(self.fitness) if self.minimize else np.argsort(-self.fitness)
        n_prod = self.pop_size // 3
        n_cons = self.pop_size // 3

        for i in range(self.pop_size):
            rank = np.where(sorted_idx == i)[0][0]
            r1 = np.random.rand()
            E = 2 * (1 - t / self.max_iter)  # energy factor

            if rank < n_prod:
                # Production (herbivore): explore broadly
                x_rand = self.lb + np.random.rand(self.dim) * (self.ub - self.lb)
                self.population[i] = (1 - E) * self.population[i] + E * r1 * (
                    x_rand - self.population[i]
                )

            elif rank < n_prod + n_cons:
                # Consumption (carnivore): exploit prey
                prey = sorted_idx[np.random.randint(n_prod)]
                C = np.random.rand()
                self.population[i] += C * E * (self.population[prey] - self.population[i])

            else:
                # Decomposition: move toward best, add randomness
                D = 3 * np.random.randn(self.dim)
                self.population[i] = self.best_pos + D * E * (
                    self.best_pos - self.population[i]
                ) / (abs(self.best_pos - self.population[i]) + 1e-10)

            self.population[i] = self.clip(self.population[i])
            self.fitness[i] = self.evaluate(self.population[i])
