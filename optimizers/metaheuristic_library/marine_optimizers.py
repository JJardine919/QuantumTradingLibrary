"""
Marine/aquatic-inspired metaheuristic optimizers (pseudo-implementations).
MRFO, JSO, MPO, TSO
"""

import math
import numpy as np
from .base import BaseOptimizer


class MantaRayForagingOptimizer(BaseOptimizer):
    """MRFO — Manta Rays Foraging Optimizer.
    Three foraging strategies: chain, cyclone, somersault.
    """

    def _iterate(self, t):
        # Sort by fitness for chain foraging
        sorted_idx = np.argsort(self.fitness) if self.minimize else np.argsort(-self.fitness)
        r = np.random.rand()

        for i in range(self.pop_size):
            if r < 0.33:
                # Chain foraging
                if i == 0:
                    # Leader follows best
                    c = 2 * np.random.rand() * (self.best_pos - self.population[sorted_idx[i]])
                    self.population[sorted_idx[i]] += c
                else:
                    # Followers follow predecessor
                    prev = self.population[sorted_idx[i - 1]]
                    c = 2 * np.random.rand() * (prev - self.population[sorted_idx[i]])
                    self.population[sorted_idx[i]] += c

            elif r < 0.66:
                # Cyclone foraging (spiral)
                beta = 2 * np.exp(1) * np.random.rand() * (1 - t / self.max_iter)
                if t / self.max_iter < np.random.rand():
                    # Exploration: reference = random position
                    ref = self.lb + np.random.rand(self.dim) * (self.ub - self.lb)
                else:
                    ref = self.best_pos
                theta = 2 * np.pi * np.random.rand()
                self.population[sorted_idx[i]] = ref + beta * (ref - self.population[sorted_idx[i]]) * np.cos(theta)

            else:
                # Somersault foraging
                S = 2  # somersault factor
                r1, r2 = np.random.rand(), np.random.rand()
                self.population[sorted_idx[i]] += S * (r1 * self.best_pos - r2 * self.population[sorted_idx[i]])

            self.population[sorted_idx[i]] = self.clip(self.population[sorted_idx[i]])
            self.fitness[sorted_idx[i]] = self.evaluate(self.population[sorted_idx[i]])


class JellyfishSearchOptimizer(BaseOptimizer):
    """JSO — Jellyfish Search Optimizer.
    Ocean current movement + jellyfish swarm behavior.
    Time control switches between exploration and exploitation.
    """

    def _iterate(self, t):
        c_t = abs((1 - t / self.max_iter) * (2 * np.random.rand() - 1))  # time control

        # Mean position (ocean current direction)
        mu = self.population.mean(axis=0)

        for i in range(self.pop_size):
            if c_t >= 0.5:
                # Ocean current movement (exploration)
                trend = self.best_pos - 3 * np.random.rand() * mu
                self.population[i] += np.random.rand(self.dim) * trend
            else:
                # Jellyfish swarm (exploitation)
                j = np.random.randint(self.pop_size)
                while j == i:
                    j = np.random.randint(self.pop_size)

                if self.is_better(self.fitness[i], self.fitness[j]):
                    direction = self.population[i] - self.population[j]
                else:
                    direction = self.population[j] - self.population[i]

                step = np.random.rand(self.dim) * direction
                self.population[i] += step

            # Boundary check with passive motion
            if np.random.rand() < 0.1:
                self.population[i] = self.lb + np.random.rand(self.dim) * (self.ub - self.lb)

            self.population[i] = self.clip(self.population[i])
            self.fitness[i] = self.evaluate(self.population[i])


class MarinePredatorOptimizer(BaseOptimizer):
    """MPO — Marine Predator Optimizer.
    Three phases based on velocity ratio:
    Phase 1: Brownian motion (exploration)
    Phase 2: Levy + Brownian (transition)
    Phase 3: Levy flight (exploitation)
    """

    def _levy_flight(self, dim):
        """Levy flight step using Mantegna's algorithm."""
        beta_lev = 1.5
        sigma = (
            math.gamma(1 + beta_lev) * np.sin(np.pi * beta_lev / 2)
            / (math.gamma((1 + beta_lev) / 2) * beta_lev * 2 ** ((beta_lev - 1) / 2))
        ) ** (1 / beta_lev)
        u = np.random.randn(dim) * sigma
        v = np.random.randn(dim)
        return u / (np.abs(v) ** (1 / beta_lev))

    def _iterate(self, t):
        CF = (1 - t / self.max_iter) ** (2 * t / self.max_iter)  # adaptive parameter

        # Construct prey and predator matrices
        # Top half = prey, bottom half = predator
        half = self.pop_size // 2

        if t < self.max_iter / 3:
            # Phase 1: prey moves faster (Brownian)
            for i in range(half):
                R_B = np.random.randn(self.dim)  # Brownian
                step = R_B * (self.best_pos - R_B * self.population[i])
                self.population[i] += 0.5 * step

        elif t < 2 * self.max_iter / 3:
            # Phase 2: mixed — Levy for top half, Brownian for bottom
            for i in range(half):
                R_L = self._levy_flight(self.dim)
                step = R_L * (self.best_pos - R_L * self.population[i])
                self.population[i] += 0.5 * step

            for i in range(half, self.pop_size):
                R_B = np.random.randn(self.dim)
                step = R_B * (R_B * self.best_pos - self.population[i])
                self.population[i] += 0.5 * CF * step

        else:
            # Phase 3: predator moves faster (Levy)
            for i in range(self.pop_size):
                R_L = self._levy_flight(self.dim)
                step = R_L * (R_L * self.best_pos - self.population[i])
                self.population[i] += 0.5 * CF * step

        # FADs effect (marine memory)
        if np.random.rand() < 0.2:
            for i in range(self.pop_size):
                U = np.random.rand(self.dim) < 0.2
                self.population[i] += CF * (self.lb + np.random.rand(self.dim) * (self.ub - self.lb)) * U.astype(float)

        for i in range(self.pop_size):
            self.population[i] = self.clip(self.population[i])
            self.fitness[i] = self.evaluate(self.population[i])


class TunicateSwarmOptimizer(BaseOptimizer):
    """TSO — Tunicate Swarm Optimizer.
    Jet propulsion + swarm intelligence behavior.
    Avoidance of conflicts, movement toward food source.
    """

    def _iterate(self, t):
        # Parameters
        c1 = 2 * np.random.rand()  # social force
        c2 = np.random.rand()
        c3 = np.random.rand()

        # pmin, pmax for jet propulsion
        p_min, p_max = 1, 4

        for i in range(self.pop_size):
            # Jet propulsion forces
            A_vec = np.zeros(self.dim)
            for d in range(self.dim):
                # Avoidance of conflicts
                if c3 >= 0.5:
                    A_vec[d] = (self.ub[d] - self.lb[d]) / p_max + c1
                else:
                    A_vec[d] = -((self.ub[d] - self.lb[d]) / p_max + c1)

            # Gravity + social forces
            G = c2 * (self.best_pos - self.population[i])

            if np.random.rand() >= 0.5:
                # Move toward best (food source)
                self.population[i] = (self.best_pos + A_vec) / 2 + G * np.random.rand(self.dim)
            else:
                # Swarm behavior: follow predecessor
                if i > 0:
                    self.population[i] = (self.population[i] + self.population[i - 1]) / 2
                else:
                    self.population[i] = (self.population[i] + self.best_pos) / 2

            self.population[i] = self.clip(self.population[i])
            self.fitness[i] = self.evaluate(self.population[i])
