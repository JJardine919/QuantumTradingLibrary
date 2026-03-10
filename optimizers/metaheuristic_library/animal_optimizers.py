"""
Animal-inspired metaheuristic optimizers (pseudo-implementations).
HBO, GWO, WO, SSO, CHHO, CO, BO
"""

import numpy as np
from .base import BaseOptimizer


class HoneyBadgerOptimizer(BaseOptimizer):
    """HBO — Honey Badger Optimizer.
    Phases: digging mode (exploitation) + honey mode (exploration).
    Dynamic decreasing factor controls transition.
    """

    def _iterate(self, t):
        C = 2  # constant
        beta = 6  # ability of getting food
        alpha = (1 - t / self.max_iter) * C  # decreasing factor

        for i in range(self.pop_size):
            r = np.random.rand()
            F = 1 if np.random.rand() < 0.5 else -1  # flag

            if r < 0.5:
                # Digging phase (exploitation)
                r3, r4 = np.random.rand(), np.random.rand()
                di = self.best_pos - self.population[i]
                S_i = F * alpha * di  # step
                self.population[i] = self.best_pos + S_i + beta * r3 * np.abs(di)
            else:
                # Honey phase (exploration)
                r5 = np.random.rand()
                j = np.random.randint(self.pop_size)
                di = self.best_pos - self.population[j]
                self.population[i] = self.best_pos + F * alpha * r5 * np.abs(di)

            self.population[i] = self.clip(self.population[i])
            new_fit = self.evaluate(self.population[i])
            if self.is_better(new_fit, self.fitness[i]):
                self.fitness[i] = new_fit


class GreyWolfOptimizer(BaseOptimizer):
    """GWO — Grey Wolf Optimizer.
    Hierarchy: alpha > beta > delta > omega.
    Encircling prey + hunting with linearly decreasing 'a'.
    """

    def _iterate(self, t):
        a = 2 - 2 * t / self.max_iter  # linearly decreasing from 2 to 0

        # Sort to find alpha, beta, delta
        sorted_idx = np.argsort(self.fitness) if self.minimize else np.argsort(-self.fitness)
        alpha_pos = self.population[sorted_idx[0]].copy()
        beta_pos = self.population[sorted_idx[1]].copy()
        delta_pos = self.population[sorted_idx[2]].copy()

        for i in range(self.pop_size):
            for d in range(self.dim):
                # Encircling: A = 2*a*r1 - a, C = 2*r2
                r1, r2 = np.random.rand(2)
                A1, C1 = 2 * a * r1 - a, 2 * r2
                D_alpha = abs(C1 * alpha_pos[d] - self.population[i][d])
                X1 = alpha_pos[d] - A1 * D_alpha

                r1, r2 = np.random.rand(2)
                A2, C2 = 2 * a * r1 - a, 2 * r2
                D_beta = abs(C2 * beta_pos[d] - self.population[i][d])
                X2 = beta_pos[d] - A2 * D_beta

                r1, r2 = np.random.rand(2)
                A3, C3 = 2 * a * r1 - a, 2 * r2
                D_delta = abs(C3 * delta_pos[d] - self.population[i][d])
                X3 = delta_pos[d] - A3 * D_delta

                self.population[i][d] = (X1 + X2 + X3) / 3.0

            self.population[i] = self.clip(self.population[i])
            self.fitness[i] = self.evaluate(self.population[i])


class WhaleOptimizer(BaseOptimizer):
    """WOA — Whale Optimization Algorithm.
    Bubble-net hunting: shrinking encircling + spiral update.
    """

    def _iterate(self, t):
        a = 2 - 2 * t / self.max_iter
        b = 1  # spiral shape constant

        for i in range(self.pop_size):
            r = np.random.rand()
            A = 2 * a * np.random.rand() - a
            C = 2 * np.random.rand()
            p = np.random.rand()

            if p < 0.5:
                if abs(A) < 1:
                    # Shrinking encircling (exploitation)
                    D = abs(C * self.best_pos - self.population[i])
                    self.population[i] = self.best_pos - A * D
                else:
                    # Search for prey (exploration)
                    j = np.random.randint(self.pop_size)
                    D = abs(C * self.population[j] - self.population[i])
                    self.population[i] = self.population[j] - A * D
            else:
                # Spiral update (bubble-net)
                D = abs(self.best_pos - self.population[i])
                l = np.random.uniform(-1, 1)
                self.population[i] = D * np.exp(b * l) * np.cos(2 * np.pi * l) + self.best_pos

            self.population[i] = self.clip(self.population[i])
            self.fitness[i] = self.evaluate(self.population[i])


class SharkSmellOptimizer(BaseOptimizer):
    """SSO — Shark Smell Optimizer.
    Velocity update based on smell concentration gradient.
    Forward movement + rotational movement.
    """

    def _iterate(self, t):
        eta = 1 - t / self.max_iter  # decreasing step

        for i in range(self.pop_size):
            # Forward movement toward stronger smell (best)
            r1 = np.random.rand(self.dim)
            velocity = eta * r1 * (self.best_pos - self.population[i])
            new_pos = self.population[i] + velocity

            # Rotational movement (local search)
            r2 = np.random.rand(self.dim)
            angle = 2 * np.pi * np.random.rand()
            rotation = eta * r2 * np.cos(angle) * (self.best_pos - new_pos)
            new_pos = new_pos + rotation

            new_pos = self.clip(new_pos)
            new_fit = self.evaluate(new_pos)
            if self.is_better(new_fit, self.fitness[i]):
                self.population[i] = new_pos
                self.fitness[i] = new_fit


class ChaoticHarrisHawkOptimizer(BaseOptimizer):
    """CHHO — Chaotic Harris Hawk Optimizer.
    Cooperative hunting with chaotic maps replacing random parameters.
    Phases: exploration (perch) + exploitation (surprise pounce).
    """

    def _chaotic_map(self, x):
        """Logistic chaotic map."""
        return 4.0 * x * (1.0 - x)

    def _iterate(self, t):
        E0 = 2 * np.random.rand() - 1  # initial energy
        E = 2 * E0 * (1 - t / self.max_iter)  # escaping energy
        chaos_val = self._chaotic_map(np.random.rand())

        for i in range(self.pop_size):
            q = chaos_val  # chaotic instead of random

            if abs(E) >= 1:
                # Exploration: perch based on random tall tree
                if q >= 0.5:
                    j = np.random.randint(self.pop_size)
                    self.population[i] = self.population[j] - chaos_val * abs(
                        self.population[j] - 2 * np.random.rand() * self.population[i]
                    )
                else:
                    self.population[i] = (
                        self.best_pos - self.population.mean(axis=0)
                    ) - chaos_val * (self.lb + np.random.rand() * (self.ub - self.lb))
            else:
                # Exploitation: surprise pounce
                r = np.random.rand()
                if r >= 0.5 and abs(E) >= 0.5:
                    # Soft besiege
                    J = 2 * (1 - chaos_val)
                    delta = self.best_pos - self.population[i]
                    self.population[i] = delta - E * abs(J * self.best_pos - self.population[i])
                elif r >= 0.5 and abs(E) < 0.5:
                    # Hard besiege
                    self.population[i] = self.best_pos - E * abs(self.best_pos - self.population[i])
                elif r < 0.5 and abs(E) >= 0.5:
                    # Soft besiege with progressive rapid dive
                    J = 2 * (1 - chaos_val)
                    Y = self.best_pos - E * abs(J * self.best_pos - self.population[i])
                    Y = self.clip(Y)
                    if self.is_better(self.evaluate(Y), self.fitness[i]):
                        self.population[i] = Y
                    else:
                        # Levy flight
                        Z = Y + np.random.randn(self.dim) * np.random.rand(self.dim)
                        Z = self.clip(Z)
                        if self.is_better(self.evaluate(Z), self.fitness[i]):
                            self.population[i] = Z
                else:
                    # Hard besiege with progressive rapid dive
                    J = 2 * (1 - chaos_val)
                    Y = self.best_pos - E * abs(J * self.best_pos - self.population.mean(axis=0))
                    Y = self.clip(Y)
                    if self.is_better(self.evaluate(Y), self.fitness[i]):
                        self.population[i] = Y
                    else:
                        Z = Y + np.random.randn(self.dim) * np.random.rand(self.dim)
                        Z = self.clip(Z)
                        if self.is_better(self.evaluate(Z), self.fitness[i]):
                            self.population[i] = Z

            self.population[i] = self.clip(self.population[i])
            self.fitness[i] = self.evaluate(self.population[i])


class CoyoteOptimizer(BaseOptimizer):
    """CO — Coyote Optimizer.
    Social condition of coyote packs.
    Cultural exchange between packs.
    """

    def __init__(self, *args, n_packs=5, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_packs = n_packs

    def _iterate(self, t):
        pack_size = self.pop_size // self.n_packs

        for p in range(self.n_packs):
            start = p * pack_size
            end = start + pack_size
            pack = self.population[start:end]
            pack_fit = self.fitness[start:end]

            # Alpha = best in pack
            alpha_idx = np.argmin(pack_fit) if self.minimize else np.argmax(pack_fit)
            alpha = pack[alpha_idx].copy()

            # Social tendency = median of pack
            tendency = np.median(pack, axis=0)

            for i in range(len(pack)):
                r1, r2 = np.random.rand(), np.random.rand()
                new_pos = pack[i] + r1 * (alpha - pack[i]) + r2 * (tendency - pack[i])
                new_pos = self.clip(new_pos)
                new_fit = self.evaluate(new_pos)
                if self.is_better(new_fit, pack_fit[i]):
                    self.population[start + i] = new_pos
                    self.fitness[start + i] = new_fit

        # Cultural exchange: swap random coyotes between packs
        if np.random.rand() < 0.1:
            p1, p2 = np.random.choice(self.n_packs, 2, replace=False)
            i1 = p1 * pack_size + np.random.randint(pack_size)
            i2 = p2 * pack_size + np.random.randint(pack_size)
            self.population[[i1, i2]] = self.population[[i2, i1]]
            self.fitness[[i1, i2]] = self.fitness[[i2, i1]]


class BonoboOptimizer(BaseOptimizer):
    """BO — Bonobo Optimizer.
    Social groups with positive/negative phases.
    Alpha-guided mating strategy.
    """

    def _iterate(self, t):
        p = 1 - t / self.max_iter  # positive phase probability decreases

        # Alpha = current best
        alpha_idx = np.argmin(self.fitness) if self.minimize else np.argmax(self.fitness)
        alpha = self.population[alpha_idx].copy()

        for i in range(self.pop_size):
            j = np.random.randint(self.pop_size)
            while j == i:
                j = np.random.randint(self.pop_size)

            if np.random.rand() < p:
                # Positive phase (exploitation): move toward alpha
                r = np.random.rand(self.dim)
                new_pos = self.population[i] + r * (alpha - self.population[i])
            else:
                # Negative phase (exploration): interact with random member
                r = np.random.rand(self.dim)
                if self.is_better(self.fitness[j], self.fitness[i]):
                    new_pos = self.population[i] + r * (self.population[j] - self.population[i])
                else:
                    new_pos = self.population[i] - r * (self.population[j] - self.population[i])

            # Mutation
            if np.random.rand() < 0.05:
                mut_dim = np.random.randint(self.dim)
                new_pos[mut_dim] = self.lb[mut_dim] + np.random.rand() * (
                    self.ub[mut_dim] - self.lb[mut_dim]
                )

            new_pos = self.clip(new_pos)
            new_fit = self.evaluate(new_pos)
            if self.is_better(new_fit, self.fitness[i]):
                self.population[i] = new_pos
                self.fitness[i] = new_fit
