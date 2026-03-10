"""
Hybrid/mathematical metaheuristic optimizers (pseudo-implementations).
GBO, NNO, PO, PFO, ESMO
"""

import numpy as np
from .base import BaseOptimizer


class GradientBasedOptimizer(BaseOptimizer):
    """GBO — Gradient-Based Optimizer.
    Gradient Search Rule (GSR) + Local Escaping Operator (LEO).
    Direction-based movement without true gradient computation.
    """

    def _iterate(self, t):
        beta_min, beta_max = 0.2, 1.2
        pr = 0.5  # probability of LEO

        for i in range(self.pop_size):
            # GSR — Gradient Search Rule
            r1, r2, r3, r4 = np.random.rand(4)
            epsilon = 5e-3 * np.random.rand()

            # Pick two random agents
            p1 = np.random.randint(self.pop_size)
            p2 = np.random.randint(self.pop_size)

            delta = 2 * r1 * abs(self.population[p1] - self.population[p2])
            step = 2 * delta * (self.population[i] - self.best_pos) / (
                self.fitness[i] - self.best_fit + epsilon
            )

            ro = 2 * r2 * (self.best_pos - self.population[i])

            # Direction of movement
            beta = beta_min + (beta_max - beta_min) * (1 - (t / self.max_iter) ** 3) ** 2
            x_new = self.population[i] - beta * step + r3 * ro

            # LEO — Local Escaping Operator
            if np.random.rand() < pr:
                k = np.random.randint(self.pop_size)
                if self.is_better(self.fitness[k], self.fitness[i]):
                    x_new += r4 * (self.population[k] - self.population[i])
                else:
                    x_new += r4 * (self.population[i] - self.population[k])

            x_new = self.clip(x_new)
            new_fit = self.evaluate(x_new)
            if self.is_better(new_fit, self.fitness[i]):
                self.population[i] = x_new
                self.fitness[i] = new_fit


class NeuralNetworkOptimizer(BaseOptimizer):
    """NNO — Neural Network Optimizer.
    Population modeled as network of neurons.
    Weights updated based on fitness signals.
    Transfer function controls activation.
    """

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def _iterate(self, t):
        w = 2 * (1 - t / self.max_iter)  # weight decay

        for i in range(self.pop_size):
            # Input layer: random connections to other agents
            n_connections = max(2, self.pop_size // 5)
            connections = np.random.choice(self.pop_size, n_connections, replace=False)

            # Hidden layer: weighted sum of differences
            net_input = np.zeros(self.dim)
            for j in connections:
                weight = 1.0 / (1.0 + abs(self.fitness[j] - self.best_fit) + 1e-10)
                net_input += weight * (self.population[j] - self.population[i])
            net_input /= n_connections

            # Activation: transfer function
            activation = self._sigmoid(w * net_input)

            # Output: update position
            bias = np.random.randn(self.dim) * 0.1
            new_pos = self.population[i] + activation * (self.best_pos - self.population[i]) + bias

            new_pos = self.clip(new_pos)
            new_fit = self.evaluate(new_pos)
            if self.is_better(new_fit, self.fitness[i]):
                self.population[i] = new_pos
                self.fitness[i] = new_fit


class PoliticalOptimizer(BaseOptimizer):
    """PO — Political Optimizer.
    Constituency allocation + election campaigns + party switching.
    Multiple parties compete for best positions.
    """

    def __init__(self, *args, n_parties=3, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_parties = n_parties

    def _iterate(self, t):
        party_size = self.pop_size // self.n_parties

        # Find party leaders (best in each party)
        leaders = []
        for p in range(self.n_parties):
            start = p * party_size
            end = min(start + party_size, self.pop_size)
            party_fit = self.fitness[start:end]
            best_local = np.argmin(party_fit) if self.minimize else np.argmax(party_fit)
            leaders.append(start + best_local)

        for p in range(self.n_parties):
            start = p * party_size
            end = min(start + party_size, self.pop_size)
            leader = leaders[p]

            for i in range(start, end):
                if i == leader:
                    continue

                # Election campaign: move toward party leader
                r1 = np.random.rand(self.dim)
                new_pos = self.population[i] + r1 * (
                    self.population[leader] - self.population[i]
                )

                # Constituency allocation: influence from global best
                r2 = np.random.rand(self.dim)
                new_pos += r2 * (self.best_pos - self.population[i]) * (1 - t / self.max_iter)

                new_pos = self.clip(new_pos)
                new_fit = self.evaluate(new_pos)
                if self.is_better(new_fit, self.fitness[i]):
                    self.population[i] = new_pos
                    self.fitness[i] = new_fit

        # Party switching: worst member switches to best party
        if np.random.rand() < 0.1:
            worst_idx = np.argmax(self.fitness) if self.minimize else np.argmin(self.fitness)
            best_leader = leaders[0]
            for l in leaders:
                if self.is_better(self.fitness[l], self.fitness[best_leader]):
                    best_leader = l
            # Move worst toward best party leader
            r = np.random.rand(self.dim)
            self.population[worst_idx] = self.population[best_leader] + r * 0.1 * (
                self.ub - self.lb
            )
            self.population[worst_idx] = self.clip(self.population[worst_idx])
            self.fitness[worst_idx] = self.evaluate(self.population[worst_idx])


class PathfinderOptimizer(BaseOptimizer):
    """PFO — Pathfinder Optimizer.
    Leader (pathfinder) + followers.
    Leader explores, followers exploit leader's path.
    """

    def _iterate(self, t):
        # Identify leader (best solution)
        leader_idx = np.argmin(self.fitness) if self.minimize else np.argmax(self.fitness)
        leader = self.population[leader_idx].copy()

        # Leader update: explores new territory
        r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
        A = 1 - t / self.max_iter  # decreasing amplitude
        prev_leader = leader.copy()

        # Leader vibration (exploration)
        u = np.random.uniform(-1, 1, self.dim)
        new_leader = leader + A * u * (leader - self.population.mean(axis=0))
        new_leader = self.clip(new_leader)
        new_fit = self.evaluate(new_leader)
        if self.is_better(new_fit, self.fitness[leader_idx]):
            self.population[leader_idx] = new_leader
            self.fitness[leader_idx] = new_fit
            leader = new_leader

        # Follower update
        for i in range(self.pop_size):
            if i == leader_idx:
                continue

            r3 = np.random.rand(self.dim)
            r4 = np.random.rand(self.dim)

            # Follow leader + neighbor interaction
            j = np.random.randint(self.pop_size)
            new_pos = (
                self.population[i]
                + r3 * (leader - self.population[i])
                + r4 * A * (self.population[j] - self.population[i])
            )

            new_pos = self.clip(new_pos)
            new_fit = self.evaluate(new_pos)
            if self.is_better(new_fit, self.fitness[i]):
                self.population[i] = new_pos
                self.fitness[i] = new_fit


class EquilibriumSlimeMouldOptimizer(BaseOptimizer):
    """ESMO — Equilibrium Slime Mould Optimizer.
    Hybrid: Equilibrium Optimizer + Slime Mould Algorithm.
    Concentration-based position update + slime oscillation.
    """

    def _iterate(self, t):
        # Equilibrium pool: top 4 + average
        sorted_idx = np.argsort(self.fitness) if self.minimize else np.argsort(-self.fitness)
        pool = [self.population[sorted_idx[k]].copy() for k in range(min(4, self.pop_size))]
        pool.append(np.mean(pool, axis=0))

        # Equilibrium parameters
        a1, a2 = 2, 1
        GP = 0.5  # generation probability
        t_ratio = (1 - t / self.max_iter) ** (a2 * t / self.max_iter)

        for i in range(self.pop_size):
            # Select random equilibrium candidate
            eq = pool[np.random.randint(len(pool))]

            # Exponential decay term
            lambda_vec = np.random.rand(self.dim)
            F_vec = a1 * np.sign(np.random.rand(self.dim) - 0.5) * (
                np.exp(-lambda_vec * t_ratio) - 1
            )

            # Generation rate
            if np.random.rand() < GP:
                GCP = 0.5 * np.random.rand(self.dim)
            else:
                GCP = np.zeros(self.dim)
            G = GCP * (eq - lambda_vec * self.population[i])

            # Slime mould oscillation component
            a_slime = np.arctanh(np.clip(1 - t / self.max_iter, -0.999, 0.999))
            r_slime = np.random.rand()
            if r_slime < 0.5:
                # Oscillation toward best
                vb = np.random.uniform(-a_slime, a_slime, self.dim)
                W = 1 + np.random.rand() * np.log10(
                    abs(self.best_fit - self.fitness[i]) / (abs(self.best_fit) + 1e-10) + 1
                )
                slime_term = vb * W * (self.best_pos - self.population[i])
            else:
                slime_term = np.zeros(self.dim)

            # Combined update
            new_pos = eq + (self.population[i] - eq) * F_vec + G / (lambda_vec + 1e-10) + slime_term * 0.3

            new_pos = self.clip(new_pos)
            new_fit = self.evaluate(new_pos)
            if self.is_better(new_fit, self.fitness[i]):
                self.population[i] = new_pos
                self.fitness[i] = new_fit
