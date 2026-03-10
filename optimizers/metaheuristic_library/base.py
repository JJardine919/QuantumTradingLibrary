"""
Base Optimizer — Common interface for all metaheuristic optimizers.
Reference: Energy Conversion and Management, Vol 258, 115521 (2022)
"""

import numpy as np
from abc import ABC, abstractmethod


class BaseOptimizer(ABC):
    """Abstract base for all metaheuristic optimizers."""

    def __init__(self, obj_func, dim, lb, ub, pop_size=30, max_iter=100, minimize=True):
        self.obj_func = obj_func
        self.dim = dim
        self.lb = np.array(lb) if np.isscalar(lb) is False else np.full(dim, lb)
        self.ub = np.array(ub) if np.isscalar(ub) is False else np.full(dim, ub)
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.minimize = minimize

        self.population = None
        self.fitness = None
        self.best_pos = None
        self.best_fit = np.inf if minimize else -np.inf
        self.convergence = []

    def initialize(self):
        """Random population within bounds."""
        self.population = self.lb + (self.ub - self.lb) * np.random.rand(self.pop_size, self.dim)
        self.fitness = np.array([self.obj_func(x) for x in self.population])
        self._update_best()

    def _update_best(self):
        if self.minimize:
            idx = np.argmin(self.fitness)
            if self.fitness[idx] < self.best_fit:
                self.best_fit = self.fitness[idx]
                self.best_pos = self.population[idx].copy()
        else:
            idx = np.argmax(self.fitness)
            if self.fitness[idx] > self.best_fit:
                self.best_fit = self.fitness[idx]
                self.best_pos = self.population[idx].copy()

    def clip(self, x):
        """Enforce bounds."""
        return np.clip(x, self.lb, self.ub)

    def evaluate(self, x):
        x = self.clip(x)
        return self.obj_func(x)

    def is_better(self, new_fit, old_fit):
        return new_fit < old_fit if self.minimize else new_fit > old_fit

    @abstractmethod
    def _iterate(self, t):
        """Single iteration update. Override in subclass."""
        pass

    def optimize(self):
        """Run full optimization loop."""
        self.initialize()
        for t in range(self.max_iter):
            self._iterate(t)
            self._update_best()
            self.convergence.append(self.best_fit)
        return self.best_pos, self.best_fit
