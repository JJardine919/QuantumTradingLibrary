"""
HBO-Quantum-TE Pipeline
========================
Honey Badger Optimizer strapped with:
  - Full TE (Transposable Element) stack: inner→outer + inverted outer→inner
  - Quantum collapse via aoi_collapse (24D octonion/Leech)
  - Mitochondria energy gating (ATP-based step adaptation)
  - Methyl blue epigenetic memory (prevents re-exploration)
  - Superposition: multiple states collapse to one via Jordan-Shadow

Reference: Energy Conversion and Management Vol 258 (2022) 115521
"""

import numpy as np
from aoi_collapse import aoi_collapse, Octonion, octonion_shadow_decompose, entropy_transponders


# ============================================================
# Transposable Element Layers
# ============================================================

class TELayer:
    """Single Transposable Element — jumps and inserts perturbations."""

    def __init__(self, name, scale, insertion_rate=0.3):
        self.name = name
        self.scale = scale  # perturbation magnitude
        self.insertion_rate = insertion_rate  # probability of TE jumping

    def transpose(self, x, rng):
        """Apply TE insertion: jumps into random dimensions."""
        mask = rng.random(len(x)) < self.insertion_rate
        jump = rng.standard_normal(len(x)) * self.scale
        return x + mask * jump


def build_te_stack():
    """
    Full TE stack — 6 layers from inner (smallest) to outer (largest).
    Models biological hierarchy: SINE < LINE < DNA transposon < LTR < ERV < Giant TE
    """
    return [
        TELayer("SINE",           scale=0.01,  insertion_rate=0.5),   # innermost — fine tuning
        TELayer("LINE",           scale=0.05,  insertion_rate=0.4),   # short interspersed
        TELayer("DNA_transposon", scale=0.10,  insertion_rate=0.3),   # cut-and-paste
        TELayer("LTR_retro",      scale=0.25,  insertion_rate=0.25),  # long terminal repeat
        TELayer("ERV",            scale=0.50,  insertion_rate=0.2),   # endogenous retrovirus
        TELayer("Giant_TE",       scale=1.00,  insertion_rate=0.15),  # outermost — structural
    ]


def apply_te_forward(x, te_stack, rng):
    """Sequential TE application: inner → outer (SINE first, Giant last)."""
    result = x.copy()
    for te in te_stack:
        result = te.transpose(result, rng)
    return result


def apply_te_inverted(x, te_stack, rng):
    """Inverted TE application: outer → inner (Giant first, SINE last).
    Hypothesis: large structural jumps first, then fine-tune inward.
    """
    result = x.copy()
    for te in reversed(te_stack):
        result = te.transpose(result, rng)
    return result


# ============================================================
# Mitochondria — Energy (ATP) Gating
# ============================================================

class Mitochondria:
    """
    ATP-based energy system. Good fitness = more ATP = bigger steps.
    Bad fitness = ATP depleted = conservative steps.
    Electron transport chain modeled as fitness-to-energy conversion.
    """

    def __init__(self, base_atp=1.0, max_atp=5.0, decay=0.95):
        self.base_atp = base_atp
        self.max_atp = max_atp
        self.decay = decay
        self.atp_pool = base_atp

    def produce_atp(self, fitness_delta):
        """
        Electron transport chain: improvement → ATP production.
        fitness_delta < 0 means improvement (for minimization).
        """
        if fitness_delta < 0:
            # Improvement → generate ATP (proton gradient)
            production = min(abs(fitness_delta) * 2.0, self.max_atp - self.atp_pool)
            self.atp_pool += production
        else:
            # No improvement → ATP consumed by maintenance
            self.atp_pool *= self.decay
            self.atp_pool = max(self.atp_pool, 0.1)

    def energy_scale(self):
        """Current energy multiplier for step sizes."""
        return self.atp_pool / self.base_atp

    def oxidative_burst(self):
        """Spend all ATP for one massive exploration step. Returns scale, resets pool."""
        scale = self.atp_pool
        self.atp_pool = self.base_atp * 0.5  # depleted after burst
        return scale


# ============================================================
# Methyl Blue — Epigenetic Memory
# ============================================================

class MethylBlue:
    """
    Methylation memory: marks visited regions of search space.
    Methylated regions get repulsion (prevents re-exploration).
    Blue = active marking (unmethylated = free to explore).

    CpG islands modeled as spatial hash buckets.
    """

    def __init__(self, dim, resolution=20, methylation_strength=0.5):
        self.dim = dim
        self.resolution = resolution
        self.strength = methylation_strength
        self.cpg_map = {}  # hash → visit count
        self.max_visits = 10

    def _hash_position(self, x, lb, ub):
        """Discretize position into CpG island."""
        normalized = (x - lb) / (ub - lb + 1e-10)
        bucket = tuple(np.clip((normalized * self.resolution).astype(int), 0, self.resolution - 1))
        return bucket

    def methylate(self, x, lb, ub):
        """Mark this region as visited (add methyl group)."""
        key = self._hash_position(x, lb, ub)
        self.cpg_map[key] = self.cpg_map.get(key, 0) + 1

    def repulsion_force(self, x, lb, ub, rng):
        """
        Returns a repulsion vector away from heavily methylated regions.
        More visits = stronger push away = explore elsewhere.
        """
        key = self._hash_position(x, lb, ub)
        visits = self.cpg_map.get(key, 0)

        if visits == 0:
            return np.zeros(self.dim)

        # Repulsion proportional to methylation level
        methylation_level = min(visits / self.max_visits, 1.0)
        repulsion_dir = rng.standard_normal(self.dim)
        repulsion_dir /= np.linalg.norm(repulsion_dir) + 1e-10
        return repulsion_dir * self.strength * methylation_level * np.linalg.norm(ub - lb) * 0.1


# ============================================================
# Superposition Chamber
# ============================================================

def superposition_collapse(candidates, obj_func, lb, ub):
    """
    Maintain multiple candidate states in superposition.
    Collapse via aoi_collapse — the Jordan component picks the
    rational/stable solution, associator measures chaos potential.

    Each candidate is projected into 24D, collapsed, and scored.
    The one with highest intent_magnitude (clearest Jordan signal) wins.
    """
    best_pos = None
    best_score = -np.inf
    collapse_data = []

    for x in candidates:
        # Pad/project candidate into 24D for quantum collapse
        if len(x) < 24:
            state_24d = np.pad(x, (0, 24 - len(x)))
        else:
            state_24d = x[:24]

        # Collapse through octonion decomposition
        result = aoi_collapse(state_24d)

        # Score = intent (Jordan clarity) - chaos penalty + fitness bonus
        fitness = obj_func(np.clip(x, lb, ub))
        intent = result['intent_magnitude']
        chaos = result['normalized_chaos']

        # Higher intent = more coherent solution, lower chaos = more stable
        # Lower fitness = better (minimization)
        superposition_score = intent - 0.3 * chaos - 0.1 * fitness

        collapse_data.append({
            'position': x.copy(),
            'fitness': fitness,
            'intent': intent,
            'chaos': chaos,
            'score': superposition_score,
            'personality': result['personality_embedding'],
            'control': result['control_vec'],
        })

        if superposition_score > best_score:
            best_score = superposition_score
            best_pos = x.copy()

    return best_pos, collapse_data


# ============================================================
# HBO-Quantum-TE Optimizer
# ============================================================

class HBOQuantumTE:
    """
    Honey Badger Optimizer enhanced with:
    - TE layers (inner→outer + inverted outer→inner)
    - Quantum collapse (aoi_collapse)
    - Mitochondria energy gating
    - Methyl blue epigenetic memory
    - Superposition state collapse

    The TE application alternates:
    - Even iterations: inner→outer (fine → coarse)
    - Odd iterations: outer→inner (coarse → fine, inverted)
    Both variants run in superposition, then collapse picks winner.
    """

    def __init__(self, obj_func, dim, lb, ub, pop_size=30, max_iter=100, seed=None):
        self.obj_func = obj_func
        self.dim = dim
        self.lb = np.full(dim, lb) if np.isscalar(lb) else np.array(lb)
        self.ub = np.full(dim, ub) if np.isscalar(ub) else np.array(ub)
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.rng = np.random.default_rng(seed)

        # Components
        self.te_stack = build_te_stack()
        self.mito = [Mitochondria() for _ in range(pop_size)]
        self.methyl = MethylBlue(dim)

        # State
        self.population = None
        self.fitness = None
        self.best_pos = None
        self.best_fit = np.inf
        self.convergence = []
        self.collapse_history = []

    def clip(self, x):
        return np.clip(x, self.lb, self.ub)

    def evaluate(self, x):
        return self.obj_func(self.clip(x))

    def initialize(self):
        self.population = self.lb + (self.ub - self.lb) * self.rng.random((self.pop_size, self.dim))
        self.fitness = np.array([self.evaluate(x) for x in self.population])
        idx = np.argmin(self.fitness)
        self.best_fit = self.fitness[idx]
        self.best_pos = self.population[idx].copy()

    def _hbo_step(self, i, alpha):
        """Core HBO update for agent i."""
        C = 2
        beta = 6
        F = 1 if self.rng.random() < 0.5 else -1
        r = self.rng.random()

        if r < 0.5:
            # Digging phase
            r3 = self.rng.random()
            di = self.best_pos - self.population[i]
            S_i = F * alpha * di
            return self.best_pos + S_i + beta * r3 * np.abs(di)
        else:
            # Honey phase
            r5 = self.rng.random()
            j = self.rng.integers(self.pop_size)
            di = self.best_pos - self.population[j]
            return self.best_pos + F * alpha * r5 * np.abs(di)

    def _iterate(self, t):
        alpha = (1 - t / self.max_iter) * 2  # HBO decreasing factor

        for i in range(self.pop_size):
            old_fit = self.fitness[i]

            # === Step 1: HBO base move ===
            hbo_pos = self._hbo_step(i, alpha)

            # === Step 2: TE wrapping — both directions in superposition ===
            te_forward = apply_te_forward(hbo_pos, self.te_stack, self.rng)
            te_inverted = apply_te_inverted(hbo_pos, self.te_stack, self.rng)

            # === Step 3: Methyl blue repulsion ===
            repulsion = self.methyl.repulsion_force(self.population[i], self.lb, self.ub, self.rng)
            te_forward += repulsion
            te_inverted += repulsion

            # === Step 4: Mitochondria energy scaling ===
            energy = self.mito[i].energy_scale()

            # Scale TE perturbations by available ATP
            te_forward_scaled = self.population[i] + energy * (te_forward - self.population[i])
            te_inverted_scaled = self.population[i] + energy * (te_inverted - self.population[i])

            # Also keep raw HBO position as candidate
            hbo_scaled = self.population[i] + energy * (hbo_pos - self.population[i])

            # === Step 5: Superposition collapse ===
            # Three candidates enter superposition, one collapses out
            candidates = [
                self.clip(hbo_scaled),
                self.clip(te_forward_scaled),
                self.clip(te_inverted_scaled),
            ]

            collapsed_pos, collapse_data = superposition_collapse(
                candidates, self.obj_func, self.lb, self.ub
            )

            collapsed_pos = self.clip(collapsed_pos)
            new_fit = self.evaluate(collapsed_pos)

            # === Step 6: Accept or reject ===
            if new_fit < self.fitness[i]:
                delta = self.fitness[i] - new_fit
                self.population[i] = collapsed_pos
                self.fitness[i] = new_fit
                self.mito[i].produce_atp(-delta)  # improvement → ATP
            else:
                self.mito[i].produce_atp(new_fit - old_fit)  # no improvement → decay

                # Oxidative burst: if stagnant too long, spend all ATP on big jump
                if self.mito[i].atp_pool < 0.3:
                    burst_scale = self.mito[i].oxidative_burst()
                    burst_pos = self.population[i] + burst_scale * self.rng.standard_normal(self.dim)
                    burst_pos = self.clip(burst_pos)
                    burst_fit = self.evaluate(burst_pos)
                    if burst_fit < self.fitness[i]:
                        self.population[i] = burst_pos
                        self.fitness[i] = burst_fit

            # === Step 7: Methylate visited region ===
            self.methyl.methylate(self.population[i], self.lb, self.ub)

        # Update global best
        idx = np.argmin(self.fitness)
        if self.fitness[idx] < self.best_fit:
            self.best_fit = self.fitness[idx]
            self.best_pos = self.population[idx].copy()

    def optimize(self, verbose=False):
        """Run full optimization."""
        self.initialize()

        for t in range(self.max_iter):
            self._iterate(t)
            self.convergence.append(self.best_fit)

            if verbose and (t % 10 == 0 or t == self.max_iter - 1):
                # Run quantum collapse on best for diagnostics
                state_24d = np.pad(self.best_pos, (0, max(0, 24 - self.dim))) if self.dim < 24 else self.best_pos[:24]
                qc = aoi_collapse(state_24d)
                avg_atp = np.mean([m.atp_pool for m in self.mito])
                methylated = len(self.methyl.cpg_map)
                print(
                    f"  t={t:4d} | fit={self.best_fit:.8f} | "
                    f"chaos={qc['normalized_chaos']:.2f} | intent={qc['intent_magnitude']:.3f} | "
                    f"ATP={avg_atp:.2f} | methyl_regions={methylated}"
                )

        return self.best_pos, self.best_fit


# ============================================================
# Benchmark
# ============================================================

def benchmark():
    """Compare plain HBO vs HBO-Quantum-TE on standard functions."""

    from metaheuristic_library import HoneyBadgerOptimizer

    test_functions = {
        'Sphere': lambda x: np.sum(x ** 2),
        'Rastrigin': lambda x: 10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x)),
        'Ackley': lambda x: (
            -20 * np.exp(-0.2 * np.sqrt(np.mean(x ** 2)))
            - np.exp(np.mean(np.cos(2 * np.pi * x)))
            + 20 + np.e
        ),
        'Rosenbrock': lambda x: np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2),
    }

    dim = 10
    pop_size = 30
    max_iter = 200

    print("=" * 80)
    print("HBO-Quantum-TE Benchmark")
    print(f"dim={dim}, pop={pop_size}, iter={max_iter}")
    print("=" * 80)

    for name, func in test_functions.items():
        print(f"\n--- {name} ---")

        # Plain HBO
        results_plain = []
        for trial in range(5):
            opt = HoneyBadgerOptimizer(
                obj_func=func, dim=dim, lb=-5.12, ub=5.12,
                pop_size=pop_size, max_iter=max_iter
            )
            _, best = opt.optimize()
            results_plain.append(best)

        # HBO-Quantum-TE
        results_quantum = []
        for trial in range(5):
            opt = HBOQuantumTE(
                obj_func=func, dim=dim, lb=-5.12, ub=5.12,
                pop_size=pop_size, max_iter=max_iter, seed=trial
            )
            _, best = opt.optimize(verbose=(trial == 0))
            results_quantum.append(best)

        plain_mean = np.mean(results_plain)
        plain_std = np.std(results_plain)
        qt_mean = np.mean(results_quantum)
        qt_std = np.std(results_quantum)

        print(f"  Plain HBO:      {plain_mean:.8f} +/- {plain_std:.8f}")
        print(f"  HBO-Quantum-TE: {qt_mean:.8f} +/- {qt_std:.8f}")

        if qt_mean < plain_mean:
            improvement = (1 - qt_mean / (plain_mean + 1e-20)) * 100
            print(f"  >>> Quantum-TE wins by {improvement:.1f}%")
        else:
            print(f"  >>> Plain HBO wins this round")

    print("\n" + "=" * 80)
    print("Benchmark complete.")


if __name__ == '__main__':
    benchmark()
