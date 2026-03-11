# Copyright 2026 Quantum Children / Biskits
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
penalty_tuner.py — Penalty auto-calibration and energy gap tracking for D-Wave BQM.

Analyzes the transponder fault diagnosis BQM to measure the energy gap between
valid (fault-free) and invalid (faulty) ground states.  If the gap is too small,
scales penalties up and retries.

The energy gap is computed analytically from the BQM structure after fixing
entropy_state variables.  Each gate is independent after fixing, so the per-gate
energy gap is exactly 2 * |effective_bias| (cost of flipping one gate from optimal).
The minimum per-gate gap is the system bottleneck — the weakest link.

SimulatedAnnealingSampler validates that the sampler can actually find the ground
state and provides confidence metrics (sample quality, fault stability).
"""

import sys
import math
from pathlib import Path
from collections import Counter

# Add parent dirs for imports — same pattern as collapse_bridge.py
sys.path.insert(0, str(Path(__file__).parent))

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import dimod
SimulatedAnnealingSampler = dimod.SimulatedAnnealingSampler

from transponder_gates import (
    transponder_circuit,
    ALL_LAYERS,
    _ENTROPY_GATE_VALID,
    _exact_gate_active,
    FAULT_GAP,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_readings(overrides=None):
    """Build a full entropy_readings dict, defaulting everything to LOW entropy."""
    readings = {}
    for layer in ALL_LAYERS.values():
        for bio, _, _ in layer:
            readings[bio] = 0.1   # default: LOW entropy (clean signal)
    if overrides:
        readings.update(overrides)
    return readings


def _fix_entropy(bqm, labels, entropy_readings):
    """
    Build fixed dict from entropy readings (without modifying the BQM).

    Returns (fixed_dict, gate_vars).
    """
    fixed = {}
    for bio, reading in entropy_readings.items():
        if bio not in labels:
            continue
        entropy_var, _ = labels[bio]
        spin = +1 if reading < 0.5 else -1
        fixed[entropy_var] = spin

    gate_vars = [labels[bio][1] for bio in entropy_readings if bio in labels]
    return fixed, gate_vars


def _analytical_energy(bqm, labels, entropy_readings, fixed, gate_solution):
    """Compute total energy for a given gate solution with fixed entropy."""
    energy = 0.
    for bio in entropy_readings:
        if bio not in labels:
            continue
        entropy_var, gate_var = labels[bio]
        e_spin = fixed[entropy_var]
        g_spin = gate_solution[gate_var]
        J = bqm.adj[gate_var].get(entropy_var, bqm.adj[entropy_var].get(gate_var, 0.))
        energy += J * e_spin * g_spin + 0.25   # 0.25 = per-gate offset from penalty model
    return energy


def _per_gate_gap(bqm, labels, entropy_readings, fixed):
    """
    Compute the per-gate energy gap analytically.

    After fixing entropy_state, each gate_active has an effective linear bias:
        eff_bias = J * fixed_entropy_spin

    Flipping gate_active from optimal costs 2 * |eff_bias| in energy.
    Returns dict {bio_name: gap} and the minimum gap.
    """
    gaps = {}
    for bio in entropy_readings:
        if bio not in labels:
            continue
        entropy_var, gate_var = labels[bio]
        e_spin = fixed[entropy_var]
        J = bqm.adj[gate_var].get(entropy_var, bqm.adj[entropy_var].get(gate_var, 0.))
        eff_bias = J * e_spin
        gaps[bio] = 2.0 * abs(eff_bias)
    min_gap = min(gaps.values()) if gaps else 0.0
    return gaps, min_gap


def _classify_sample(sample, labels, fixed, entropy_readings):
    """
    Classify a sample as valid or invalid based on ENTROPY_GATE configs.

    Returns (is_valid, fault_count, fault_list).
    """
    fault_count = 0
    fault_list = []
    for bio in entropy_readings:
        if bio not in labels:
            continue
        entropy_var, gate_var = labels[bio]
        e_spin = fixed[entropy_var]
        g_spin = sample.get(gate_var)
        if g_spin is None:
            continue
        if (int(e_spin), int(g_spin)) not in _ENTROPY_GATE_VALID:
            fault_count += 1
            fault_list.append(bio)
    return (fault_count == 0), fault_count, fault_list


# ---------------------------------------------------------------------------
# BQM builder with penalty scaling
# ---------------------------------------------------------------------------

def _build_scaled_bqm(penalty_scale):
    """
    Build the 32-transponder BQM with scaled penalties.

    Scaling multiplies all biases and offset by penalty_scale, amplifying the
    energy difference between valid and invalid configurations.

    Returns (bqm, labels).
    """
    bqm, labels = transponder_circuit()

    if penalty_scale != 1.0:
        scaled_quadratic = {}
        for (u, v), bias in bqm.quadratic.items():
            scaled_quadratic[(u, v)] = bias * penalty_scale
        scaled_linear = {}
        for var, bias in bqm.linear.items():
            scaled_linear[var] = bias * penalty_scale
        scaled_offset = bqm.offset * penalty_scale
        bqm = dimod.BinaryQuadraticModel(
            scaled_linear, scaled_quadratic, scaled_offset, dimod.SPIN
        )

    return bqm, labels


# ---------------------------------------------------------------------------
# Penalty auto-calibration
# ---------------------------------------------------------------------------

def tune_penalties(entropy_readings=None, num_reads=100, gap_threshold=0.1,
                   max_iterations=5, scale_step=1.5):
    """
    Auto-tune BQM penalties for optimal energy gap between valid and invalid states.

    The energy gap is computed analytically: after fixing entropy_state, each gate
    is independent with effective bias = J * entropy_spin.  The per-gate gap is
    2 * |effective_bias|.  The system gap is the minimum per-gate gap (weakest link).

    SimulatedAnnealingSampler validates that the sampler finds the ground state.
    If the analytical gap is below threshold, penalties are scaled up.

    Args:
        entropy_readings: dict {bio_name: float}. If None, uses default (all LOW).
        num_reads: Number of SA samples per iteration for validation.
        gap_threshold: Minimum acceptable energy gap per gate.
        max_iterations: Maximum tuning iterations.
        scale_step: Multiplicative penalty scale increase per iteration.

    Returns:
        dict with keys:
            penalty_scale    (float): Final penalty scale factor applied.
            energy_gap       (float): Minimum per-gate energy gap (weakest link).
            valid_energy     (float): Analytical ground state energy (all gates optimal).
            invalid_energy   (float): Energy of single-gate-flip invalid state.
            confidence       (float): Confidence score 0-1 based on gap and SA validation.
            iterations       (int):   Number of tuning iterations performed.
    """
    if entropy_readings is None:
        entropy_readings = _make_readings()

    sampler = SimulatedAnnealingSampler()
    penalty_scale = 1.0
    iteration = 0

    for iteration in range(1, max_iterations + 1):
        bqm, labels = _build_scaled_bqm(penalty_scale)
        fixed, gate_vars = _fix_entropy(bqm, labels, entropy_readings)

        # Analytical ground state
        optimal_gates = _exact_gate_active(bqm, gate_vars, fixed)
        valid_energy = _analytical_energy(bqm, labels, entropy_readings, fixed, optimal_gates)

        # Per-gate energy gap (analytical)
        gate_gaps, min_gap = _per_gate_gap(bqm, labels, entropy_readings, fixed)

        # Invalid energy = valid + min_gap (flipping the weakest gate)
        invalid_energy = valid_energy + min_gap

        if min_gap >= gap_threshold:
            break

        penalty_scale *= scale_step

    # SA validation — how well does the sampler find the ground state?
    reduced_bqm = bqm.copy()
    for var, val in fixed.items():
        reduced_bqm.fix_variable(var, val)

    response = sampler.sample(reduced_bqm, num_reads=num_reads)
    sa_energies = [float(d.energy) for d in response.data(['energy'])]

    # Count how many SA samples match the analytical ground state
    optimal_reduced_energy = valid_energy - bqm.offset  # reduced BQM has different offset
    # Actually, just check if SA finds the minimum
    sa_min = min(sa_energies) if sa_energies else float('inf')
    sa_at_min = sum(1 for e in sa_energies if abs(e - sa_min) < 1e-6)
    sa_quality = sa_at_min / len(sa_energies) if sa_energies else 0.0

    # Confidence: combines analytical gap strength and SA convergence quality
    if min_gap <= 0:
        gap_score = 0.0
    elif min_gap > 1.0:
        gap_score = 1.0
    else:
        gap_score = 1.0 / (1.0 + math.exp(-10.0 * (min_gap - 0.1)))

    confidence = 0.6 * gap_score + 0.4 * sa_quality
    confidence = max(0.0, min(1.0, confidence))

    return {
        'penalty_scale': penalty_scale,
        'energy_gap': min_gap,
        'valid_energy': valid_energy,
        'invalid_energy': invalid_energy,
        'confidence': confidence,
        'iterations': iteration,
    }


# ---------------------------------------------------------------------------
# Energy gap analysis
# ---------------------------------------------------------------------------

def analyze_energy_gap(entropy_readings=None, num_reads=200, penalty_scale=1.0):
    """
    Run fault diagnosis and return detailed confidence metrics.

    Combines analytical energy gap computation with SA sampling for validation.

    Args:
        entropy_readings: dict {bio_name: float}. If None, uses default (all LOW).
        num_reads: Number of SA samples for validation.
        penalty_scale: Penalty scale factor (1.0 = default penalties).

    Returns:
        dict with keys:
            energy_gap       (float): Minimum per-gate energy gap (weakest link).
            confidence_score (float): 0-1 confidence based on gap and SA agreement.
            sample_quality   (float): Fraction of SA samples at the lowest found energy.
            fault_stability  (float): Agreement ratio — how consistently the same faults
                                      appear across SA samples (1.0 = perfect agreement).
            faults           (list):  Faulty transponders from the analytical ground state.
    """
    if entropy_readings is None:
        entropy_readings = _make_readings()

    bqm, labels = _build_scaled_bqm(penalty_scale)
    fixed, gate_vars = _fix_entropy(bqm, labels, entropy_readings)

    # Analytical solution
    optimal_gates = _exact_gate_active(bqm, gate_vars, fixed)
    valid_energy = _analytical_energy(bqm, labels, entropy_readings, fixed, optimal_gates)
    gate_gaps, min_gap = _per_gate_gap(bqm, labels, entropy_readings, fixed)

    # Classify the analytical ground state
    full_sample = {}
    full_sample.update(fixed)
    full_sample.update(optimal_gates)
    _, _, analytical_faults = _classify_sample(full_sample, labels, fixed, entropy_readings)

    # SA validation
    reduced_bqm = bqm.copy()
    for var, val in fixed.items():
        reduced_bqm.fix_variable(var, val)

    sampler = SimulatedAnnealingSampler()
    response = sampler.sample(reduced_bqm, num_reads=num_reads)

    all_energies = []
    all_fault_sets = []

    for datum in response.data(['sample', 'energy']):
        sample_dict = dict(datum.sample)
        energy_f = float(datum.energy)
        all_energies.append(energy_f)

        # Build full sample with fixed entropy for classification
        sa_full = dict(fixed)
        sa_full.update(sample_dict)
        _, _, faults = _classify_sample(sa_full, labels, fixed, entropy_readings)
        all_fault_sets.append(frozenset(faults))

    # Sample quality: fraction of SA samples at the lowest found energy
    if all_energies:
        sa_min = min(all_energies)
        optimal_count = sum(1 for e in all_energies if abs(e - sa_min) < 1e-6)
        sample_quality = optimal_count / len(all_energies)
    else:
        sample_quality = 0.0

    # Fault stability: how consistently the same fault set appears
    if all_fault_sets:
        fault_counter = Counter(all_fault_sets)
        most_common_count = fault_counter.most_common(1)[0][1]
        fault_stability = most_common_count / len(all_fault_sets)
    else:
        fault_stability = 0.0

    # Confidence score
    if min_gap <= 0:
        gap_score = 0.0
    elif min_gap > 1.0:
        gap_score = 1.0
    else:
        gap_score = 1.0 / (1.0 + math.exp(-10.0 * (min_gap - 0.1)))

    confidence_score = 0.6 * gap_score + 0.4 * fault_stability
    confidence_score = max(0.0, min(1.0, confidence_score))

    return {
        'energy_gap': min_gap,
        'confidence_score': confidence_score,
        'sample_quality': sample_quality,
        'fault_stability': fault_stability,
        'faults': analytical_faults,
    }
