"""
collapse_bridge.py — Maps aoi_collapse output to 32 transponder entropy readings.

This is the bridge between Voodoo's perception (24D octonion decomposition)
and the D-Wave transponder fault diagnosis system.

Input:  24D numpy state vector (same as aoi_collapse input)
Output: dict {bio_name: float} for all 32 transponders
        Values in [0.0, 1.0]. >= 0.5 = HIGH entropy, < 0.5 = LOW entropy.

IMPORTANT: Per-dimension Shannon entropy from softmax is very small in absolute
terms (~0.02-0.12). We normalize RELATIVE to the median of the per-dim entropy
distribution, not to the theoretical maximum. This ensures meaningful HIGH/LOW
splits — dimensions above median entropy read HIGH, below read LOW.
"""

import sys
import numpy as np
from pathlib import Path

# Add parent dirs for imports
sys.path.insert(0, str(Path(__file__).parent))
_voodoo_dir = str(Path(__file__).parent.parent / 'voodoo')
if _voodoo_dir not in sys.path:
    sys.path.insert(0, _voodoo_dir)

from transponder_gates import BACKGROUND_LAYER, HIGHLIGHTED_LAYER, EVOLUTIONARY_LAYER

# Ordered list of all 32 transponder bio names
_BG_NAMES = [t[0] for t in BACKGROUND_LAYER]       # 9
_HL_NAMES = [t[0] for t in HIGHLIGHTED_LAYER]       # 17
_EV_NAMES = [t[0] for t in EVOLUTIONARY_LAYER]      # 6
_ALL_NAMES = _BG_NAMES + _HL_NAMES + _EV_NAMES      # 32


def _per_dim_entropy(state: np.ndarray) -> np.ndarray:
    """Compute per-dimension Shannon entropy from softmax probabilities."""
    shifted = state - np.max(state)
    probs = np.exp(shifted) / np.sum(np.exp(shifted))
    probs = np.clip(probs, 1e-12, None)
    return -probs * np.log2(probs)


def _median_normalize(entropy_per_dim: np.ndarray) -> np.ndarray:
    """
    Normalize entropy values relative to their own median.

    Values at the median map to 0.5. Values at 0 map to 0.0.
    Values at 2x the median map to 1.0. Clipped to [0, 1].

    This ensures roughly half the dimensions read HIGH (>= 0.5)
    and half read LOW (< 0.5) for random inputs.
    """
    # If all per-dim entropies are nearly equal, the input is uniform/calm → all LOW
    if np.std(entropy_per_dim) < 1e-10:
        return np.zeros_like(entropy_per_dim)
    median = np.median(entropy_per_dim)
    if median < 1e-15:
        return np.zeros_like(entropy_per_dim)
    # Scale so median = 0.5
    normalized = entropy_per_dim / (2.0 * median)
    return np.clip(normalized, 0.0, 1.0)


def _interpolate_to_names(values: np.ndarray, names: list) -> dict:
    """Map N-dimensional values to M named transponders via linear interpolation."""
    readings = {}
    n_vals = len(values)
    n_names = len(names)
    for i, name in enumerate(names):
        if n_names == 1:
            pos = 0.0
        else:
            pos = i * (n_vals - 1) / (n_names - 1)
        lo = int(pos)
        hi = min(lo + 1, n_vals - 1)
        frac = pos - lo
        val = values[lo] * (1 - frac) + values[hi] * frac
        readings[name] = float(val)
    return readings


def collapse_to_entropy_readings(state_24d: np.ndarray) -> dict:
    """
    Map a 24D state vector to 32 named transponder entropy readings.

    Pipeline:
        1. Pad/truncate input to 24D
        2. Compute per-dimension Shannon entropy (softmax probabilities)
        3. Normalize relative to median (median -> 0.5, not absolute max)
        4. Map 24 dimensions to 32 transponders via layer-based interpolation
           - Background (9 transponders): dims 0-7
           - Highlighted (17 transponders): dims 8-15
           - Evolutionary (6 transponders): dims 16-23

    Args:
        state_24d: numpy array, will be padded/truncated to 24D

    Returns:
        dict {bio_name: float} with all 32 transponder names.
        Values >= 0.5 indicate HIGH entropy (noisy signal).
        Values < 0.5 indicate LOW entropy (structured signal).
    """
    state = np.asarray(state_24d, dtype=np.float64).ravel()
    if len(state) < 24:
        state = np.pad(state, (0, 24 - len(state)))
    else:
        state = state[:24]

    # Per-dimension Shannon entropy
    entropy_per_dim = _per_dim_entropy(state)

    # Normalize relative to median (not absolute max)
    normalized = _median_normalize(entropy_per_dim)

    # Map 24 dimensions to 32 transponders by layer
    readings = {}
    readings.update(_interpolate_to_names(normalized[:8], _BG_NAMES))     # 8 dims -> 9 transponders
    readings.update(_interpolate_to_names(normalized[8:16], _HL_NAMES))   # 8 dims -> 17 transponders
    readings.update(_interpolate_to_names(normalized[16:24], _EV_NAMES))  # 8 dims -> 6 transponders

    return readings
