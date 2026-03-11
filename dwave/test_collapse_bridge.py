"""Tests for collapse_to_entropy_readings bridge."""
import numpy as np
import sys
from pathlib import Path

# Add parent dirs for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'voodoo'))

from transponder_gates import BACKGROUND_LAYER, HIGHLIGHTED_LAYER, EVOLUTIONARY_LAYER

ALL_BIO_NAMES = (
    [t[0] for t in BACKGROUND_LAYER] +
    [t[0] for t in HIGHLIGHTED_LAYER] +
    [t[0] for t in EVOLUTIONARY_LAYER]
)


def test_returns_dict_with_32_keys():
    from collapse_bridge import collapse_to_entropy_readings
    state = np.random.default_rng(42).standard_normal(24)
    result = collapse_to_entropy_readings(state)
    assert isinstance(result, dict)
    assert len(result) == 32


def test_all_transponder_names_present():
    from collapse_bridge import collapse_to_entropy_readings
    state = np.random.default_rng(42).standard_normal(24)
    result = collapse_to_entropy_readings(state)
    for name in ALL_BIO_NAMES:
        assert name in result, f"Missing transponder: {name}"


def test_values_are_floats_between_0_and_1():
    from collapse_bridge import collapse_to_entropy_readings
    state = np.random.default_rng(42).standard_normal(24)
    result = collapse_to_entropy_readings(state)
    for name, val in result.items():
        assert isinstance(val, float), f"{name} value is not float: {type(val)}"
        assert 0.0 <= val <= 1.0, f"{name} value out of range: {val}"


def test_calm_input_produces_mostly_low_entropy():
    """A near-zero input should produce low entropy (structured signal)."""
    from collapse_bridge import collapse_to_entropy_readings
    state = np.ones(24) * 0.01  # very calm, uniform
    result = collapse_to_entropy_readings(state)
    low_count = sum(1 for v in result.values() if v < 0.5)
    assert low_count >= 20, f"Only {low_count}/32 LOW for calm input"


def test_chaotic_input_produces_some_high_entropy():
    """A high-variance input should produce some high entropy readings."""
    from collapse_bridge import collapse_to_entropy_readings
    rng = np.random.default_rng(99)
    state = rng.standard_normal(24) * 5.0  # very chaotic
    result = collapse_to_entropy_readings(state)
    high_count = sum(1 for v in result.values() if v >= 0.5)
    assert high_count >= 1, f"Zero HIGH readings for chaotic input"


def test_deterministic():
    """Same input produces same output."""
    from collapse_bridge import collapse_to_entropy_readings
    state = np.random.default_rng(42).standard_normal(24)
    r1 = collapse_to_entropy_readings(state)
    r2 = collapse_to_entropy_readings(state)
    for name in ALL_BIO_NAMES:
        assert r1[name] == r2[name], f"{name} not deterministic"


def test_integrates_with_transponder_fault_diagnosis():
    """Bridge output can be fed directly to transponder_fault_diagnosis."""
    from collapse_bridge import collapse_to_entropy_readings
    from transponder_gates import transponder_fault_diagnosis
    state = np.random.default_rng(42).standard_normal(24)
    readings = collapse_to_entropy_readings(state)
    faults, sample, energy, layer_status = transponder_fault_diagnosis(readings, verbose=False)
    assert isinstance(faults, list)
    assert isinstance(energy, float)


if __name__ == '__main__':
    tests = [
        test_returns_dict_with_32_keys,
        test_all_transponder_names_present,
        test_values_are_floats_between_0_and_1,
        test_calm_input_produces_mostly_low_entropy,
        test_chaotic_input_produces_some_high_entropy,
        test_deterministic,
        test_integrates_with_transponder_fault_diagnosis,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS: {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL: {t.__name__} -- {e}")
    print(f"\n{passed}/{len(tests)} passed")
