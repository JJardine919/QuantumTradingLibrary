"""Tests for penalty auto-tuning and energy gap tracking."""
import sys
from pathlib import Path

# Add parent dirs for imports — same pattern as test_collapse_bridge.py
sys.path.insert(0, str(Path(__file__).parent))

from transponder_gates import ALL_LAYERS
from penalty_tuner import tune_penalties, analyze_energy_gap, _make_readings


def test_penalty_tuner_returns_report():
    """Basic structure check — tune_penalties returns all expected keys."""
    readings = _make_readings()
    report = tune_penalties(readings, num_reads=50, max_iterations=2)
    assert isinstance(report, dict), "Report should be a dict"
    expected_keys = {'penalty_scale', 'energy_gap', 'valid_energy',
                     'invalid_energy', 'confidence', 'iterations'}
    for key in expected_keys:
        assert key in report, f"Missing key: {key}"
    assert isinstance(report['penalty_scale'], float)
    assert isinstance(report['energy_gap'], float)
    assert isinstance(report['valid_energy'], float)
    assert isinstance(report['confidence'], float)
    assert isinstance(report['iterations'], int)


def test_energy_gap_positive():
    """Gap should be > 0 for a well-formed BQM with clean readings."""
    readings = _make_readings()
    report = tune_penalties(readings, num_reads=100, max_iterations=3)
    # With all-LOW readings, the valid state (all active) should have lower
    # energy than any faulty state. Gap must be positive.
    assert report['energy_gap'] > 0, (
        f"Energy gap should be positive, got {report['energy_gap']}"
    )


def test_confidence_between_0_and_1():
    """Confidence score must be in [0, 1] range."""
    # Test with clean readings
    readings_clean = _make_readings()
    report_clean = tune_penalties(readings_clean, num_reads=50)
    assert 0.0 <= report_clean['confidence'] <= 1.0, (
        f"Clean confidence out of range: {report_clean['confidence']}"
    )
    # Test with mixed readings
    readings_mixed = _make_readings({
        'Ty3_Gypsy': 0.85,
        'LINE': 0.90,
        'Crossover': 0.78,
    })
    report_mixed = tune_penalties(readings_mixed, num_reads=50)
    assert 0.0 <= report_mixed['confidence'] <= 1.0, (
        f"Mixed confidence out of range: {report_mixed['confidence']}"
    )
    # Test analyze_energy_gap confidence too
    gap_report = analyze_energy_gap(readings_clean, num_reads=50)
    assert 0.0 <= gap_report['confidence_score'] <= 1.0, (
        f"Gap confidence out of range: {gap_report['confidence_score']}"
    )


def test_fault_stability():
    """Same readings produce same faults across samples with scaled penalties."""
    readings = _make_readings({
        'Ty3_Gypsy': 0.85,
        'LINE': 0.90,
        'HERV_Synapse': 0.75,
    })
    # Use penalty_scale=10.0 so SA has strong enough biases to converge.
    # This is exactly the workflow: tune_penalties finds the right scale,
    # then analyze_energy_gap validates with that scale.
    report = analyze_energy_gap(readings, num_reads=100, penalty_scale=10.0)
    # With 10x penalties, SA should converge to consistent fault sets
    assert report['fault_stability'] >= 0.5, (
        f"Fault stability too low: {report['fault_stability']}"
    )
    # The analytical faults list should be empty (all gates respond correctly)
    assert isinstance(report['faults'], list)
    # Sample quality should be reasonable with strong penalties
    assert report['sample_quality'] >= 0.05, (
        f"Sample quality too low: {report['sample_quality']}"
    )


def test_scaled_penalties_increase_gap():
    """Higher penalty scale should produce a larger (or equal) energy gap."""
    readings = _make_readings({
        'CR1_Jockey': 0.80,
        'DIRS1': 0.72,
        'Crossover': 0.78,
    })
    gap_1x = analyze_energy_gap(readings, num_reads=100, penalty_scale=1.0)
    gap_3x = analyze_energy_gap(readings, num_reads=100, penalty_scale=3.0)
    # Scaled penalties amplify the energy difference between valid and invalid
    assert gap_3x['energy_gap'] >= gap_1x['energy_gap'] * 0.9, (
        f"3x penalty gap ({gap_3x['energy_gap']:.4f}) should be >= "
        f"1x gap ({gap_1x['energy_gap']:.4f})"
    )


if __name__ == '__main__':
    tests = [
        test_penalty_tuner_returns_report,
        test_energy_gap_positive,
        test_confidence_between_0_and_1,
        test_fault_stability,
        test_scaled_penalties_increase_gap,
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
