"""
Smoke test: aoi_collapse -> collapse_bridge -> transponder_fault_diagnosis

Full pipeline: 24D state -> perception -> entropy readings -> D-Wave BQM -> fault report
"""
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'voodoo'))

from aoi_collapse import aoi_collapse
from collapse_bridge import collapse_to_entropy_readings
from transponder_gates import transponder_fault_diagnosis

print("=" * 60)
print("  SMOKE TEST: Full Voodoo -> D-Wave Pipeline")
print("=" * 60)

rng = np.random.default_rng(42)

for label, scale in [("calm", 0.3), ("normal", 1.0), ("chaotic", 4.0)]:
    state = rng.standard_normal(24) * scale

    # Step 1: Perception
    collapse = aoi_collapse(state)
    chaos = collapse['normalized_chaos']

    # Step 2: Bridge
    readings = collapse_to_entropy_readings(state)
    high_count = sum(1 for v in readings.values() if v >= 0.5)
    low_count = 32 - high_count

    # Step 3: Fault diagnosis
    faults, sample, energy, layer_status = transponder_fault_diagnosis(readings, verbose=False)

    print(f"\n  [{label}] chaos={chaos:.1f}/10  HIGH={high_count}  LOW={low_count}  faults={len(faults)}  energy={energy:.4f}")
    if faults:
        print(f"    Faulty: {faults}")
    for layer, (active, minimum, ok) in layer_status.items():
        status = "OK" if ok else "BELOW MIN"
        print(f"    {layer}: {active} active (min {minimum}) {status}")

print("\n" + "=" * 60)
print("  PIPELINE COMPLETE")
print("=" * 60)
