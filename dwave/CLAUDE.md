# D-Wave Integration — Project Context

## What's Built
- `transponder_gates.py` — BQM penalty model for 32-transponder entropy engine. All 4 tests pass.
- `demo_local.py` — Circuit fault diagnosis demo adapted for local SimulatedAnnealingSampler.
- `circuit_fault_diagnosis/` — D-Wave's original example (reference implementation).

## What's Next
1. `collapse_to_entropy_readings()` — Bridge from aoi_collapse to transponder entropy inputs
2. Penalty tuning for SA/QPU
3. Energy gap tracking
4. QPU swap (waiting on D-Wave LaunchPad)

## Key Concepts
- BQM (Binary Quadratic Model) — chosen model type for transponder optimization
- ENTROPY_GATE: 2 Ising spins per transponder (entropy_state, gate_active)
- Valid: LOW+active, HIGH+attenuated. Fault: HIGH+active, LOW+attenuated.
- Super-logarithm target: 34.031437
- 3 layers: Background (9), Highlighted (17), Evolutionary (6)

## D-Wave Ocean SDK
- Version: 9.3.0
- Runner: `py -3.12`
- Current solver: SimulatedAnnealingSampler (local, no QPU)
- QPU solver: DWaveSampler / LeapHybridSampler (after LaunchPad approval)
