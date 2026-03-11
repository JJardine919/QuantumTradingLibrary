# Circuit Fault Diagnosis — Local (Simulated Annealing) version
# Swaps DWaveSampler for SimulatedAnnealingSampler so we can run without QPU

import sys
import pandas as pd
import dimod

from circuit_fault_diagnosis.circuits import three_bit_multiplier
from circuit_fault_diagnosis.gates import GATES

NUM_READS = 1000

def run_diagnosis(A, B, P, verbose=True):
    """Run fault diagnosis for given inputs A, B and observed product P."""
    bqm, labels = three_bit_multiplier()

    fixed_variables = {}
    fixed_variables.update(zip(('a2', 'a1', 'a0'), "{:03b}".format(A)))
    fixed_variables.update(zip(('b2', 'b1', 'b0'), "{:03b}".format(B)))
    fixed_variables.update(zip(('p5', 'p4', 'p3', 'p2', 'p1', 'p0'), "{:06b}".format(P)))

    correct = A * B
    print("=" * 60)
    print(f"  A = {A} ({A:03b})  x  B = {B} ({B:03b})")
    print(f"  Correct A*B = {correct} ({correct:06b})")
    print(f"  Observed  P = {P} ({P:06b})")
    if P == correct:
        print("  STATUS: No fault (correct product)")
    else:
        print(f"  STATUS: FAULT — {bin(correct ^ P).count('1')} bits differ")
    print("=" * 60)

    fixed_variables = {var: 1 if x == '1' else -1 for (var, x) in fixed_variables.items()}

    # fix variables
    for var, value in fixed_variables.items():
        bqm.fix_variable(var, value)
    # fix any disconnected aux variables
    for var in list(bqm.variables):
        if 'aux' in str(var) and len(list(bqm.iter_neighborhood(var))) == 0:
            bqm.fix_variable(var, 1)

    # Use simulated annealing (local, no QPU needed)
    sampler = dimod.SimulatedAnnealingSampler()
    response = sampler.sample_ising(bqm.linear,
                                    bqm.quadratic,
                                    num_reads=NUM_READS)

    # Process results
    min_energy = next(response.data()).energy

    best_samples = [dict(datum.sample) for datum in response.data() if datum.energy == min_energy]
    for sample in best_samples:
        for variable in list(sample.keys()):
            if 'aux' in variable:
                sample.pop(variable)
        sample.update(fixed_variables)

    best_results = []
    for sample in best_samples:
        result = {}
        for gate_type, gates in sorted(labels.items()):
            _, configurations = GATES[gate_type]
            for gate_name, gate in sorted(gates.items()):
                result[gate_name] = 'valid' if tuple(sample[var] for var in gate) in configurations else 'fault'
        best_results.append(result)
    best_results = pd.DataFrame(best_results)

    num_faults = next(best_results.itertuples()).count('fault')
    best_results = best_results.drop_duplicates().reset_index(drop=True)
    num_ground_states = len(best_results)

    print(f"\n  Minimum fault diagnosis: {num_faults} faulty component(s)")
    print(f"  {num_ground_states} distinct fault state(s) observed\n")

    if verbose:
        pd.set_option('display.width', 120)
        print(best_results)
    print()
    return num_faults, num_ground_states, best_results


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  CIRCUIT FAULT DIAGNOSIS — Simulated Annealing (Local)")
    print("  D-Wave Ocean SDK demo running without QPU")
    print("=" * 60 + "\n")

    # Test case 1: Correct product (no faults expected)
    print(">>> TEST 1: Correct multiplication (expect 0 faults)")
    run_diagnosis(A=6, B=5, P=30)

    # Test case 2: Single faulty component — 5 incorrect bits
    print(">>> TEST 2: Single fault — A=6, B=5, P=32 (5 bits wrong)")
    run_diagnosis(A=6, B=5, P=32)

    # Test case 3: Two faulty components — all 6 bits wrong
    print(">>> TEST 3: Two faults — A=6, B=5, P=33 (all 6 bits wrong)")
    run_diagnosis(A=6, B=5, P=33)

    # Test case 4: Maximum faults
    print(">>> TEST 4: Max faults — A=7, B=6, P=1")
    run_diagnosis(A=7, B=6, P=1)
