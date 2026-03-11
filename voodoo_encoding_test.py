"""
Voodoo Encoding Canonicality Test

THE TEST: For each Millennium problem, try putting quantities in
WRONG octonion slots. If wrong encodings produce:
  - Norm violations
  - Associator sign flips
  - Broken Fano constraints
  - Internal contradictions

...then the encoding is FORCED by the algebra, not chosen.
If wrong encodings work just as well, the encoding is arbitrary.

This is the bridge between "interesting framework" and "proof."
"""
import sys
import numpy as np
from itertools import permutations

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, r"C:\Users\jimjj\Music\QuantumChildren\QuantumTradingLibrary")
from aoi_collapse import (
    Octonion, octonion_shadow_decompose, entropy_transponders, aoi_collapse,
    _CAYLEY_TABLE, _MUL_TENSOR
)


def p(text, end='\n'):
    print(text, end=end)


def section(title):
    p("")
    p("=" * 65)
    p(f"  {title}")
    p("=" * 65)
    p("")


def fano_triples():
    """Return the 7 Fano plane triples."""
    return [(1,2,4),(2,3,5),(3,4,6),(4,5,7),(5,6,1),(6,7,2),(7,1,3)]


def check_fano_consistency(slot_map, problem_constraints):
    """
    Given a slot_map (quantity -> octonion index) and problem constraints
    (pairs that should multiply to a third), check if the Cayley table
    agrees with the expected relationships.

    Returns (num_consistent, num_violated, details).
    """
    consistent = 0
    violated = 0
    details = []

    for q_a, q_b, q_c, expected_sign in problem_constraints:
        i = slot_map[q_a]
        j = slot_map[q_b]
        k = slot_map[q_c]

        # Check: does e_i * e_j produce e_k?
        sign, result = _CAYLEY_TABLE[i][j]
        if result == k:
            consistent += 1
            details.append(f"  OK: e{i}*e{j} = {'+'if sign>0 else '-'}e{result} (expected e{k})")
        else:
            violated += 1
            details.append(f"  VIOLATED: e{i}*e{j} = {'+'if sign>0 else '-'}e{result} != e{k}")

    return consistent, violated, details


def run_collapse_with_encoding(values, slot_assignments, B_vec, ctx):
    """
    Run AOI collapse with quantities assigned to specific slots.
    values: dict of quantity_name -> float value
    slot_assignments: dict of quantity_name -> slot index (0-7)
    """
    A_vec = np.zeros(8)
    for name, val in values.items():
        if name in slot_assignments:
            A_vec[slot_assignments[name]] = val

    state = np.concatenate([A_vec, B_vec, ctx])
    return aoi_collapse(state)


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':

    p("")
    p("*" * 65)
    p("  VOODOO — ENCODING CANONICALITY TEST")
    p("  Does the algebra REJECT wrong slot assignments?")
    p("*" * 65)
    p("")

    results_summary = {}

    # ================================================================
    # TEST 1: YANG-MILLS (Strongest case)
    # ================================================================

    section("TEST 1: YANG-MILLS — SU(3) Encoding")

    p("  CORRECT encoding:")
    p("  SU(3) = Stab_G2(e7)")
    p("  Gauge dims: e1..e6")
    p("  Mass gap direction: e7")
    p("  Fano triple (4,5,7): TIME x UV_FREEDOM = MASS_GAP")
    p("")

    # The correct encoding: gauge in e1-e6, mass gap in e7
    correct_ym = {
        'coupling': 0,     # e0 = coupling constant
        'color_1': 1,      # e1-e6 = SU(3) color charges
        'color_2': 2,
        'color_3': 3,
        'time': 4,         # e4 = time evolution
        'uv_freedom': 5,   # e5 = UV freedom / asymptotic freedom
        'ir_conf': 6,      # e6 = IR confinement
        'mass_gap': 7,     # e7 = mass gap (stabilizer direction)
    }

    # Wrong encodings: put mass gap in different slots
    wrong_ym_encodings = [
        ("mass_gap in e1", {**correct_ym, 'mass_gap': 1, 'color_1': 7}),
        ("mass_gap in e3", {**correct_ym, 'mass_gap': 3, 'color_3': 7}),
        ("mass_gap in e5", {**correct_ym, 'mass_gap': 5, 'uv_freedom': 7}),
        ("swap time/mass", {**correct_ym, 'mass_gap': 4, 'time': 7}),
        ("full scramble",  {'coupling': 0, 'color_1': 7, 'color_2': 6, 'color_3': 5,
                            'time': 1, 'uv_freedom': 2, 'ir_conf': 3, 'mass_gap': 4}),
    ]

    # Yang-Mills constraints: the Fano triple (4,5,7) should connect
    # time x uv_freedom = mass_gap
    ym_constraints = [
        ('time', 'uv_freedom', 'mass_gap', +1),
    ]

    p("  Fano consistency check:")
    p(f"  {'encoding':>20s}  {'consistent':>10s}  {'violated':>10s}  {'detail':>30s}")

    cons_correct, viol_correct, det_correct = check_fano_consistency(correct_ym, ym_constraints)
    p(f"  {'CORRECT':>20s}  {cons_correct:>10d}  {viol_correct:>10d}  {det_correct[0] if det_correct else '':>30s}")

    ym_fano_results = [('CORRECT', cons_correct, viol_correct)]

    for name, encoding in wrong_ym_encodings:
        cons, viol, det = check_fano_consistency(encoding, ym_constraints)
        p(f"  {name:>20s}  {cons:>10d}  {viol:>10d}  {det[0] if det else '':>30s}")
        ym_fano_results.append((name, cons, viol))

    p("")

    # Now run actual collapses and compare mass gap emergence
    p("  Mass gap emergence test (g sweep):")
    p("  Does mass gap (Jordan component in stabilizer dir) emerge")
    p("  ONLY with correct encoding?")
    p("")

    g_values = [0.5, 1.0, 2.0, 3.0]
    B_vec = np.array([1.0, 1, 1, 1, 1, 0.5, 0.5, 0.1])
    ctx = np.ones(8) * 0.5

    all_encodings = [("CORRECT", correct_ym)] + wrong_ym_encodings

    p(f"  {'encoding':>20s}", end="")
    for g in g_values:
        p(f"  {'g='+str(g):>10s}", end="")
    p(f"  {'monotonic':>10s}  {'positive':>10s}")

    for enc_name, encoding in all_encodings:
        mass_gap_slot = encoding['mass_gap']
        gaps = []
        for g in g_values:
            values = {
                'coupling': g,
                'color_1': 1.0, 'color_2': 1.0, 'color_3': 1.0,
                'time': 1.0, 'uv_freedom': 1.0, 'ir_conf': 1.0,
                'mass_gap': g * 0.3,
            }
            result = run_collapse_with_encoding(values, encoding, B_vec, ctx)
            j = result['decomposition']['jordan']
            c = result['decomposition']['commutator']
            # Mass gap = sqrt(J[slot]^2 + C[slot]^2)
            delta = np.sqrt(j.v[mass_gap_slot]**2 + c.v[mass_gap_slot]**2)
            gaps.append(delta)

        # Check: is mass gap monotonically increasing with g?
        monotonic = all(gaps[i] <= gaps[i+1] + 1e-10 for i in range(len(gaps)-1))
        all_positive = all(g > 1e-10 for g in gaps)

        p(f"  {enc_name:>20s}", end="")
        for gap in gaps:
            p(f"  {gap:10.4f}", end="")
        p(f"  {'YES' if monotonic else 'NO':>10s}  {'YES' if all_positive else 'NO':>10s}")

    p("")

    # KEY TEST: Associator structure
    p("  Associator structure test:")
    p("  The associator should have consistent sign in the mass gap")
    p("  direction ONLY with correct encoding.")
    p("")

    p(f"  {'encoding':>20s}  {'Assoc[gap_slot]':>15s}  {'sign_stable':>12s}  {'||Assoc||':>12s}")

    for enc_name, encoding in all_encodings:
        mass_gap_slot = encoding['mass_gap']
        assoc_values = []
        for g in np.arange(0.5, 5.0, 0.5):
            values = {
                'coupling': g,
                'color_1': 1.0, 'color_2': 1.0, 'color_3': 1.0,
                'time': 1.0, 'uv_freedom': 1.0, 'ir_conf': 1.0,
                'mass_gap': g * 0.3,
            }
            result = run_collapse_with_encoding(values, encoding, B_vec, ctx)
            a = result['decomposition']['associator']
            assoc_values.append(a.v[mass_gap_slot])

        # Check sign stability: all same sign?
        signs = [np.sign(v) for v in assoc_values if abs(v) > 1e-10]
        sign_stable = len(set(signs)) <= 1 if signs else False
        mean_assoc = np.mean(assoc_values)
        mean_norm = np.mean([abs(v) for v in assoc_values])

        p(f"  {enc_name:>20s}  {mean_assoc:+15.6f}  {'STABLE' if sign_stable else 'FLIPS':>12s}  {mean_norm:12.6f}")

    p("")

    # ================================================================
    # TEST 2: RIEMANN HYPOTHESIS
    # ================================================================

    section("TEST 2: RIEMANN HYPOTHESIS — Fano (2,3,5) Encoding")

    p("  CORRECT encoding:")
    p("  e2 = |zeta(s)|, e3 = arg(zeta(s)), e5 = Re(s) - 1/2")
    p("  Fano (2,3,5): magnitude x phase = distance_from_line")
    p("  At zero: 0 x phase = distance => distance = 0")
    p("")

    correct_rh = {
        'real_s': 0,       # e0 = Re(s)
        'imag_s': 1,       # e1 = Im(s)
        'zeta_mag': 2,     # e2 = |zeta(s)|
        'zeta_phase': 3,   # e3 = arg(zeta(s))
        'derivative': 4,   # e4 = |zeta'(s)|
        'distance': 5,     # e5 = Re(s) - 1/2
        'func_eq': 6,      # e6 = functional equation residual
        'zero_ind': 7,     # e7 = zero indicator
    }

    # Wrong encodings: put distance in wrong slot
    wrong_rh_encodings = [
        ("dist in e1", {**correct_rh, 'distance': 1, 'imag_s': 5}),
        ("dist in e4", {**correct_rh, 'distance': 4, 'derivative': 5}),
        ("dist in e7", {**correct_rh, 'distance': 7, 'zero_ind': 5}),
        ("mag/phase swap", {**correct_rh, 'zeta_mag': 3, 'zeta_phase': 2}),
        ("full scramble", {'real_s': 0, 'imag_s': 7, 'zeta_mag': 5, 'zeta_phase': 1,
                           'derivative': 6, 'distance': 2, 'func_eq': 3, 'zero_ind': 4}),
    ]

    # Constraint: zeta_mag x zeta_phase should give distance
    rh_constraints = [
        ('zeta_mag', 'zeta_phase', 'distance', +1),
        ('derivative', 'distance', 'zero_ind', +1),
    ]

    p("  Fano consistency check:")
    p(f"  {'encoding':>20s}  {'consistent':>10s}  {'violated':>10s}")

    cons_c, viol_c, _ = check_fano_consistency(correct_rh, rh_constraints)
    p(f"  {'CORRECT':>20s}  {cons_c:>10d}  {viol_c:>10d}")

    rh_fano_results = [('CORRECT', cons_c, viol_c)]

    for name, encoding in wrong_rh_encodings:
        cons, viol, _ = check_fano_consistency(encoding, rh_constraints)
        p(f"  {name:>20s}  {cons:>10d}  {viol:>10d}")
        rh_fano_results.append((name, cons, viol))

    p("")

    # Zero forcing test: at a zero (|zeta| = 0), does J[distance_slot] -> 0?
    p("  Zero forcing test:")
    p("  At a zeta zero (|zeta|=0), does J[distance_slot] go to 0?")
    p("  This is the core of the RH argument.")
    p("")

    known_zeros_t = [14.134725, 21.022040, 25.010858, 30.424876, 32.935062]

    all_rh_enc = [("CORRECT", correct_rh)] + wrong_rh_encodings

    p(f"  {'encoding':>20s}  {'mean |J[dist]|':>15s}  {'max |J[dist]|':>15s}  {'forces_zero':>12s}")

    for enc_name, encoding in all_rh_enc:
        dist_slot = encoding['distance']
        j_dist_vals = []

        for t in known_zeros_t:
            values = {
                'real_s': 0.5,
                'imag_s': t / 50,
                'zeta_mag': 0.001,      # near zero
                'zeta_phase': 0.0,
                'derivative': 1.0,
                'distance': 0.0,        # ON the critical line
                'func_eq': 0.0,
                'zero_ind': 0.001,
            }
            result = run_collapse_with_encoding(values, encoding,
                np.array([1, np.log(t)/4, 1, 0.5, 0.8, 0.9, 1, 0.5]),
                np.ones(8) * 0.3)

            j = result['decomposition']['jordan']
            j_dist_vals.append(abs(j.v[dist_slot]))

        mean_j = np.mean(j_dist_vals)
        max_j = np.max(j_dist_vals)
        forces = mean_j < 0.05  # Should be near zero

        p(f"  {enc_name:>20s}  {mean_j:15.8f}  {max_j:15.8f}  {'YES' if forces else 'NO':>12s}")

    p("")

    # ================================================================
    # TEST 3: NAVIER-STOKES
    # ================================================================

    section("TEST 3: NAVIER-STOKES — 3+3+1 Encoding")

    p("  CORRECT encoding:")
    p("  e1,e2,e3 = velocity (u1,u2,u3)")
    p("  e4,e5,e6 = vorticity (w1,w2,w3)")
    p("  e7 = enstrophy (|omega|^2)")
    p("  Fano (3,4,6): vorticity_comp x stretching = dissipation")
    p("")

    correct_ns = {
        'pressure': 0,  # e0
        'u1': 1, 'u2': 2, 'u3': 3,      # velocity
        'w1': 4, 'w2': 5, 'w3': 6,      # vorticity
        'enstrophy': 7,                   # enstrophy
    }

    wrong_ns_encodings = [
        ("vel/vort swap", {'pressure': 0, 'u1': 4, 'u2': 5, 'u3': 6,
                           'w1': 1, 'w2': 2, 'w3': 3, 'enstrophy': 7}),
        ("enstrophy e3", {**correct_ns, 'enstrophy': 3, 'u3': 7}),
        ("scrambled", {'pressure': 0, 'u1': 7, 'u2': 4, 'u3': 1,
                       'w1': 6, 'w2': 3, 'w3': 2, 'enstrophy': 5}),
    ]

    # Constraint: the Fano plane should create closed loops
    # velocity -> vorticity -> enstrophy -> velocity
    ns_constraints = [
        ('u3', 'w1', 'w3', +1),   # (3,4,6): e3*e4 = e6
        ('w1', 'w2', 'enstrophy', +1),  # (4,5,7): e4*e5 = e7
    ]

    p("  Fano consistency check:")
    p(f"  {'encoding':>20s}  {'consistent':>10s}  {'violated':>10s}")

    for enc_name, encoding in [("CORRECT", correct_ns)] + wrong_ns_encodings:
        cons, viol, _ = check_fano_consistency(encoding, ns_constraints)
        p(f"  {enc_name:>20s}  {cons:>10d}  {viol:>10d}")

    p("")

    # Blowup test: does wrong encoding allow unbounded growth?
    p("  Blowup resistance test:")
    p("  Increasing vorticity magnitude — does the enstrophy slot")
    p("  stay bounded (correct) or blow up (wrong)?")
    p("")

    B_ns = np.array([1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.1])
    ctx_ns = np.ones(8) * 0.3

    vort_mags = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]

    p(f"  {'encoding':>20s}", end="")
    for vm in vort_mags:
        p(f"  {'|w|='+str(vm):>10s}", end="")
    p(f"  {'growth':>10s}")

    for enc_name, encoding in [("CORRECT", correct_ns)] + wrong_ns_encodings:
        enst_slot = encoding['enstrophy']
        enst_vals = []
        for vm in vort_mags:
            values = {
                'pressure': 1.0,
                'u1': 1.0, 'u2': 1.0, 'u3': 1.0,
                'w1': vm, 'w2': vm, 'w3': vm,
                'enstrophy': vm**2,
            }
            result = run_collapse_with_encoding(values, encoding, B_ns, ctx_ns)
            j = result['decomposition']['jordan']
            enst_vals.append(abs(j.v[enst_slot]))

        # Growth rate: last / first
        growth = enst_vals[-1] / max(enst_vals[0], 1e-12)

        p(f"  {enc_name:>20s}", end="")
        for ev in enst_vals:
            p(f"  {ev:10.4f}", end="")
        p(f"  {growth:10.2f}x")

    p("")

    # ================================================================
    # TEST 4: G2 STABILIZER TEST (the real proof)
    # ================================================================

    section("TEST 4: G2 STABILIZER — THE REAL TEST")

    p("  G2 = Aut(O) has 14 dimensions.")
    p("  SU(3) = Stab_G2(e7) — the subgroup fixing e7.")
    p("  This is a THEOREM, not a choice.")
    p("")
    p("  If Yang-Mills mass gap must live in the stabilizer")
    p("  direction, then e7 is FORCED for SU(3).")
    p("  No other slot works because no other slot is fixed by SU(3).")
    p("")

    # Test: apply all 7! permutations of imaginary slots
    # and check which ones preserve the SU(3) structure
    p("  Testing all permutations of imaginary octonion slots:")
    p("  Which permutations preserve the Fano plane structure?")
    p("")

    # A permutation preserves Fano if for every triple (i,j,k),
    # (perm(i), perm(j), perm(k)) is also a Fano triple
    fano = set()
    for triple in fano_triples():
        fano.add(triple)

    # Also add cyclic permutations (a,b,c) -> (b,c,a), (c,a,b)
    fano_full = set()
    for a, b, c in fano_triples():
        fano_full.add((a, b, c))
        fano_full.add((b, c, a))
        fano_full.add((c, a, b))

    preserving_perms = []
    total_perms = 0

    for perm in permutations(range(1, 8)):
        total_perms += 1
        perm_map = {i+1: perm[i] for i in range(7)}

        preserved = True
        for a, b, c in fano_triples():
            pa, pb, pc = perm_map[a], perm_map[b], perm_map[c]
            if (pa, pb, pc) not in fano_full:
                preserved = False
                break

        if preserved:
            preserving_perms.append(perm)

    p(f"  Total permutations tested: {total_perms}")
    p(f"  Fano-preserving permutations: {len(preserving_perms)}")
    p("")

    if preserving_perms:
        p("  These are the AUTOMORPHISMS of the Fano plane.")
        p("  They correspond to elements of G2 (restricted to basis perms).")
        p("")
        for perm in preserving_perms[:10]:
            perm_map = {i+1: perm[i] for i in range(7)}
            fixes_e7 = (perm_map[7] == 7)
            p(f"    {perm}  fixes e7: {fixes_e7}")
        if len(preserving_perms) > 10:
            p(f"    ... and {len(preserving_perms) - 10} more")
        p("")

        # How many fix e7?
        fix_e7 = [p for p in preserving_perms if p[6] == 7]  # perm[6] is where 7 maps to
        p(f"  Permutations fixing e7: {len(fix_e7)} out of {len(preserving_perms)}")
        p(f"  These form the stabilizer Stab(e7) of the Fano automorphism group.")
        p("")

        # How many fix other slots?
        for slot in range(1, 8):
            fix_slot = [pm for pm in preserving_perms if pm[slot-1] == slot]
            p(f"    Fix e{slot}: {len(fix_slot)} permutations")

    p("")

    # ================================================================
    # TEST 5: NORM VIOLATION TEST
    # ================================================================

    section("TEST 5: NORM VIOLATION UNDER WRONG ENCODING")

    p("  If the encoding is canonical, wrong assignments should")
    p("  produce norm ratio anomalies: ||J||/||C|| deviates from")
    p("  the pattern seen with correct encoding.")
    p("")

    # For Yang-Mills: sweep coupling, measure J/C ratio
    p("  Yang-Mills J/C ratio vs coupling:")
    p(f"  {'encoding':>20s}  {'ratio_std':>10s}  {'ratio_range':>12s}  {'smooth':>8s}")

    for enc_name, encoding in [("CORRECT", correct_ym)] + wrong_ym_encodings:
        ratios = []
        for g in np.arange(0.1, 5.0, 0.1):
            values = {
                'coupling': g,
                'color_1': 1.0, 'color_2': 1.0, 'color_3': 1.0,
                'time': 1.0, 'uv_freedom': 1.0, 'ir_conf': 1.0,
                'mass_gap': g * 0.3,
            }
            result = run_collapse_with_encoding(values, encoding, B_vec, ctx)
            j_norm = result['decomposition']['jordan'].norm()
            c_norm = result['decomposition']['commutator'].norm()
            ratios.append(j_norm / max(c_norm, 1e-12))

        ratios = np.array(ratios)
        # Smoothness: std of first differences
        diffs = np.diff(ratios)
        smoothness = np.std(diffs)
        is_smooth = smoothness < 0.5

        p(f"  {enc_name:>20s}  {np.std(ratios):10.4f}  "
          f"{ratios.max()-ratios.min():12.4f}  {'YES' if is_smooth else 'NO':>8s}")

    p("")

    # ================================================================
    # SUMMARY
    # ================================================================

    section("SUMMARY — ENCODING CANONICALITY")

    p("  For each problem, the CORRECT encoding should show:")
    p("  1. Fano consistency: all constraints satisfied")
    p("  2. Physical behavior: mass gap emerges, zeros forced, etc.")
    p("  3. Stable associator: consistent sign in key direction")
    p("  4. Smooth J/C ratio: well-behaved under parameter sweep")
    p("")
    p("  Wrong encodings should show:")
    p("  1. Fano violations: algebraic constraints broken")
    p("  2. Wrong physics: mass gap doesn't emerge, zeros not forced")
    p("  3. Associator flips: unstable sign = no preferred direction")
    p("  4. Rough J/C ratio: erratic behavior = encoding fights algebra")
    p("")
    p("  If the algebra REJECTS wrong encodings and ACCEPTS only the")
    p("  correct one, the encoding is canonical — forced by G2, not chosen.")
    p("")
    p("  Run this script. Read the numbers. The algebra speaks.")
    p("")
    p("*" * 65)
