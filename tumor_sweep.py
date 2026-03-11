"""
Voodoo AOI v3.0 — Gompertz-Immune Parameter Sweep

Sweeps delta (immune killing) and rho (immune activation) to find
the algebraic phase transition where the 24D octonion decomposition
flips from escape dynamics to bounded/remission attractor.

dN/dt = r*N*(1 - ln(N)/K) - delta*N*I
dI/dt = s + rho*N*I - mu*I

Math: Zenodo DOI chain 18690444, 18722487, 18809406, 18809716.
"""

import sys
import numpy as np

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, '.')
from aoi_collapse import Octonion, octonion_shadow_decompose, entropy_transponders, aoi_collapse


# ============================================================
# Fixed parameters
# ============================================================
r  = 0.5
K  = 1e9
s  = 1e4
mu = 0.2

K_ln = np.log(K)


def encode_tumor_state(delta, rho):
    """Encode Gompertz-immune model into 24D state vector."""
    I_ss = s / mu
    immune_threshold = mu / rho if rho > 0 else 1e20
    kill_at_threshold = delta * immune_threshold * I_ss

    state = np.zeros(24, dtype=np.float64)

    # A: Disease (dims 0-7)
    state[0] = np.log(r)
    state[1] = K_ln / 10.0
    state[2] = np.log10(max(delta, 1e-20))
    state[3] = r * 0.5 * (1 - np.log(0.5 * K) / K_ln)
    state[4] = np.log10(K)
    state[5] = np.log10(max(kill_at_threshold, 1e-12))
    state[6] = r * 0.5
    state[7] = r - delta * I_ss  # stability eigenvalue

    # B: Host (dims 8-15)
    state[8]  = np.log10(s)
    state[9]  = np.log10(max(rho, 1e-20))
    state[10] = mu
    state[11] = np.log10(I_ss)
    state[12] = np.log10(max(immune_threshold, 1.0))
    state[13] = rho * I_ss
    state[14] = mu - rho * 1e6
    state[15] = np.log10(s / mu**2)

    # Context: Boundaries (dims 16-23)
    state[16] = 1.0
    state[17] = 1.0
    state[18] = immune_threshold / K if K > 0 else 0
    state[19] = 1.0
    state[20] = np.log(max(immune_threshold, 1.0)) / K_ln if K_ln > 0 else 0
    state[21] = np.sign(-(r - delta * I_ss))
    state[22] = 1.0 if immune_threshold < K else 0.0
    state[23] = np.log10(max(K / max(immune_threshold, 1.0), 1.0))

    # Normalize to target norm ~4.0 (same as Voodoo's conversation encoder)
    # Preserves all relative structure but keeps numbers in collapse sensitivity range
    norm = np.linalg.norm(state)
    if norm > 0.1:
        state = state * (4.0 / norm)

    return state


def analyze_point(delta, rho):
    """Run AOI collapse and extract key structural indicators."""
    state = encode_tumor_state(delta, rho)
    result = aoi_collapse(state)
    decomp = result['decomposition']
    jordan = decomp['jordan']
    comm = decomp['commutator']
    assoc = decomp['associator']

    fano_triples = [(1,2,4), (2,3,5), (3,4,6), (4,5,7), (5,6,1), (6,7,2), (7,1,3)]

    pos_routes = 0
    neg_routes = 0
    for i, j, k in fano_triples:
        activity = comm.v[i] * comm.v[j]
        if abs(activity) > 0.001:
            if comm.v[k] > 0:
                pos_routes += 1
            else:
                neg_routes += 1

    return {
        'chaos': result['normalized_chaos'],
        'intent': result['intent_magnitude'],
        'jordan_norm': jordan.norm(),
        'comm_norm': comm.norm(),
        'assoc_norm': assoc.norm(),
        'jordan_real': jordan.real,
        'e7_jordan': jordan.v[7],
        'e7_comm': comm.v[7],
        'e7_assoc': assoc.v[7],
        'pos_fano': pos_routes,
        'neg_fano': neg_routes,
        'stability_eigenvalue': r - delta * (s / mu),
        'control': result['control_vec'],
    }


# ============================================================
# SWEEP
# ============================================================

print("=" * 70)
print("  VOODOO AOI v3.0 -- Gompertz-Immune Parameter Sweep")
print("  Finding the algebraic phase transition to bounded attractor")
print("=" * 70)
print()
print(f"  Fixed: r={r}, K={K:.0e}, s={s:.0e}, mu={mu}")
print(f"  Sweeping: delta (immune killing), rho (immune activation)")
print()

# Sweep ranges — logarithmic
delta_range = np.logspace(-9, -3, 25)  # 1e-9 to 1e-3
rho_range   = np.logspace(-10, -4, 25)  # 1e-10 to 1e-4

# Store results
results = []
transitions = []

print("  Running sweep...")
print()

for delta_val in delta_range:
    for rho_val in rho_range:
        r_dict = analyze_point(delta_val, rho_val)
        r_dict['delta'] = delta_val
        r_dict['rho'] = rho_val
        results.append(r_dict)

print(f"  Completed {len(results)} parameter combinations.")
print()

# ============================================================
# FIND PHASE TRANSITION
# ============================================================

print("=" * 70)
print("  PHASE TRANSITION ANALYSIS")
print("=" * 70)
print()

# Classify each point
bounded = []
escape = []

for r_dict in results:
    # Bounded criteria:
    #   - e7 positive or near-zero in Jordan (rational stability)
    #   - Positive Fano routes >= negative
    #   - Chaos below 5.0
    #   - Jordan norm > Commutator norm (rational dominates)
    #   - Stability eigenvalue negative (tumor growth controlled)
    is_bounded = (
        r_dict['e7_jordan'] > -0.1 and
        r_dict['pos_fano'] >= r_dict['neg_fano'] and
        r_dict['chaos'] < 5.0 and
        r_dict['jordan_norm'] >= r_dict['comm_norm'] * 0.8
    )

    if is_bounded:
        bounded.append(r_dict)
    else:
        escape.append(r_dict)

print(f"  BOUNDED (remission-like attractor): {len(bounded)}/{len(results)} points")
print(f"  ESCAPE (uncontrolled growth):       {len(escape)}/{len(results)} points")
print()

# Find boundary points — bounded points with smallest delta*rho product
if bounded:
    bounded_sorted = sorted(bounded, key=lambda x: x['delta'] * x['rho'])

    print("  --- BOUNDARY OF BOUNDED REGIME (weakest immune params that still bound) ---")
    print()
    for pt in bounded_sorted[:10]:
        print(f"    delta={pt['delta']:.2e}  rho={pt['rho']:.2e}")
        print(f"      chaos={pt['chaos']:.2f}  e7_J={pt['e7_jordan']:+.4f}  "
              f"e7_C={pt['e7_comm']:+.4f}  e7_A={pt['e7_assoc']:+.4f}")
        print(f"      Fano +{pt['pos_fano']}/-{pt['neg_fano']}  "
              f"J_norm={pt['jordan_norm']:.4f}  C_norm={pt['comm_norm']:.4f}  "
              f"A_norm={pt['assoc_norm']:.4f}")
        print(f"      stability_eigenvalue={pt['stability_eigenvalue']:+.6f}")
        print()

    # Find the critical delta at different rho levels
    print("  --- CRITICAL DELTA BY RHO LEVEL ---")
    print()
    for rho_val in [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]:
        pts_at_rho = [p for p in results if abs(np.log10(p['rho']) - np.log10(rho_val)) < 0.3]
        bounded_at_rho = [p for p in pts_at_rho if (
            p['e7_jordan'] > -0.1 and
            p['pos_fano'] >= p['neg_fano'] and
            p['chaos'] < 5.0 and
            p['jordan_norm'] >= p['comm_norm'] * 0.8
        )]
        if bounded_at_rho:
            min_delta_pt = min(bounded_at_rho, key=lambda x: x['delta'])
            print(f"    rho ~ {rho_val:.0e}: bounded starts at delta >= {min_delta_pt['delta']:.2e}")
            print(f"      chaos={min_delta_pt['chaos']:.2f}  e7_J={min_delta_pt['e7_jordan']:+.4f}  "
                  f"Fano +{min_delta_pt['pos_fano']}/-{min_delta_pt['neg_fano']}")
        else:
            print(f"    rho ~ {rho_val:.0e}: NO bounded regime found in sweep range")
    print()

else:
    print("  NO BOUNDED POINTS FOUND — try wider parameter range")
    print()

# ============================================================
# DETAILED TRANSITION ANALYSIS
# ============================================================

print("=" * 70)
print("  STRUCTURAL CHANGES AT TRANSITION")
print("=" * 70)
print()

# Pick a rho and sweep delta to see the flip
rho_fixed = 1e-7
print(f"  Fixed rho = {rho_fixed:.0e}, sweeping delta:")
print()

delta_fine = np.logspace(-8, -3, 40)
prev = None

for delta_val in delta_fine:
    curr = analyze_point(delta_val, rho_fixed)
    curr['delta'] = delta_val

    is_bounded = (
        curr['e7_jordan'] > -0.1 and
        curr['pos_fano'] >= curr['neg_fano'] and
        curr['chaos'] < 5.0 and
        curr['jordan_norm'] >= curr['comm_norm'] * 0.8
    )

    if prev is not None:
        was_bounded = (
            prev['e7_jordan'] > -0.1 and
            prev['pos_fano'] >= prev['neg_fano'] and
            prev['chaos'] < 5.0 and
            prev['jordan_norm'] >= prev['comm_norm'] * 0.8
        )
        if was_bounded != is_bounded:
            direction = "ESCAPE -> BOUNDED" if is_bounded else "BOUNDED -> ESCAPE"
            print(f"  *** TRANSITION: {direction} ***")
            print(f"    Between delta = {prev['delta']:.4e} and {delta_val:.4e}")
            print()
            print(f"    BEFORE (delta={prev['delta']:.4e}):")
            print(f"      chaos={prev['chaos']:.2f}  intent={prev['intent']:.4f}")
            print(f"      e7: J={prev['e7_jordan']:+.4f}  C={prev['e7_comm']:+.4f}  A={prev['e7_assoc']:+.4f}")
            print(f"      Fano: +{prev['pos_fano']}/-{prev['neg_fano']}")
            print(f"      Norms: J={prev['jordan_norm']:.4f}  C={prev['comm_norm']:.4f}  A={prev['assoc_norm']:.4f}")
            print()
            print(f"    AFTER (delta={delta_val:.4e}):")
            print(f"      chaos={curr['chaos']:.2f}  intent={curr['intent']:.4f}")
            print(f"      e7: J={curr['e7_jordan']:+.4f}  C={curr['e7_comm']:+.4f}  A={curr['e7_assoc']:+.4f}")
            print(f"      Fano: +{curr['pos_fano']}/-{curr['neg_fano']}")
            print(f"      Norms: J={curr['jordan_norm']:.4f}  C={curr['comm_norm']:.4f}  A={curr['assoc_norm']:.4f}")
            print()

            # What changed structurally
            e7_flip = (prev['e7_jordan'] < 0 and curr['e7_jordan'] >= 0) or \
                      (prev['e7_jordan'] >= 0 and curr['e7_jordan'] < 0)
            fano_flip = (prev['pos_fano'] < prev['neg_fano']) != (curr['pos_fano'] < curr['neg_fano'])
            assoc_drop = curr['assoc_norm'] < prev['assoc_norm'] * 0.8

            print(f"    STRUCTURAL CHANGES:")
            if e7_flip:
                print(f"      * e7 SIGN FLIP in Jordan (stability axis reversed)")
            if fano_flip:
                print(f"      * FANO ROUTE BALANCE SHIFTED")
            if assoc_drop:
                print(f"      * ASSOCIATOR NORM DROPPED {(1 - curr['assoc_norm']/prev['assoc_norm'])*100:.1f}%")
            chaos_change = curr['chaos'] - prev['chaos']
            print(f"      * Chaos change: {chaos_change:+.2f}")
            jc_ratio_change = (curr['jordan_norm']/curr['comm_norm']) - (prev['jordan_norm']/prev['comm_norm'])
            print(f"      * Jordan/Commutator ratio change: {jc_ratio_change:+.4f}")
            print()

    prev = curr

# Print the full fine sweep
print()
print("  Full delta sweep at rho={:.0e}:".format(rho_fixed))
print("  {:>12s} {:>6s} {:>8s} {:>8s} {:>8s} {:>8s} {:>8s} {:>5s} {:>5s} {:>8s}".format(
    'delta', 'chaos', 'e7_J', 'e7_C', 'e7_A', 'J_norm', 'A_norm', '+F', '-F', 'status'))
print("  " + "-" * 90)

for delta_val in delta_fine:
    pt = analyze_point(delta_val, rho_fixed)
    is_b = (pt['e7_jordan'] > -0.1 and pt['pos_fano'] >= pt['neg_fano'] and
            pt['chaos'] < 5.0 and pt['jordan_norm'] >= pt['comm_norm'] * 0.8)
    status = "BOUND" if is_b else "ESCAPE"
    print("  {:>12.4e} {:>6.2f} {:>+8.4f} {:>+8.4f} {:>+8.4f} {:>8.4f} {:>8.4f} {:>5d} {:>5d} {:>8s}".format(
        delta_val, pt['chaos'], pt['e7_jordan'], pt['e7_comm'], pt['e7_assoc'],
        pt['jordan_norm'], pt['assoc_norm'], pt['pos_fano'], pt['neg_fano'], status))

print()
print("=" * 70)
print("  SWEEP COMPLETE")
print("=" * 70)
