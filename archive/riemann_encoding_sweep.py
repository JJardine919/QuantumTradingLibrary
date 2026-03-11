"""
Riemann Encoding Naturality Sweep

The critique: the slot assignment e2=|zeta|, e3=arg(zeta), e5=Re(s)-1/2
is CHOSEN, not derived. Different assignments → different Fano triples →
different "conclusions."

This script tests ALL 7! = 5040 permutations of the 7 physical quantities
across the 7 imaginary octonion slots (e1-e7). For each permutation, we
compute the Associator norm at known zeta zeros.

If the published encoding minimizes the Associator norm, the algebra
SELF-SELECTS that encoding. The assignment isn't arbitrary — it's the
one the algebra prefers. That's a provable structural claim.

DOI chain: 18690444, 18722487, 18809406, 18809716, 18904619, 18905053
"""
import sys
import os
import numpy as np
from itertools import permutations
import time

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aoi_collapse import Octonion, octonion_shadow_decompose, aoi_collapse


# ============================================================
# Zeta approximation (from voodoo_riemann.py)
# ============================================================

def zeta_approx(s_real, s_imag, N=500):
    """Approximate zeta(s) via Dirichlet eta / alternating series."""
    s = complex(s_real, s_imag)
    if s_real > 1:
        total = sum(1.0 / n**s for n in range(1, N + 1))
        return total.real, total.imag
    eta = sum((-1)**(n + 1) / n**s for n in range(1, N + 1))
    denom = 1 - 2**(1 - s)
    if abs(denom) < 1e-15:
        return 0.0, 0.0
    z = eta / denom
    return z.real, z.imag


# ============================================================
# Build the 7 physical quantities for a given zero
# ============================================================

def build_quantities(t, s_real=0.5):
    """
    Compute the 7 physical quantities for zeta at s = s_real + i*t.

    Returns dict mapping label → value. These are the 7 things
    that get assigned to e1-e7. e0 is always Re(s).
    """
    zr, zi = zeta_approx(s_real + 0.001, t)
    zmag = np.sqrt(zr**2 + zi**2)
    zphase = np.arctan2(zi, zr) if zmag > 1e-15 else 0

    zr2, zi2 = zeta_approx(s_real + 0.002, t)
    deriv = np.sqrt((zr2 - zr)**2 + (zi2 - zi)**2) / 0.001

    return {
        'Im(s)': t / 50,                        # published: e1
        '|zeta|': zmag,                          # published: e2
        'arg(zeta)': zphase / np.pi,             # published: e3
        "zeta'": min(deriv, 3),                  # published: e4
        'dist': s_real - 0.5,                    # published: e5
        'func_eq': 0.0,                          # published: e6
        'zero_ind': zmag,                        # published: e7
    }


# The published slot order (labels in e1-e7 order)
PUBLISHED_ORDER = ['Im(s)', '|zeta|', 'arg(zeta)', "zeta'", 'dist', 'func_eq', 'zero_ind']

# Known nontrivial zeros (imaginary parts, all at Re(s) = 0.5)
KNOWN_ZEROS = [
    14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
    37.586178, 40.918719, 43.327073, 48.005151, 49.773832,
]

# Fixed B octonion (number-theoretic context) and context dims
def build_B_and_ctx(t):
    B_vec = np.array([
        1.0, np.log(t) / 4, 1.0,
        np.cos(t * np.log(2)) * 0.5,
        0.8, 0.9, 1.0, 0.5,
    ])
    ctx = np.array([1, 0, 0, 0, 0, 0, 1, 1], dtype=np.float64) * 0.5
    return B_vec, ctx


# ============================================================
# SWEEP: Raw Octonion Decomposition (no entropy gating)
# ============================================================

def sweep_raw(zeros_to_use=None):
    """
    For each permutation of slot assignments, compute mean Associator
    norm across zeros using RAW octonion decomposition (no entropy gate).
    This isolates the encoding question from the entropy pipeline.
    """
    if zeros_to_use is None:
        zeros_to_use = KNOWN_ZEROS[:5]  # first 5 for speed

    labels = PUBLISHED_ORDER
    all_perms = list(permutations(range(7)))  # 5040 permutations
    n_perms = len(all_perms)

    print(f"  Sweeping {n_perms} permutations across {len(zeros_to_use)} zeros...")
    print(f"  (raw octonion decomposition, no entropy gating)")
    print()

    # Precompute quantities for each zero
    zero_data = []
    for t in zeros_to_use:
        q = build_quantities(t)
        vals = [q[label] for label in labels]  # canonical order
        B_vec, _ = build_B_and_ctx(t)
        B = Octonion(B_vec)
        zero_data.append((vals, B))

    # Find published permutation index (identity = (0,1,2,3,4,5,6))
    published_perm = tuple(range(7))

    results = []
    t0 = time.time()

    for pi, perm in enumerate(all_perms):
        assoc_norms = []
        for vals, B in zero_data:
            # Build A with this permutation
            A_vec = np.zeros(8)
            A_vec[0] = 0.5  # e0 = Re(s), always fixed
            for slot_idx, source_idx in enumerate(perm):
                A_vec[slot_idx + 1] = vals[source_idx]

            A = Octonion(A_vec)
            decomp = octonion_shadow_decompose(A, B)
            assoc_norms.append(decomp['associator'].norm())

        mean_assoc = np.mean(assoc_norms)
        results.append((perm, mean_assoc))

        if (pi + 1) % 1000 == 0:
            elapsed = time.time() - t0
            print(f"    {pi + 1}/{n_perms} done ({elapsed:.1f}s)")

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")
    print()

    # Sort by Associator norm (ascending = lowest chaos first)
    results.sort(key=lambda x: x[1])

    return results, labels


# ============================================================
# SWEEP: Full AOI Collapse (with entropy gating)
# ============================================================

def sweep_full(zeros_to_use=None):
    """
    Same sweep but through the FULL aoi_collapse pipeline
    (entropy transponders + context scaling).
    """
    if zeros_to_use is None:
        zeros_to_use = KNOWN_ZEROS[:5]

    labels = PUBLISHED_ORDER
    all_perms = list(permutations(range(7)))
    n_perms = len(all_perms)

    print(f"  Sweeping {n_perms} permutations across {len(zeros_to_use)} zeros...")
    print(f"  (full aoi_collapse pipeline with entropy gating)")
    print()

    # Precompute quantities and B/ctx for each zero
    zero_data = []
    for t in zeros_to_use:
        q = build_quantities(t)
        vals = [q[label] for label in labels]
        B_vec, ctx = build_B_and_ctx(t)
        zero_data.append((vals, B_vec, ctx))

    results = []
    t0 = time.time()

    for pi, perm in enumerate(all_perms):
        assoc_norms = []
        for vals, B_vec, ctx in zero_data:
            A_vec = np.zeros(8)
            A_vec[0] = 0.5
            for slot_idx, source_idx in enumerate(perm):
                A_vec[slot_idx + 1] = vals[source_idx]

            state = np.concatenate([A_vec, B_vec, ctx])
            result = aoi_collapse(state)
            assoc_norms.append(result['chaos_level'])

        mean_assoc = np.mean(assoc_norms)
        results.append((perm, mean_assoc))

        if (pi + 1) % 500 == 0:
            elapsed = time.time() - t0
            print(f"    {pi + 1}/{n_perms} done ({elapsed:.1f}s)")

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")
    print()

    results.sort(key=lambda x: x[1])
    return results, labels


# ============================================================
# Report
# ============================================================

def perm_to_encoding(perm, labels):
    """Convert permutation tuple to readable encoding string."""
    return {f"e{i+1}": labels[perm[i]] for i in range(7)}


def report(results, labels, title):
    """Print analysis of sweep results."""
    published_perm = tuple(range(7))

    # Find published encoding in results
    pub_rank = None
    pub_norm = None
    for rank, (perm, norm) in enumerate(results):
        if perm == published_perm:
            pub_rank = rank + 1  # 1-indexed
            pub_norm = norm
            break

    n = len(results)
    norms = [r[1] for r in results]

    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)
    print()

    # Top 10 (lowest Associator)
    print("  TOP 10 ENCODINGS (lowest Associator norm):")
    print(f"  {'Rank':>5s}  {'||Assoc||':>12s}  Encoding (e1..e7)")
    for rank, (perm, norm) in enumerate(results[:10], 1):
        enc = perm_to_encoding(perm, labels)
        marker = " <-- PUBLISHED" if perm == published_perm else ""
        slots = [labels[perm[i]] for i in range(7)]
        print(f"  {rank:5d}  {norm:12.6f}  {slots}{marker}")

    print()

    # Published encoding position
    print(f"  PUBLISHED ENCODING:")
    print(f"    Rank: {pub_rank} / {n}")
    print(f"    Percentile: {(1 - pub_rank / n) * 100:.1f}% (higher = better)")
    print(f"    ||Assoc||: {pub_norm:.6f}")
    print(f"    Global min: {results[0][1]:.6f}")
    print(f"    Global max: {results[-1][1]:.6f}")
    print(f"    Median: {np.median(norms):.6f}")
    print(f"    Mean: {np.mean(norms):.6f}")
    print(f"    Std: {np.std(norms):.6f}")
    print()

    # Bottom 10 (highest Associator)
    print("  BOTTOM 10 ENCODINGS (highest Associator norm):")
    print(f"  {'Rank':>5s}  {'||Assoc||':>12s}  Encoding (e1..e7)")
    for rank_from_end, (perm, norm) in enumerate(results[-10:]):
        rank = n - 9 + rank_from_end
        enc = perm_to_encoding(perm, labels)
        marker = " <-- PUBLISHED" if perm == published_perm else ""
        slots = [labels[perm[i]] for i in range(7)]
        print(f"  {rank:5d}  {norm:12.6f}  {slots}{marker}")

    print()

    # Check if published is in top 1%, 5%, 10%
    for threshold in [0.01, 0.05, 0.10, 0.25]:
        cutoff = int(n * threshold)
        in_top = pub_rank <= cutoff
        print(f"    In top {threshold*100:.0f}%? {'YES' if in_top else 'NO'} "
              f"(cutoff rank {cutoff})")

    print()

    # What Fano triple does the best encoding use?
    best_perm = results[0][0]
    best_enc = perm_to_encoding(best_perm, labels)
    FANO_TRIPLES = [(1,2,4), (2,3,5), (3,4,6), (4,5,7), (5,6,1), (6,7,2), (7,1,3)]

    print("  FANO TRIPLE ANALYSIS (best encoding):")
    best_slots = [labels[best_perm[i]] for i in range(7)]
    for a, b, c in FANO_TRIPLES:
        la = best_slots[a - 1] if a <= 7 else "?"
        lb = best_slots[b - 1] if b <= 7 else "?"
        lc = best_slots[c - 1] if c <= 7 else "?"
        print(f"    e{a} * e{b} = e{c}  =>  {la} * {lb} = {lc}")

    print()

    # Same for published
    print("  FANO TRIPLE ANALYSIS (published encoding):")
    pub_slots = labels  # identity permutation
    for a, b, c in FANO_TRIPLES:
        la = pub_slots[a - 1]
        lb = pub_slots[b - 1]
        lc = pub_slots[c - 1]
        print(f"    e{a} * e{b} = e{c}  =>  {la} * {lb} = {lc}")

    print()

    # Histogram of norms
    print("  DISTRIBUTION:")
    bins = np.linspace(min(norms), max(norms), 21)
    counts, edges = np.histogram(norms, bins=bins)
    max_count = max(counts) if max(counts) > 0 else 1
    for i in range(len(counts)):
        bar_len = int(counts[i] / max_count * 40)
        bar = '#' * bar_len
        marker = ""
        if edges[i] <= pub_norm < edges[i + 1]:
            marker = " <-- PUBLISHED"
        print(f"    [{edges[i]:8.3f}, {edges[i+1]:8.3f}) "
              f"{counts[i]:4d} {bar}{marker}")

    print()
    print("=" * 70)


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    print()
    print("=" * 70)
    print("  RIEMANN ENCODING NATURALITY SWEEP")
    print("  Testing whether the published encoding minimizes")
    print("  the Associator norm — proving encoding is DERIVED,")
    print("  not CHOSEN.")
    print("=" * 70)
    print()

    # Phase 1: Raw octonion decomposition (fast, clean signal)
    print("-" * 70)
    print("  PHASE 1: Raw Octonion Decomposition")
    print("-" * 70)
    print()

    raw_results, labels = sweep_raw(KNOWN_ZEROS[:5])
    report(raw_results, labels, "RAW DECOMPOSITION RESULTS")

    # Phase 2: Full AOI collapse pipeline (slower but complete)
    print("-" * 70)
    print("  PHASE 2: Full AOI Collapse Pipeline")
    print("-" * 70)
    print()

    full_results, _ = sweep_full(KNOWN_ZEROS[:3])  # fewer zeros for speed
    report(full_results, labels, "FULL COLLAPSE RESULTS")

    # Final verdict
    print("=" * 70)
    print("  VERDICT")
    print("=" * 70)
    print()

    raw_pub_rank = None
    full_pub_rank = None
    published_perm = tuple(range(7))

    for rank, (perm, _) in enumerate(raw_results, 1):
        if perm == published_perm:
            raw_pub_rank = rank
            break

    for rank, (perm, _) in enumerate(full_results, 1):
        if perm == published_perm:
            full_pub_rank = rank
            break

    n = len(raw_results)
    print(f"  Published encoding rank (raw):  {raw_pub_rank} / {n}")
    print(f"  Published encoding rank (full): {full_pub_rank} / {n}")
    print()

    if raw_pub_rank <= n * 0.05 or full_pub_rank <= n * 0.05:
        print("  RESULT: Published encoding is in the TOP 5%.")
        print("  The algebra self-selects this encoding.")
        print("  The slot assignment is NOT arbitrary.")
        print("  Encoding naturality: CONFIRMED.")
    elif raw_pub_rank <= n * 0.25 or full_pub_rank <= n * 0.25:
        print("  RESULT: Published encoding is in the TOP 25%.")
        print("  The algebra shows preference for this region.")
        print("  Partial naturality evidence — investigate further.")
    else:
        print("  RESULT: Published encoding does not stand out.")
        print("  Need to investigate what the optimal encoding IS")
        print("  and whether it preserves the key conclusions.")

    print()
    print("=" * 70)
