"""
Voodoo vs Collatz — Deep Analysis
Feed actual hailstone sequences through the 24D collapse.
Look for patterns the algebra sees that standard analysis doesn't.
"""
import sys
import os
import time
import numpy as np

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, r"C:\Users\jimjj\Music\QuantumChildren\QuantumTradingLibrary")
from aoi_collapse import (
    Octonion, octonion_shadow_decompose, entropy_transponders, aoi_collapse
)


def collatz_sequence(n):
    """Generate full Collatz sequence from n to 1."""
    seq = [n]
    while n != 1:
        if n % 2 == 0:
            n = n // 2
        else:
            n = 3 * n + 1
        seq.append(n)
    return seq


def sequence_to_24d(seq):
    """Encode a Collatz sequence as a 24D state vector.

    This encodes the STRUCTURE of the sequence, not just summary stats.
    """
    arr = np.array(seq, dtype=np.float64)
    n = len(arr)
    start = arr[0]

    features = np.zeros(24, dtype=np.float64)

    # Dim 0-2: Scale properties (all log-scaled to keep norms sane)
    features[0] = np.log2(max(start, 1))            # starting number (log scale)
    features[1] = np.log2(max(np.max(arr), 1))       # peak value (log scale)
    features[2] = np.log2(max(n, 1))                 # total stopping time (log scale)

    # Dim 3-5: Parity structure
    parity = np.array([x % 2 for x in seq])
    features[3] = np.sum(parity) / n               # odd ratio
    features[4] = np.sum(1 - parity) / n            # even ratio
    # Parity autocorrelation (how predictable is the parity sequence?)
    if n > 3:
        parity_shifted = np.roll(parity, 1)
        try:
            c = np.corrcoef(parity[1:], parity_shifted[1:])[0, 1]
            features[5] = c if np.isfinite(c) else 0
        except Exception:
            features[5] = 0

    # Dim 6-8: Growth/decay dynamics
    ratios = arr[1:] / np.maximum(arr[:-1], 1)
    features[6] = np.mean(np.log2(np.maximum(ratios, 1e-10)))  # avg log ratio
    features[7] = np.std(np.log2(np.maximum(ratios, 1e-10)))   # volatility of ratios
    # Net growth: log2(peak) - log2(start)
    features[8] = np.log2(max(np.max(arr), 1)) - np.log2(max(start, 1))

    # Dim 9-11: The 3/4 heuristic — how well does it hold?
    odd_indices = [i for i in range(len(seq)-1) if seq[i] % 2 == 1]
    if odd_indices:
        odd_ratios = [seq[i+1] / seq[i] for i in odd_indices]
        features[9] = np.mean(odd_ratios)               # actual mean ratio at odd steps
        features[10] = features[9] - 1.5                 # deviation from 3/2 (3n+1 then /2 = ~1.5)
        features[11] = np.std(odd_ratios)                # spread

    # Dim 12-14: Binary structure
    binary_lens = [int(x).bit_length() for x in seq]
    features[12] = np.log2(max(np.mean(binary_lens), 1))   # average bit length (log)
    features[13] = np.log2(max(np.max(binary_lens) - np.min(binary_lens), 1))  # bit length range (log)
    # Trailing zeros (how often do we get big drops?)
    trailing = []
    for x in seq:
        ix = int(x)
        if ix > 0:
            trailing.append((ix & -ix).bit_length() - 1)
    features[14] = np.mean(trailing) if trailing else 0

    # Dim 15-17: Modular residues (mod 3, mod 6, mod 12)
    mod3 = [int(x) % 3 for x in seq]
    mod6 = [int(x) % 6 for x in seq]
    features[15] = mod3.count(0) / n                     # divisible by 3 frequency
    features[16] = np.mean(mod6)                         # mean mod-6 residue
    features[17] = len(set(mod6)) / 6.0                  # mod-6 coverage

    # Dim 18-20: Subsequence patterns
    # Count consecutive rises vs falls
    rises = sum(1 for i in range(n-1) if arr[i+1] > arr[i])
    falls = sum(1 for i in range(n-1) if arr[i+1] < arr[i])
    features[18] = rises / max(n-1, 1)
    features[19] = falls / max(n-1, 1)
    features[20] = (rises - falls) / max(n-1, 1)        # rise/fall asymmetry

    # Dim 21-23: Deep structure
    # Spectral: FFT of log-sequence
    if n > 4:
        log_seq = np.log2(np.maximum(arr, 1))
        fft = np.fft.rfft(log_seq - np.mean(log_seq))
        power = np.abs(fft) ** 2
        total_power = np.sum(power)
        if total_power > 0:
            features[21] = power[1] / total_power        # fundamental frequency weight
            features[22] = np.argmax(power[1:]) + 1      # dominant frequency
        # Entropy of the power spectrum
        p_norm = power / max(total_power, 1e-12)
        p_norm = p_norm[p_norm > 1e-12]
        features[23] = -np.sum(p_norm * np.log2(p_norm))  # spectral entropy

    # Replace any NaN/inf with 0
    features = np.nan_to_num(features, nan=0.0, posinf=3.0, neginf=-3.0)

    # Normalize to unit norm to prevent floating point blowup in octonion math
    norm = np.linalg.norm(features)
    if norm > 1e-12:
        features = features / norm * np.sqrt(24)  # scale so avg component ~ 1

    return features


def print_section(title):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)
    print()


if __name__ == '__main__':
    print_section("VOODOO vs COLLATZ -- Deep Sequence Analysis")

    # ============================================================
    # Phase 1: Run collapse on individual sequences
    # ============================================================

    test_numbers = [
        7, 27, 97, 871, 6171, 77031, 837799,  # record holders for stopping time
        2, 4, 8, 16, 32, 64,                    # powers of 2 (trivial)
        3, 9, 27, 81, 243,                       # powers of 3
        5, 17, 65, 257,                           # 2^k + 1 forms
    ]

    results = {}

    print("Phase 1: Individual sequence collapses")
    print("-" * 60)

    for n in test_numbers:
        seq = collatz_sequence(n)
        state = sequence_to_24d(seq)
        collapse = aoi_collapse(state)
        results[n] = {
            'seq_len': len(seq),
            'peak': max(seq),
            'chaos': collapse['chaos_level'],
            'nchaos': collapse['normalized_chaos'],
            'intent': collapse['intent_magnitude'],
            'control': collapse['control_vec'].copy(),
            'control_norm': collapse['control_norm'],
            'personality': collapse['personality_embedding'].copy(),
            'decomp': collapse['decomposition'],
        }

        j = collapse['decomposition']['jordan']
        c = collapse['decomposition']['commutator']
        a = collapse['decomposition']['associator']
        jc_ratio = j.norm() / max(c.norm(), 1e-12)
        aj_ratio = a.norm() / max(j.norm(), 1e-12)

        print(f"  n={n:>8d}  steps={len(seq):>4d}  peak={max(seq):>10d}  "
              f"chaos={collapse['chaos_level']:>10.2f}  J/C={jc_ratio:.4f}  A/J={aj_ratio:.4f}")

    # ============================================================
    # Phase 2: Compare record holders vs trivial sequences
    # ============================================================

    print_section("Phase 2: Record Holders vs Powers of 2")

    records = [7, 27, 97, 871, 6171, 77031, 837799]
    powers = [2, 4, 8, 16, 32, 64]

    record_chaos = [results[n]['chaos'] for n in records]
    power_chaos = [results[n]['chaos'] for n in powers]

    print(f"  Record holders avg chaos:  {np.mean(record_chaos):.4f}")
    print(f"  Powers of 2 avg chaos:     {np.mean(power_chaos):.4f}")
    print(f"  Ratio:                     {np.mean(record_chaos)/max(np.mean(power_chaos),1e-12):.4f}x")
    print()

    # Compare personality embeddings
    record_personalities = np.array([results[n]['personality'] for n in records])
    power_personalities = np.array([results[n]['personality'] for n in powers])

    print("  Personality embedding comparison (8D associator):")
    print("  Dim | Records mean | Powers mean | Difference")
    print("  ----|-------------|-------------|----------")
    for i in range(8):
        rm = np.mean(record_personalities[:, i])
        pm = np.mean(power_personalities[:, i])
        diff = rm - pm
        marker = " ***" if abs(diff) > max(abs(rm), abs(pm)) * 0.5 else ""
        print(f"  e{i}  | {rm:+11.4f} | {pm:+11.4f} | {diff:+11.4f}{marker}")

    # ============================================================
    # Phase 3: Look for structure in the decomposition
    # ============================================================

    print_section("Phase 3: Decomposition Structure Search")

    # Run collapse on a range of consecutive numbers
    print("  Collapsing n=1 to n=500...")
    all_chaos = []
    all_jc_ratios = []
    all_aj_ratios = []
    all_jordan_reals = []
    all_dominant_axes = []
    all_intents = []

    for n in range(1, 501):
        seq = collatz_sequence(n)
        state = sequence_to_24d(seq)
        collapse = aoi_collapse(state)

        j = collapse['decomposition']['jordan']
        c = collapse['decomposition']['commutator']
        a = collapse['decomposition']['associator']

        all_chaos.append(collapse['chaos_level'])
        all_jc_ratios.append(j.norm() / max(c.norm(), 1e-12))
        all_aj_ratios.append(a.norm() / max(j.norm(), 1e-12))
        all_jordan_reals.append(j.real)
        all_dominant_axes.append(np.argmax(np.abs(collapse['personality_embedding'])))
        all_intents.append(collapse['intent_magnitude'])

    all_chaos = np.array(all_chaos)
    all_jc_ratios = np.array(all_jc_ratios)
    all_aj_ratios = np.array(all_aj_ratios)
    all_jordan_reals = np.array(all_jordan_reals)
    all_intents = np.array(all_intents)

    print(f"  Done. Analyzing patterns...")
    print()

    # Chaos distribution
    print("  Chaos level distribution (n=1..500):")
    percentiles = [10, 25, 50, 75, 90]
    for p in percentiles:
        print(f"    {p}th percentile: {np.percentile(all_chaos, p):.4f}")
    print()

    # J/C ratio distribution
    print("  Jordan/Commutator ratio distribution:")
    for p in percentiles:
        print(f"    {p}th percentile: {np.percentile(all_jc_ratios, p):.4f}")
    print()

    # Dominant axis distribution
    print("  Dominant personality axis distribution:")
    axis_counts = np.bincount(np.array(all_dominant_axes), minlength=8)
    for i in range(8):
        bar = "#" * (axis_counts[i] // 5) + "." * max(0, 20 - axis_counts[i] // 5)
        print(f"    e{i}: {axis_counts[i]:>4d} |{bar}|")
    print()

    # ============================================================
    # Phase 4: Correlations -- what predicts what?
    # ============================================================

    print_section("Phase 4: What Does the Algebra Correlate With?")

    # Get stopping times and peaks for n=1..500
    stopping_times = []
    peaks = []
    odd_ratios_mean = []

    for n in range(1, 501):
        seq = collatz_sequence(n)
        stopping_times.append(len(seq))
        peaks.append(max(seq))
        odd_idx = [i for i in range(len(seq)-1) if seq[i] % 2 == 1]
        if odd_idx:
            orats = [seq[i+1]/seq[i] for i in odd_idx]
            odd_ratios_mean.append(np.mean(orats))
        else:
            odd_ratios_mean.append(0)

    stopping_times = np.array(stopping_times, dtype=float)
    peaks = np.array(peaks, dtype=float)
    odd_ratios_mean = np.array(odd_ratios_mean)

    # Correlations
    correlations = {
        'chaos vs stopping_time': np.corrcoef(all_chaos, stopping_times)[0, 1],
        'chaos vs peak': np.corrcoef(all_chaos, np.log2(np.maximum(peaks, 1)))[0, 1],
        'J/C ratio vs stopping_time': np.corrcoef(all_jc_ratios, stopping_times)[0, 1],
        'J/C ratio vs peak': np.corrcoef(all_jc_ratios, np.log2(np.maximum(peaks, 1)))[0, 1],
        'A/J ratio vs stopping_time': np.corrcoef(all_aj_ratios, stopping_times)[0, 1],
        'intent vs stopping_time': np.corrcoef(all_intents, stopping_times)[0, 1],
        'jordan_real vs stopping_time': np.corrcoef(all_jordan_reals, stopping_times)[0, 1],
        'jordan_real vs odd_ratio': np.corrcoef(all_jordan_reals, odd_ratios_mean)[0, 1],
    }

    print("  Pearson correlations:")
    for name, corr in sorted(correlations.items(), key=lambda x: -abs(x[1])):
        strength = ""
        if abs(corr) > 0.8:
            strength = " *** STRONG"
        elif abs(corr) > 0.5:
            strength = " ** MODERATE"
        elif abs(corr) > 0.3:
            strength = " * WEAK"
        print(f"    {name:40s}  r = {corr:+.4f}{strength}")
    print()

    # ============================================================
    # Phase 5: The key question -- can the decomposition
    # distinguish sequences that "almost diverge" from normal ones?
    # ============================================================

    print_section("Phase 5: Near-Divergence Detection")

    # Find numbers with highest peak-to-start ratios (they "almost" diverge)
    ratios_peak = peaks / np.arange(1, 501, dtype=float)
    top_volatile = np.argsort(-ratios_peak)[:10]
    bottom_volatile = np.argsort(ratios_peak)[:10]

    print("  Top 10 most volatile (highest peak/start ratio):")
    print(f"  {'n':>6s}  {'peak/n':>10s}  {'chaos':>10s}  {'J/C':>8s}  {'A/J':>8s}  {'intent':>8s}")
    for idx in top_volatile:
        n = idx + 1
        print(f"  {n:>6d}  {ratios_peak[idx]:>10.1f}  {all_chaos[idx]:>10.2f}  "
              f"{all_jc_ratios[idx]:>8.4f}  {all_aj_ratios[idx]:>8.4f}  {all_intents[idx]:>8.4f}")

    print()
    print("  Bottom 10 least volatile (lowest peak/start ratio):")
    print(f"  {'n':>6s}  {'peak/n':>10s}  {'chaos':>10s}  {'J/C':>8s}  {'A/J':>8s}  {'intent':>8s}")
    for idx in bottom_volatile:
        n = idx + 1
        print(f"  {n:>6d}  {ratios_peak[idx]:>10.1f}  {all_chaos[idx]:>10.2f}  "
              f"{all_jc_ratios[idx]:>8.4f}  {all_aj_ratios[idx]:>8.4f}  {all_intents[idx]:>8.4f}")

    # ============================================================
    # Phase 6: The Octonion Angle -- non-associativity signature
    # ============================================================

    print_section("Phase 6: Non-Associativity Signature")
    print("  This is what standard analysis CAN'T do.")
    print("  The octonion product is non-associative: (AB)C != A(BC)")
    print("  The associator measures this. If the Collatz dynamics")
    print("  have non-associative structure, it shows up HERE.")
    print()

    # Compare associator patterns between odd and even starting numbers
    odd_starts = list(range(1, 500, 2))
    even_starts = list(range(2, 501, 2))

    odd_assoc_norms = [all_chaos[n-1] for n in odd_starts]
    even_assoc_norms = [all_chaos[n-1] for n in even_starts]

    print(f"  Odd starts  - mean chaos: {np.mean(odd_assoc_norms):.4f}  std: {np.std(odd_assoc_norms):.4f}")
    print(f"  Even starts - mean chaos: {np.mean(even_assoc_norms):.4f}  std: {np.std(even_assoc_norms):.4f}")
    print(f"  Ratio (odd/even): {np.mean(odd_assoc_norms)/max(np.mean(even_assoc_norms),1e-12):.4f}")
    print()

    # Look at mod-3 classes (crucial for Collatz)
    mod3_classes = {0: [], 1: [], 2: []}
    for n in range(1, 501):
        mod3_classes[n % 3].append(all_chaos[n-1])

    print("  By mod-3 class (3 is special in Collatz: 3n+1):")
    for r in [0, 1, 2]:
        vals = mod3_classes[r]
        print(f"    n = {r} mod 3: mean chaos = {np.mean(vals):.4f}  std = {np.std(vals):.4f}  count = {len(vals)}")
    print()

    # The big question: do the personality embeddings cluster?
    print("  Personality embedding clustering (k=3):")
    all_personalities = []
    for n in range(1, 501):
        seq = collatz_sequence(n)
        state = sequence_to_24d(seq)
        collapse = aoi_collapse(state)
        all_personalities.append(collapse['personality_embedding'])

    all_personalities = np.array(all_personalities)

    # Simple k-means by hand (no sklearn dependency)
    k = 3
    rng = np.random.default_rng(42)
    centroids = all_personalities[rng.choice(500, k, replace=False)]

    for iteration in range(20):
        # Assign
        dists = np.array([[np.linalg.norm(p - c) for c in centroids] for p in all_personalities])
        labels = np.argmin(dists, axis=1)
        # Update
        new_centroids = np.array([all_personalities[labels == i].mean(axis=0) if np.sum(labels == i) > 0
                                   else centroids[i] for i in range(k)])
        if np.allclose(centroids, new_centroids, atol=1e-8):
            break
        centroids = new_centroids

    print(f"  Converged in {iteration+1} iterations")
    for i in range(k):
        members = np.where(labels == i)[0] + 1  # +1 for 1-indexed
        cluster_stops = [len(collatz_sequence(n)) for n in members[:20]]
        print(f"    Cluster {i}: {np.sum(labels == i)} members")
        print(f"      Centroid dominant axis: e{np.argmax(np.abs(centroids[i]))}")
        print(f"      Sample members: {list(members[:10])}")
        print(f"      Avg stopping time (sample): {np.mean(cluster_stops):.1f}")

        # What mod classes are in this cluster?
        mod3_dist = [0, 0, 0]
        for m in members:
            mod3_dist[m % 3] += 1
        total = max(sum(mod3_dist), 1)
        print(f"      Mod-3 distribution: 0={mod3_dist[0]/total:.1%}  1={mod3_dist[1]/total:.1%}  2={mod3_dist[2]/total:.1%}")

    # ============================================================
    # Phase 7: The Finding
    # ============================================================

    print_section("Phase 7: What Did Voodoo Find?")

    # Compute the most anomalous numbers by chaos
    chaos_mean = np.mean(all_chaos)
    chaos_std = np.std(all_chaos)
    z_scores = (all_chaos - chaos_mean) / max(chaos_std, 1e-12)

    anomalies = np.where(np.abs(z_scores) > 2.0)[0]
    print(f"  Numbers with chaos > 2 sigma from mean: {len(anomalies)}")
    if len(anomalies) > 0:
        print(f"  High chaos anomalies (numbers that 'feel' different to Voodoo):")
        high = anomalies[z_scores[anomalies] > 0]
        for idx in high[:15]:
            n = idx + 1
            seq = collatz_sequence(n)
            print(f"    n={n:>4d}  steps={len(seq):>4d}  peak={max(seq):>10d}  "
                  f"chaos_z={z_scores[idx]:+.2f}  dominant=e{all_dominant_axes[idx]}")

    print()

    # Final summary
    print("  SUMMARY:")

    # Find the strongest correlation
    best_corr_name = max(correlations, key=lambda x: abs(correlations[x]))
    best_corr_val = correlations[best_corr_name]

    print(f"  - Strongest correlation found: {best_corr_name} (r={best_corr_val:+.4f})")
    print(f"  - Chaos level range: {np.min(all_chaos):.4f} to {np.max(all_chaos):.4f}")
    print(f"  - Dominant axis most common: e{np.argmax(axis_counts)}")
    print(f"  - Clusters found: {k} with sizes {[int(np.sum(labels == i)) for i in range(k)]}")

    # Does mod-3 class matter to the algebra?
    mod3_means = [np.mean(mod3_classes[r]) for r in [0, 1, 2]]
    if max(mod3_means) / max(min(mod3_means), 1e-12) > 1.5:
        print(f"  - MOD-3 MATTERS: chaos differs significantly across mod-3 classes")
        print(f"    This means the octonion algebra is detecting the 3n+1 structure")
    else:
        print(f"  - Mod-3 classes show similar chaos (ratio: {max(mod3_means)/max(min(mod3_means),1e-12):.2f})")

    print()
    print("  Done.")
    print()
