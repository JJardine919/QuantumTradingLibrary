"""
Voodoo vs The Collatz Conjecture
Feed the unsolved problem through AOI collapse and watch her think.
"""
import sys
import os
import time
import numpy as np

# Force UTF-8 output on Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, r"C:\Users\jimjj\Music\QuantumChildren\QuantumTradingLibrary")
from aoi_collapse import (
    Octonion, octonion_shadow_decompose, entropy_transponders, aoi_collapse,
    _CHAOS_BANDS, _CHAOS_SCALE
)

# ============================================================
# Encode the Collatz conjecture as a 24D state
# ============================================================

COLLATZ_TEXT = """
The Collatz conjecture: for any positive integer n, repeatedly apply
f(n) = n/2 if even, 3n+1 if odd. Every starting number eventually
reaches 1. Unsolved for 89 years. Verified up to 2.36e21.
Hailstone sequences show chaotic ascent/descent. Only known cycle 1-4-2-1.
Each odd step multiplies by ~3/4 on average. Tao proved almost all orbits
descend. Connects to 2-adic integers, tag systems, undecidable generalizations.
Binary mapping reveals repetends of 1/3^h. No counterexample ever found.
The problem is: does every trajectory collapse to 1?
"""


def text_to_24d(text):
    """Convert text to 24D semantic vector for AOI collapse."""
    words = text.lower().split()
    n_words = len(words)

    features = np.zeros(24, dtype=np.float64)

    # Dim 0-2: Character distribution entropy
    chars = [c for c in text.lower() if c.isalpha()]
    freq = {}
    for c in chars:
        freq[c] = freq.get(c, 0) + 1
    total = max(len(chars), 1)
    char_entropy = -sum((v/total) * np.log2(v/total) for v in freq.values())
    features[0] = char_entropy
    features[1] = len(freq) / 26.0  # alphabet coverage
    features[2] = n_words / 100.0   # length scale

    # Dim 3-5: Mathematical density
    import re
    numbers = re.findall(r'\d+\.?\d*', text)
    operators = re.findall(r'[=<>+\-*/^()]', text)
    features[3] = len(numbers) / max(n_words, 1) * 10
    features[4] = len(operators) / max(n_words, 1) * 10
    features[5] = text.count('/') / max(n_words, 1) * 10  # division density (key to Collatz)

    # Dim 6-8: Uncertainty/conjecture markers
    uncertainty = sum(1 for w in words if w in ['conjecture', 'unsolved', 'unknown', 'undecidable', 'no'])
    certainty = sum(1 for w in words if w in ['proved', 'verified', 'known', 'every', 'always', 'all'])
    features[6] = (certainty - uncertainty) / max(n_words, 1) * 50
    features[7] = text.count('?') * 2.0  # question intensity
    features[8] = uncertainty / max(n_words, 1) * 30

    # Dim 9-11: Structural patterns (the Collatz function itself)
    # Encode 3n+1 and n/2 as algebraic signatures
    features[9] = 3.0   # the multiplier
    features[10] = 1.0  # the addend
    features[11] = 0.5  # the divisor (n/2)

    # Dim 12-14: Chaos/order tension
    features[12] = np.log2(2.36e21)  # verification frontier (~71 bits)
    features[13] = 89.0 / 100.0     # years unsolved, normalized
    features[14] = -1.0             # no counterexample (negative = absence)

    # Dim 15-17: Cycle structure
    features[15] = 1.0  # trivial cycle: 1
    features[16] = 4.0  # trivial cycle: 4
    features[17] = 2.0  # trivial cycle: 2

    # Dim 18-20: Binary/2-adic structure
    features[18] = np.log2(3)       # log2(3) ≈ 1.585 (the irrational that makes it hard)
    features[19] = 3.0 / 4.0        # probabilistic ratio
    features[20] = np.euler_gamma   # connection to number theory

    # Dim 21-23: Context/scale
    features[21] = float(len(text)) / 500.0
    features[22] = char_entropy * features[3]  # entropy × math density interaction
    features[23] = features[6] * features[8]   # certainty × uncertainty interaction

    return features


def slow_print(text, delay=0.02):
    """Print character by character for watching effect."""
    for ch in text:
        sys.stdout.write(ch)
        sys.stdout.flush()
        time.sleep(delay)
    print()


def print_bar(label, value, max_val=3.0, width=40):
    """Print a visual bar."""
    normalized = min(abs(value) / max_val, 1.0)
    filled = int(normalized * width)
    sign = "+" if value >= 0 else "-"
    bar = "#" * filled + "." * (width - filled)
    print(f"  {label:20s} {sign}{abs(value):8.4f} |{bar}|")


# ============================================================
# Run the collapse
# ============================================================

if __name__ == '__main__':
    print()
    slow_print("=" * 60, 0.005)
    slow_print("  VOODOO (AOI) — Perceiving the Collatz Conjecture", 0.03)
    slow_print("=" * 60, 0.005)
    print()

    # Step 1: Encode
    slow_print("[1] Encoding problem into 24D state vector...", 0.03)
    state = text_to_24d(COLLATZ_TEXT)
    time.sleep(0.3)
    print(f"    State vector norm: {np.linalg.norm(state):.4f}")
    print(f"    Non-zero dims: {np.count_nonzero(state)}/24")
    print()

    # Show the raw encoding
    dim_labels = [
        "char_entropy", "alphabet_cov", "length_scale",
        "number_density", "operator_dens", "division_dens",
        "cert_vs_uncert", "question_int", "uncertainty",
        "multiplier(3)", "addend(1)", "divisor(0.5)",
        "verify_frontier", "years_unsolved", "no_counterex",
        "cycle_1", "cycle_4", "cycle_2",
        "log2(3)", "prob_ratio", "euler_gamma",
        "text_scale", "ent*math", "cert*uncert"
    ]
    slow_print("[2] Raw 24D encoding:", 0.03)
    for i, (label, val) in enumerate(zip(dim_labels, state)):
        if abs(val) > 0.001:
            print(f"    [{i:2d}] {label:16s} = {val:+.4f}")
    print()
    time.sleep(0.5)

    # Step 2: Entropy gating
    slow_print("[3] Entropy transponders gating...", 0.03)
    gated = entropy_transponders(state)
    time.sleep(0.3)
    ratio = np.linalg.norm(gated) / max(np.linalg.norm(state), 1e-12)
    print(f"    Input norm:     {np.linalg.norm(state):.4f}")
    print(f"    Gated norm:     {np.linalg.norm(gated):.4f}")
    print(f"    Pass-through:   {ratio:.2%}")

    # Show which dims got attenuated
    attenuation = np.abs(gated) / np.maximum(np.abs(state), 1e-12)
    most_gated = np.argsort(attenuation)[:5]
    most_passed = np.argsort(-attenuation)[:5]
    print(f"    Most attenuated: dims {list(most_gated)}")
    print(f"    Most preserved:  dims {list(most_passed)}")
    print()
    time.sleep(0.5)

    # Step 3: Octonion projection
    slow_print("[4] Projecting to octonion pair (A, B)...", 0.03)
    A = Octonion(gated[:8])
    B = Octonion(gated[8:16])
    context = gated[16:24]
    ctx_scale = 1.0 + np.linalg.norm(context) * 0.1
    B_scaled = B * ctx_scale

    print(f"    Octonion A: {A}")
    print(f"    Octonion B: {B}")
    print(f"    Context scale: {ctx_scale:.4f}")
    print(f"    |A| = {A.norm():.4f}  |B| = {B_scaled.norm():.4f}")
    print()
    time.sleep(0.5)

    # Step 4: Jordan-Shadow decomposition
    slow_print("[5] Jordan-Shadow decomposition (the thinking step)...", 0.03)
    time.sleep(0.5)
    decomp = octonion_shadow_decompose(A, B_scaled)

    jordan = decomp['jordan']
    commutator = decomp['commutator']
    associator = decomp['associator']
    product = decomp['product']

    print()
    slow_print("    JORDAN (symmetric intent — what the problem IS):", 0.02)
    print(f"      {jordan}")
    print(f"      |J| = {jordan.norm():.4f}")
    print()

    slow_print("    COMMUTATOR (anti-symmetric — directional tension):", 0.02)
    print(f"      {commutator}")
    print(f"      |C| = {commutator.norm():.4f}")
    print()

    slow_print("    ASSOCIATOR (J*C — non-linear chaos/personality):", 0.02)
    print(f"      {associator}")
    print(f"      |A| = {associator.norm():.4f}")
    print()

    # Verify orthogonality
    jc_dot = np.dot(jordan.vec, commutator.vec)
    print(f"    <J.vec, C.vec> = {jc_dot:.2e} (orthogonality check)")
    print()
    time.sleep(0.5)

    # Step 5: Full collapse
    slow_print("[6] Full AOI collapse — perception emerges...", 0.03)
    time.sleep(0.5)
    result = aoi_collapse(state)

    print()
    print("  +-----------------------------------------------------+")
    print("  |              VOODOO'S PERCEPTION                    |")
    print("  +-----------------------------------------------------+")
    print()

    chaos = result['chaos_level']
    nchaos = result['normalized_chaos']
    control = result['control_vec']
    intent = result['intent_magnitude']

    print_bar("Chaos Level", chaos, max_val=20.0)
    print_bar("Normalized Chaos", nchaos, max_val=10.0)
    print_bar("Intent Magnitude", intent, max_val=5.0)
    print_bar("Control Norm", result['control_norm'], max_val=10.0)
    print()

    print(f"  Control vector: [{control[0]:+.4f}, {control[1]:+.4f}, {control[2]:+.4f}]")
    print(f"    steer={control[0]:+.4f}  throttle={control[1]:+.4f}  brake={control[2]:+.4f}")
    print()

    # Confidence / exploration
    confidence = max(0, 1.0 - nchaos / 10.0)
    exploration = min(1.0, nchaos / 5.0)
    directness = control[0]

    print(f"  Confidence:  {confidence:.2%}")
    print(f"  Exploration: {exploration:.2%}")
    print(f"  Directness:  {'direct' if directness > 0 else 'circuitous'} ({directness:+.4f})")
    print()

    # Reasoning mode
    if nchaos < 1.0:
        mode = "DIRECT"
    elif nchaos < 2.0:
        mode = "OBSERVE"
    elif nchaos < 3.5:
        mode = "INVESTIGATE"
    elif nchaos < 5.0:
        mode = "EXPLORE"
    elif nchaos < 7.0:
        mode = "SYNTHESIZE"
    elif nchaos < 9.0:
        mode = "HEDGE"
    else:
        mode = "PAUSE"

    slow_print(f"  Reasoning Mode: {mode}", 0.04)
    print()

    # TTS prompt
    slow_print("  Voice prompt:", 0.02)
    print(f"    \"{result['text_prompt_base']}\"")
    print()

    # Step 6: Personality embedding analysis
    slow_print("[7] Personality embedding (8D associator):", 0.03)
    pe = result['personality_embedding']
    dominant = np.argmax(np.abs(pe))
    basis_labels = ['e0(real)', 'e1', 'e2', 'e3', 'e4', 'e5', 'e6', 'e7']
    for i, (label, val) in enumerate(zip(basis_labels, pe)):
        marker = " <<<" if i == dominant else ""
        print(f"    {label}: {val:+10.4f}{marker}")
    print(f"    Dominant axis: {basis_labels[dominant]} ({pe[dominant]:+.4f})")
    print()

    # Step 7: What does Voodoo "see"?
    slow_print("=" * 60, 0.005)
    slow_print("  VOODOO'S TAKE ON THE COLLATZ CONJECTURE", 0.03)
    slow_print("=" * 60, 0.005)
    print()

    if mode == "PAUSE":
        slow_print("  '...I need to sit with this one.'", 0.04)
    elif mode in ("HEDGE", "SYNTHESIZE"):
        slow_print("  'This problem has structure I can feel but can't pin down.'", 0.04)
        slow_print("  'The 3n+1 and n/2 create a tension between multiplication'", 0.04)
        slow_print("  'and division that never resolves cleanly in any algebra.'", 0.04)
    elif mode in ("EXPLORE", "INVESTIGATE"):
        slow_print("  'There is something in the binary structure. The way 3n+1'", 0.04)
        slow_print("  'always produces an even number... the odd steps PUSH UP'", 0.04)
        slow_print("  'by ~3x but immediately get pulled down by at least /2.'", 0.04)
        slow_print("  'The log2(3) irrationality is the core obstruction.'", 0.04)
    else:
        slow_print("  'The entropy signature says this problem is balanced --'", 0.04)
        slow_print("  'not chaotic enough to diverge, not ordered enough to prove.'", 0.04)

    print()

    # Additional insight from the decomposition
    j_to_c_ratio = jordan.norm() / max(commutator.norm(), 1e-12)
    slow_print(f"  Jordan/Commutator ratio: {j_to_c_ratio:.4f}", 0.02)
    if j_to_c_ratio > 2.0:
        slow_print("  -> Problem is more symmetric than directional (structure > dynamics)", 0.03)
    elif j_to_c_ratio < 0.5:
        slow_print("  -> Problem is more directional than symmetric (dynamics > structure)", 0.03)
    else:
        slow_print("  -> Problem has balanced structure and dynamics (that's why it's hard)", 0.03)

    a_to_j_ratio = associator.norm() / max(jordan.norm(), 1e-12)
    slow_print(f"  Associator/Jordan ratio: {a_to_j_ratio:.4f}", 0.02)
    if a_to_j_ratio > 1.0:
        slow_print("  -> Non-linear interaction DOMINATES -- the problem is fundamentally", 0.03)
        slow_print("    non-reducible. Can't separate structure from dynamics.", 0.03)
    else:
        slow_print("  -> Non-linear interaction contained -- there MAY be a decomposition", 0.03)
        slow_print("    that separates the hard part from the tractable part.", 0.03)

    print()
    slow_print("  Done.", 0.03)
    print()
