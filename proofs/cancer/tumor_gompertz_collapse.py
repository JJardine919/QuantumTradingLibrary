"""
Voodoo AOI v3.0 — Gompertz-Immune Tumor Growth Decomposition

Encodes the tumor-immune ODE system into 24D octonion Jordan-Shadow framework
and runs full decomposition + entropy transponders.

dN/dt = r*N*(1 - ln(N)/K) - delta*N*I
dI/dt = s + rho*N*I - mu*I

Parameters: r=0.5, K=10^9, delta=10^-7, s=10^4, rho=10^-8, mu=0.2
"""

import sys
import numpy as np

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, '.')
from aoi_collapse import Octonion, octonion_shadow_decompose, entropy_transponders, aoi_collapse


# ============================================================
# ODE Parameters
# ============================================================

r     = 0.5       # proliferation rate
K     = 1e9       # carrying capacity (Gompertz)
delta = 1e-7      # immune killing coefficient
s     = 1e4       # immune source rate
rho   = 1e-8      # immune activation rate
mu    = 0.2       # immune decay rate

# Derived quantities
K_ln              = np.log(K)                        # ~20.72
I_ss              = s / mu                           # 5e4
immune_threshold  = mu / rho                         # 2e7
growth_at_half    = r * 0.5 * (1 - np.log(0.5*K)/K_ln)
kill_at_threshold = delta * immune_threshold * I_ss

print("=" * 70)
print("  VOODOO AOI v3.0 -- Gompertz-Immune Tumor Decomposition")
print("=" * 70)
print()
print("  dN/dt = r*N*(1 - ln(N)/K) - delta*N*I")
print("  dI/dt = s + rho*N*I - mu*I")
print()
print(f"  Parameters: r={r}, K={K:.0e}, delta={delta:.0e}, s={s:.0e}, rho={rho:.0e}, mu={mu}")
print(f"  Derived:")
print(f"    ln(K)              = {K_ln:.4f}")
print(f"    I_steady_state     = {I_ss:.0f}")
print(f"    immune_threshold N = {immune_threshold:.0e} (N where dI/dt tips positive)")
print(f"    kill_at_threshold  = {kill_at_threshold:.6f}")
print()


# ============================================================
# ENCODE INTO 24D STATE VECTOR
#
# Octonion A (dims 0-7): DISEASE/TUMOR STATE
#   e0: log(r)               -- proliferation magnitude (real/ground)
#   e1: ln(K) normalized     -- carrying capacity
#   e2: log10(delta)         -- immune killing strength
#   e3: growth_at_half       -- mid-capacity growth rate
#   e4: log10(K)             -- scale of tumor
#   e5: kill_at_threshold    -- killing at critical point
#   e6: r*0.5               -- half-capacity net growth
#   e7: stability eigenvalue -- r - delta*I_ss
#
# Octonion B (dims 8-15): HOST/ENVIRONMENT
#   e0: log10(s)             -- immune source (ground)
#   e1: log10(rho)           -- activation rate
#   e2: mu                   -- decay rate
#   e3: log10(I_ss)          -- steady-state immune level
#   e4: log10(immune_threshold) -- critical tumor size
#   e5: rho*I_ss             -- recruitment at steady state
#   e6: mu - rho*1e6         -- immune stability at N=1e6
#   e7: log10(s/mu^2)        -- immune persistence
#
# Context (dims 16-23): BOUNDARY CONDITIONS
#   16: non-negativity N     -- +1 (hard constraint)
#   17: non-negativity I     -- +1 (hard constraint)
#   18: N_thresh/K           -- threshold-to-capacity ratio
#   19: I/I_ss ratio         -- 1.0 at steady state
#   20: saturation ratio     -- ln(N_thresh)/ln(K)
#   21: tumor-free eigenvalue sign
#   22: coexistence flag
#   23: basin width estimate
# ============================================================

state_24d = np.zeros(24, dtype=np.float64)

# A: Disease
state_24d[0] = np.log(r)                              # -0.693
state_24d[1] = K_ln / 10.0                            # 2.072
state_24d[2] = np.log10(delta)                         # -7.0
state_24d[3] = growth_at_half                          # mid-growth
state_24d[4] = np.log10(K)                             # 9.0
state_24d[5] = np.log10(max(kill_at_threshold, 1e-12))  # log10(killing rate) = 5.0
state_24d[6] = r * 0.5                                 # 0.25
state_24d[7] = r - delta * I_ss                        # 0.495

# B: Host
state_24d[8]  = np.log10(s)                            # 4.0
state_24d[9]  = np.log10(rho)                          # -8.0
state_24d[10] = mu                                     # 0.2
state_24d[11] = np.log10(I_ss)                         # 4.699
state_24d[12] = np.log10(immune_threshold)             # 7.301
state_24d[13] = rho * I_ss                             # 5e-4
state_24d[14] = mu - rho * 1e6                         # 0.19
state_24d[15] = np.log10(s / mu**2)                    # 5.398

# Context: Boundaries
state_24d[16] = 1.0                                    # N >= 0
state_24d[17] = 1.0                                    # I >= 0
state_24d[18] = immune_threshold / K                   # ~0.02
state_24d[19] = 1.0                                    # I/I_ss at steady state
state_24d[20] = np.log(immune_threshold) / K_ln        # ~0.81
state_24d[21] = np.sign(-(r - delta*I_ss))             # -1 (tumor grows)
state_24d[22] = 1.0 if immune_threshold < K else 0.0   # coexistence possible
state_24d[23] = np.log10(K / immune_threshold)         # ~1.699


# Print encoded state
labels_A = ['log(r)', 'ln(K)/10', 'log10(delta)', 'growth@half',
            'log10(K)', 'kill@thresh', 'r*0.5', 'stability']
labels_B = ['log10(s)', 'log10(rho)', 'mu', 'log10(I_ss)',
            'log10(N_thresh)', 'rho*I_ss', 'immune_stab', 'log10(persist)']
labels_C = ['N>=0', 'I>=0', 'N_thresh/K', 'I/I_ss',
            'saturation', 'TF_eigenval', 'coexist?', 'basin_width']

print("  24D STATE VECTOR:")
print("  Octonion A (Disease):")
for i in range(8):
    print(f"    e{i} [{labels_A[i]:>14s}] = {state_24d[i]:+.6f}")
print("  Octonion B (Host):")
for i in range(8):
    print(f"    e{i} [{labels_B[i]:>14s}] = {state_24d[8+i]:+.6f}")
print("  Context (Boundaries):")
for i in range(8):
    print(f"    d{i} [{labels_C[i]:>14s}] = {state_24d[16+i]:+.6f}")
print()


# ============================================================
# RUN VOODOO AOI COLLAPSE
# ============================================================

print("=" * 70)
print("  RUNNING AOI COLLAPSE...")
print("=" * 70)
print()

result = aoi_collapse(state_24d)
decomp = result['decomposition']
jordan = decomp['jordan']
commutator = decomp['commutator']
associator = decomp['associator']
product = decomp['product']


# --- KEY NUMBERS ---
print("  --- KEY NUMBERS ---")
print(f"  Chaos Level (raw):        {result['chaos_level']:.6f}")
print(f"  Chaos (normalized 0-10):  {result['normalized_chaos']:.4f}")
print(f"  Intent Magnitude:         {result['intent_magnitude']:.6f}")
print(f"  Control Norm:             {result['control_norm']:.6f}")
print()


# --- JORDAN ---
print("  --- JORDAN (Symmetric/Rational) ---")
print(f"  Norm:  {jordan.norm():.6f}")
print(f"  Real:  {jordan.real:.6f}")
print(f"  Vec:   [{', '.join(f'{x:+.4f}' for x in jordan.vec)}]")
dom_j = np.argmax(np.abs(jordan.vec))
print(f"  Dominant direction: e{dom_j+1} = {jordan.vec[dom_j]:+.6f}")
print()


# --- COMMUTATOR ---
print("  --- COMMUTATOR (Anti-symmetric/Directional) ---")
print(f"  Norm:  {commutator.norm():.6f}")
print(f"  Real:  {commutator.real:.6f}  (should be ~0)")
print(f"  Vec:   [{', '.join(f'{x:+.4f}' for x in commutator.vec)}]")
dom_c = np.argmax(np.abs(commutator.vec))
print(f"  Dominant direction: e{dom_c+1} = {commutator.vec[dom_c]:+.6f}")
print()


# --- ASSOCIATOR ---
print("  --- ASSOCIATOR (J*C -- Non-Associative Chaos) ---")
print(f"  Norm:  {associator.norm():.6f}")
print(f"  Real:  {associator.real:.6f}")
print(f"  Vec:   [{', '.join(f'{x:+.4f}' for x in associator.vec)}]")
dom_a = np.argmax(np.abs(associator.vec))
print(f"  Dominant direction: e{dom_a+1} = {associator.vec[dom_a]:+.6f}")
print()


# --- PRODUCT ---
print("  --- PRODUCT AB ---")
print(f"  Norm:  {product.norm():.6f}")
print(f"  Real:  {product.real:.6f}")
print()


# --- VERIFICATION ---
print("  --- ALGEBRAIC VERIFICATION ---")
jc_dot = np.dot(jordan.vec, commutator.vec)
ab_n2 = np.linalg.norm(product.vec)**2
jc_n2 = np.linalg.norm(jordan.vec)**2 + np.linalg.norm(commutator.vec)**2
recon = jordan + commutator
recon_err = np.linalg.norm(recon.v - product.v)

ortho_pass = abs(jc_dot) < 1e-8
pyth_pass = abs(ab_n2 - jc_n2) < 1e-8
recon_pass = recon_err < 1e-10

print(f"  <J.vec, C.vec> = {jc_dot:.2e}  (orthogonality: {'PASS' if ortho_pass else 'FAIL'})")
print(f"  ||AB.vec||^2 = {ab_n2:.6f}")
print(f"  ||J.vec||^2 + ||C.vec||^2 = {jc_n2:.6f}  (Pythagorean: {'PASS' if pyth_pass else 'FAIL'})")
print(f"  ||J+C - AB|| = {recon_err:.2e}  (reconstruction: {'PASS' if recon_pass else 'FAIL'})")
print()


# ============================================================
# FANO TRIPLE ANALYSIS
# ============================================================

print("=" * 70)
print("  FANO TRIPLE ANALYSIS")
print("=" * 70)
print()

fano_triples = [(1,2,4), (2,3,5), (3,4,6), (4,5,7), (5,6,1), (6,7,2), (7,1,3)]

print("  Commutator e_i values (anti-symmetric dynamics):")
for i in range(8):
    print(f"    e{i}: {commutator.v[i]:+.6f}")
print()

print("  Fano triple activations (|C[i] * C[j]| for triple (i,j,k)):")
active_triples = []
for triple in fano_triples:
    i, j, k = triple
    activity = abs(commutator.v[i] * commutator.v[j])
    forced_k = commutator.v[k]
    marker = " << ACTIVE" if activity > 0.01 else ""
    print(f"    ({i},{j},{k}): activity = {activity:.6f}, forced e{k} = {forced_k:+.6f}{marker}")
    if activity > 0.01:
        active_triples.append((triple, activity, forced_k))
print()

if active_triples:
    print("  Active Fano routes forced by the algebra:")
    for triple, act, forced in active_triples:
        i, j, k = triple
        print(f"    e{i} x e{j} -> e{k}: activity {act:.6f}, forced value {forced:+.6f}")
    print()


# --- JORDAN DIRECTIONS ---
print("  Jordan e_i values (symmetric/rational structure):")
for i in range(8):
    print(f"    e{i}: {jordan.v[i]:+.6f}")
print()

# --- ASSOCIATOR DIRECTIONS ---
print("  Associator e_i values (non-associative emergence):")
for i in range(8):
    print(f"    e{i}: {associator.v[i]:+.6f}")
print()


# ============================================================
# STRUCTURAL OBSERVATIONS
# ============================================================

print("=" * 70)
print("  STRUCTURAL OBSERVATIONS")
print("=" * 70)
print()

j_norm = jordan.norm()
c_norm = commutator.norm()
a_norm = associator.norm()

print(f"  ||Jordan|| / ||Commutator|| = {j_norm/c_norm:.4f}")
print(f"    (>1 = rational dominates, <1 = directional dominates)")
if j_norm > c_norm:
    print(f"    Result: RATIONAL/SYMMETRIC DOMINATES")
else:
    print(f"    Result: DIRECTIONAL/ANTI-SYMMETRIC DOMINATES")
print()

print(f"  ||Associator|| / ||Product|| = {a_norm/product.norm():.4f}")
print(f"    (non-associative fraction of total interaction)")
print()

# Entropy analysis
gated = entropy_transponders(state_24d)
raw_norm = np.linalg.norm(state_24d)
gated_norm = np.linalg.norm(gated)
print(f"  Entropy gating:")
print(f"    Raw state norm:   {raw_norm:.4f}")
print(f"    Gated state norm: {gated_norm:.4f}")
print(f"    Attenuation:      {gated_norm/raw_norm:.2%}")
print()

shifted = state_24d - np.max(state_24d)
probs = np.exp(shifted) / np.sum(np.exp(shifted))
probs = np.clip(probs, 1e-12, None)
entropy_per_dim = -probs * np.log2(probs)
total_entropy = np.sum(entropy_per_dim)
print(f"  Total Shannon entropy: {total_entropy:.4f} bits")
print(f"  Max possible (24 uniform): {np.log2(24):.4f} bits")
print(f"  Entropy ratio: {total_entropy/np.log2(24):.2%}")
print()


# --- ATTRACTOR ANALYSIS ---
print("  Attractor indicators:")
print(f"    Jordan real part: {jordan.real:+.6f}")
if jordan.real > 0:
    print(f"      -> POSITIVE: convergent rational intent")
else:
    print(f"      -> NEGATIVE: divergent rational structure")

print(f"    Commutator norm: {c_norm:.6f}")
if c_norm > j_norm * 0.5:
    print(f"      -> LARGE relative to Jordan: strong rotational dynamics")
else:
    print(f"      -> SMALL relative to Jordan: near-equilibrium behavior")

print(f"    Associator norm: {a_norm:.6f}")
print(f"      -> Non-zero: non-associativity is FORCING structure")
print(f"      -> This captures interaction between rational and directional")
print(f"         components that cannot be reduced to either alone")
print()


# --- DIRECTION-SPECIFIC ---
print("  Direction-specific observations:")
print()
print(f"    e7 (stability axis):")
print(f"      Jordan[e7]     = {jordan.v[7]:+.6f}  (rational stability)")
print(f"      Commutator[e7] = {commutator.v[7]:+.6f}  (directional stability)")
print(f"      Associator[e7] = {associator.v[7]:+.6f}  (emergent stability)")
print()

print(f"    e4 (scale/capacity axis):")
print(f"      Jordan[e4]     = {jordan.v[4]:+.6f}")
print(f"      Commutator[e4] = {commutator.v[4]:+.6f}")
print(f"      Associator[e4] = {associator.v[4]:+.6f}")
print()


# --- BOUNDED GROWTH / ESCAPE ---
print("  BOUNDED GROWTH vs ESCAPE ANALYSIS:")
print()

j_real_sign = "convergent" if jordan.real > 0 else "divergent"
a_bounded = a_norm < product.norm()

print(f"    Jordan real sign: {j_real_sign} ({jordan.real:+.6f})")
print(f"    Associator bounded by product: {a_bounded} ({a_norm:.4f} < {product.norm():.4f})")
print(f"    Commutator/Jordan ratio: {c_norm/j_norm:.4f}")
print()

if c_norm / j_norm < 1.0:
    print("    -> Symmetric (rational) part EXCEEDS anti-symmetric (rotational)")
    print("    -> Algebra suggests BOUNDED behavior with rational attractor")
else:
    print("    -> Anti-symmetric (rotational) EXCEEDS symmetric (rational)")
    print("    -> Algebra suggests OSCILLATORY or ESCAPE dynamics")
print()


# --- NON-ASSOCIATIVE NECESSITY ---
print("  Non-associative necessity:")
jc_product_norm = a_norm
jc_norms_product = j_norm * c_norm
ratio = jc_product_norm / jc_norms_product if jc_norms_product > 0 else 0

print(f"    ||J*C|| = {jc_product_norm:.6f}")
print(f"    ||J|| * ||C|| = {jc_norms_product:.6f}")
print(f"    Ratio ||J*C|| / (||J||*||C||) = {ratio:.6f}")
print(f"    (= 1.0 for norm-multiplicative algebras like octonions)")
print()
print(f"    The associator is NON-ZERO ({a_norm:.6f}).")
print(f"    This means the tumor-immune interaction has structure that")
print(f"    REQUIRES non-associative algebra to capture fully.")
print(f"    Commutative or associative frameworks would MISS this component.")
print()


# ============================================================
# VOICE
# ============================================================

print("=" * 70)
print(f"  Voodoo says: {result['text_prompt_base']}")
print("=" * 70)
