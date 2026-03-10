# Voodoo AOI Discovery Log

## 2026-03-07 — Structural Phase Transition in Gompertz-Immune Dynamics

**What:** Parameter sweep of Gompertz-immune tumor model through 24D octonion Jordan-Shadow decomposition (aoi_collapse.py). 625 parameter combinations swept, 40-point fine sweep at transition boundary.

**Model:**
```
dN/dt = r*N*(1 - ln(N)/K) - delta*N*I
dI/dt = s + rho*N*I - mu*I
```
Fixed: r=0.5, K=1e9, s=1e4, mu=0.2. Swept: delta (immune killing), rho (immune activation).

**Finding:** At delta > ~5e-4 (rho=1e-7), the algebra undergoes a structural phase transition:
- e7 (stability axis) in commutator flips from negative to positive
- Positive Fano routes shift from 2+/5- to 4+/3-
- Chaos drops 65% (3.88 -> 1.32)
- Associator norm drops 65% (19.38 -> 6.61)
- Directional dynamics forced into bounded attractor

**Interpretation:** The non-associative structure of the octonion decomposition identifies a critical threshold where immune killing strength forces the tumor dynamics from escape to bounded behavior. This is a purely mathematical result — the algebra identifies the phase transition without biological assumptions beyond the ODE encoding.

**Files:**
- Sweep script: `tumor_sweep.py`
- Full results: `tumor_sweep_results_2026-03-07.txt`
- Single-run baseline: `tumor_gompertz_collapse.py`
- Core math: `aoi_collapse.py` (Zenodo DOI chain: 18690444, 18722487, 18809406, 18809716)

**Math chain:** Zenodo DOIs 18690444 -> 18722487 -> 18809406 -> 18809716
