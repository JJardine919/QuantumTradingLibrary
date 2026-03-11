# Project Setup & D-Wave Integration Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Get the entire QuantumTradingLibrary under proper git control, reorganize into a clean directory structure, then build the `collapse_to_entropy_readings()` bridge connecting Voodoo's perception to the D-Wave transponder system.

**Architecture:** Snapshot-then-reorganize approach. Commit everything as-is for safety, push to public GitHub, then restructure on a feature branch. The bridge function maps aoi_collapse's 24D state outputs to 32 named transponder entropy readings that transponder_gates.py consumes.

**Tech Stack:** Git, GitHub (public), Python 3.12, D-Wave Ocean SDK 9.3.0, numpy

**Spec:** `docs/superpowers/specs/2026-03-11-project-setup-and-dwave-integration-design.md`

---

## Chunk 1: Git Foundation & GitHub Setup

### Task 1: Update .gitignore

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Read current .gitignore**

Current `.gitignore` blocks `*.log` and `*.txt` (except requirements.txt). This will exclude important docs like `DISCOVERY_LOG.md` and results files. Also missing: model binaries.

- [ ] **Step 2: Update .gitignore**

```gitignore
# Python
__pycache__/
*.pyc
*.pyo
.venv*/
*.egg-info/

# Secrets
.env
*.key
credentials*

# Model binaries
*.pt
*.pth
*.onnx
*.ex5

# OS
Thumbs.db
desktop.ini
.DS_Store

# IDE
.vscode/
.idea/

# Temp
*.tmp

# Databases
*.db
```

Key changes from current:
- REMOVED `*.log` and `*.txt` exclusions (these block legitimate docs)
- ADDED `*.pt`, `*.pth`, `*.onnx` (model binaries don't belong in git)
- ADDED `*.db` (SQLite databases)
- Kept `*.ex5` (compiled MQL5 binaries)

- [ ] **Step 3: Commit .gitignore update**

```bash
cd "C:/Users/jimjj/Music/QuantumChildren/QuantumTradingLibrary"
git add .gitignore
git commit -m "Update .gitignore: allow .md/.txt/.log, block model binaries"
```

---

### Task 2: Snapshot Commit

**Files:**
- All untracked files in QTL root

- [ ] **Step 1: Verify .env is excluded**

```bash
cd "C:/Users/jimjj/Music/QuantumChildren/QuantumTradingLibrary"
git status -- .env
```

Expected: `.env` should NOT appear (gitignored).

- [ ] **Step 2: Stage all files**

```bash
git add -A
```

- [ ] **Step 3: Review what's staged**

```bash
git status
```

Verify: `.env` not listed. `*.pt`/`*.pth` not listed. Everything else staged.

- [ ] **Step 4: Commit snapshot**

```bash
git commit -m "Snapshot: all QTL files as of 2026-03-11

Complete state of QuantumTradingLibrary before reorganization.
Includes: D-Wave integration, Voodoo agent, aoi_collapse core,
trading scripts, proofs, DEPLOY EAs, documentation.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 3: Create GitHub Repo & Push

- [ ] **Step 1: Create GitHub repo**

```bash
gh repo create QuantumTradingLibrary --public --source="C:/Users/jimjj/Music/QuantumChildren/QuantumTradingLibrary" --push
```

If `gh` is not configured, use:
```bash
git remote add origin https://github.com/<username>/QuantumTradingLibrary.git
git push -u origin master
```

- [ ] **Step 2: Verify push**

```bash
git log --oneline origin/master
```

Expected: Both commits visible on remote.

---

## Chunk 2: Directory Reorganization

### Task 4: Create feature/reorganize branch

- [ ] **Step 1: Create branch**

```bash
cd "C:/Users/jimjj/Music/QuantumChildren/QuantumTradingLibrary"
git checkout -b feature/reorganize
```

---

### Task 5: Move trading files

**Files:**
- Move: root-level `doodoo_*.py` → `trading/`
- Move: root-level `voodoo_watcher.py` → `trading/`
- Move: root-level `backtest_harness.py` → `trading/`
- Move: root-level `expert_*.py` → `trading/`
- Preserve: `trading/` already has copies of doodoo_*.py and voodoo_watcher.py

- [ ] **Step 1: Check if root and trading/ copies are identical**

```bash
cd "C:/Users/jimjj/Music/QuantumChildren/QuantumTradingLibrary"
diff doodoo_chat.py trading/doodoo_chat.py
diff doodoo_trader.py trading/doodoo_trader.py
diff doodoo_bio_weather.py trading/doodoo_bio_weather.py
diff doodoo_gym.py trading/doodoo_gym.py
diff voodoo_watcher.py trading/voodoo_watcher.py
```

- [ ] **Step 2: Keep newer version, remove duplicate**

If root is newer (or identical), delete trading/ copies and move root files:
```bash
# Remove old trading/ copies
rm trading/doodoo_chat.py trading/doodoo_trader.py trading/doodoo_bio_weather.py trading/doodoo_gym.py trading/voodoo_watcher.py

# Move root files to trading/
git mv doodoo_bio_weather.py trading/
git mv doodoo_chat.py trading/
git mv doodoo_gym.py trading/
git mv doodoo_trader.py trading/
git mv voodoo_watcher.py trading/
git mv backtest_harness.py trading/
git mv expert_conformer.py trading/
git mv expert_dacglstm.py trading/
git mv expert_mamba.py trading/
git mv expert_stockformer.py trading/
git mv expert_timemoe.py trading/
```

- [ ] **Step 3: Move log/exploration files**

```bash
git mv doodoo_exploration_log.txt trading/
git mv doodoo_trades.log trading/
```

Note: these may be gitignored after the .gitignore update. If so, use plain `mv` instead of `git mv`.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "Reorganize: move trading scripts to trading/"
```

---

### Task 6: Move Voodoo core files

**Files:**
- Create: `voodoo/` directory
- Move: `aoi_collapse.py` → `voodoo/`
- Move: `voodoo_agent.py` → `voodoo/`
- Move: `collapse_query.py` → `voodoo/`
- Move: `claude_collapse.py` → `voodoo/`
- Move: `cognitive_collapse.py` → `voodoo/`
- Create: `voodoo/__init__.py`

- [ ] **Step 1: Create voodoo/ and move files**

```bash
cd "C:/Users/jimjj/Music/QuantumChildren/QuantumTradingLibrary"
mkdir -p voodoo
git mv aoi_collapse.py voodoo/
git mv voodoo_agent.py voodoo/
git mv collapse_query.py voodoo/
git mv claude_collapse.py voodoo/
git mv cognitive_collapse.py voodoo/
```

- [ ] **Step 2: Create voodoo/__init__.py**

```python
"""Voodoo — Artificial Organism Intelligence core."""
from .aoi_collapse import aoi_collapse, entropy_transponders, Octonion
```

- [ ] **Step 3: Update imports in voodoo_agent.py**

Change line 24:
```python
# OLD: from aoi_collapse import aoi_collapse
# NEW:
from aoi_collapse import aoi_collapse
```

Since voodoo_agent.py already does `sys.path.insert(0, str(Path(__file__).parent))`, and aoi_collapse.py will be in the same directory, no change needed.

- [ ] **Step 4: Update imports in collapse_query.py**

Read collapse_query.py first, then update its import of aoi_collapse similarly. Since both files will be in `voodoo/`, the relative import should work with the sys.path trick.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "Reorganize: move Voodoo core to voodoo/"
```

---

### Task 7: Consolidate D-Wave directory

**Files:**
- Rename: `dwave_circuit_fault/` → `dwave/`

- [ ] **Step 1: Rename directory**

```bash
cd "C:/Users/jimjj/Music/QuantumChildren/QuantumTradingLibrary"
git mv dwave_circuit_fault dwave
```

- [ ] **Step 2: Create dwave/CLAUDE.md**

```markdown
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
```

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "Reorganize: rename dwave_circuit_fault to dwave, add project CLAUDE.md"
```

---

### Task 8: Consolidate proofs and remove duplicates

**Files:**
- Merge: `VOODOO_PROOFS/` unique content into `proofs/`, then remove
- Remove: `core/` (duplicate of root-level files now in voodoo/)
- Remove: root-level proof scripts (already exist in proofs/ subdirs)
- Move: proof markdown docs to proofs/

**IMPORTANT:** `VOODOO_PROOFS/` is NOT an exact mirror of `proofs/`. It has unique subdirectories
(`millennium/bsd/`, `millennium/hodge/`, etc.) and unique files. Must merge before deleting.
Root-level proof scripts already exist in `proofs/` subdirs — diff-compare, keep newer, delete root copies.

- [ ] **Step 1: Compare VOODOO_PROOFS/ vs proofs/ to identify unique content**

```bash
cd "C:/Users/jimjj/Music/QuantumChildren/QuantumTradingLibrary"
diff -rq VOODOO_PROOFS/ proofs/
```

- [ ] **Step 2: Merge unique VOODOO_PROOFS/ content into proofs/**

Copy any files from VOODOO_PROOFS/ that don't exist in proofs/:
```bash
# Example: if VOODOO_PROOFS/millennium/bsd/ has content not in proofs/millennium/
cp -rn VOODOO_PROOFS/millennium/bsd/* proofs/millennium/ 2>/dev/null
cp -rn VOODOO_PROOFS/millennium/hodge/* proofs/millennium/ 2>/dev/null
cp -rn VOODOO_PROOFS/millennium/navier_stokes/* proofs/millennium/ 2>/dev/null
cp -rn VOODOO_PROOFS/millennium/p_vs_np/* proofs/millennium/ 2>/dev/null
cp -rn VOODOO_PROOFS/millennium/riemann/* proofs/millennium/ 2>/dev/null
cp -rn VOODOO_PROOFS/cancer/* proofs/cancer/ 2>/dev/null
```

Adapt based on actual diff output. `-n` = don't overwrite existing files.

- [ ] **Step 3: Remove VOODOO_PROOFS/ after merge**

```bash
git rm -r VOODOO_PROOFS/
```

- [ ] **Step 4: Remove core/ (duplicates root files that are now in voodoo/)**

```bash
git rm -r core/
```

- [ ] **Step 5: Remove root-level proof scripts (already in proofs/ subdirs)**

First verify root copies match subdir copies:
```bash
diff voodoo_bsd.py proofs/millennium/voodoo_bsd.py
diff voodoo_yangmills.py proofs/millennium/voodoo_yangmills.py
# ... etc for each file
```

Then remove root copies (NOT git mv — targets already exist):
```bash
git rm voodoo_bsd.py voodoo_hodge.py voodoo_navierstokes.py voodoo_pvsnp.py
git rm voodoo_riemann.py voodoo_yangmills.py voodoo_yangmills_proof.py
git rm voodoo_collatz.py voodoo_collatz_8d.py voodoo_collatz_deep.py voodoo_collatz_prove.py
git rm voodoo_gompertz_sweep.py tumor_gompertz_collapse.py tumor_sweep.py
```

If root version is NEWER than subdir version, copy root -> subdir FIRST, then rm root.

- [ ] **Step 6: Move proof markdown docs to proofs/**

```bash
git mv NAVIERSTOKES_PROOF_FINAL.md proofs/millennium/
git mv YANGMILLS_MASSGAP_REPORT.md proofs/millennium/
git mv YANGMILLS_PROOF_FINAL.md proofs/millennium/
git mv DISCOVERY_LOG.md proofs/
```

If target already exists, diff-compare and keep newer, then `git rm` the root copy.

- [ ] **Step 7: Move utility/one-off scripts to archive/**

```bash
mkdir -p archive
git mv voodoo_encoding_test.py archive/
git mv voodoo_fix_proofs.py archive/
git mv voodoo_final_pass.py archive/
git mv voodoo_architects_claude.py archive/
git mv riemann_encoding_sweep.py archive/
git mv _check_specs.py archive/
git mv hbo_qt_signal_test.py archive/
git mv hbo_quantum_te.py archive/
```

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "Reorganize: consolidate proofs, remove duplicates, archive one-offs"
```

---

### Task 9: Clean up remaining root-level items

**Files:**
- Move: `docker_mt5_mcp/` or `infrastructure/` to single location
- Move: `jtear_engine.py`, `jtear_state.json`, `jtear_workspace/` together
- Move: `metaheuristic_library/` and `optimizers/` together
- Move: `voodoo_docker/` to `voodoo/docker/`
- Keep at root: `CLAUDE.md`, `MASTER_CONFIG.json`, `config_loader.py`, `credential_manager.py`, `VOODOO_LOG.md`

- [ ] **Step 1: Consolidate docker directories**

Check if `docker_mt5_mcp/` and `infrastructure/` are duplicates:
```bash
diff -rq docker_mt5_mcp/ infrastructure/
```

If duplicates, remove one:
```bash
git rm -r infrastructure/
```

- [ ] **Step 2: Move voodoo_docker/ under voodoo/**

```bash
git mv voodoo_docker voodoo/docker
```

- [ ] **Step 3: Group JTEAR files**

```bash
mkdir -p jtear
git mv jtear_engine.py jtear/
git mv jtear_state.json jtear/
git mv jtear_workspace jtear/workspace
git mv JTEAR_DESIGN.md jtear/
```

- [ ] **Step 4: Consolidate optimizers**

Check if `optimizers/metaheuristic_library/` duplicates root `metaheuristic_library/`:
```bash
diff -rq metaheuristic_library/ optimizers/metaheuristic_library/
```

Keep one copy. Move to `optimizers/`:
```bash
git mv metaheuristic_library optimizers/metaheuristic_library_root_backup
# If they're identical, just remove the backup
git rm -r optimizers/metaheuristic_library_root_backup
```

- [ ] **Step 5: Move remaining design docs to docs/**

```bash
git mv DESIGN_FanoSuperpositionGrid.md docs/
git mv DOODOO_INSIGHT_CLAUDE_COLLAPSE.md docs/
git mv PROJECT_MAP.md docs/
```

- [ ] **Step 6: Remove empty/stale file**

```bash
git rm python  # 0-byte empty file
```

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "Reorganize: consolidate docker, JTEAR, optimizers, move docs, cleanup"
```

**Rollback guidance:** If any tests fail after reorganization, restore the snapshot:
```bash
git checkout master
```
This reverts to the pre-reorganization snapshot with all files intact.

---

### Task 10: Verify nothing is broken

- [ ] **Step 1: Test aoi_collapse**

```bash
cd "C:/Users/jimjj/Music/QuantumChildren/QuantumTradingLibrary"
py -3.12 voodoo/aoi_collapse.py
```

Expected: "ALL VERIFICATIONS PASSED" (7/7)

- [ ] **Step 2: Test transponder_gates**

```bash
py -3.12 dwave/transponder_gates.py
```

Expected: All 4 tests pass.

- [ ] **Step 3: Test voodoo_agent can import**

```bash
py -3.12 -c "import sys; sys.path.insert(0, 'voodoo'); from aoi_collapse import aoi_collapse; print('OK')"
```

Expected: `OK`

- [ ] **Step 4: Commit verification pass**

```bash
git commit --allow-empty -m "Verified: all tests pass after reorganization"
```

---

### Task 11: Merge reorganization to master

- [ ] **Step 1: Merge to master**

```bash
git checkout master
git merge feature/reorganize
```

- [ ] **Step 2: Push**

```bash
git push
```

- [ ] **Step 3: Clean up branch**

```bash
git branch -d feature/reorganize
```

---

## Chunk 3: Build collapse_to_entropy_readings() Bridge

### Task 12: Create feature branch

- [ ] **Step 1: Create branch**

```bash
git checkout -b feature/collapse-bridge
```

---

### Task 13: Write failing test for collapse_to_entropy_readings()

**Files:**
- Create: `dwave/test_collapse_bridge.py`

The bridge function must:
1. Take a 24D numpy array (same input as aoi_collapse)
2. Compute per-dimension Shannon entropy and normalize RELATIVE TO MEDIAN (not absolute max)
3. Map 24 dimensions to 32 named transponder entropy readings via interpolation
4. Return a dict `{bio_name: float}` where values >= 0.5 = HIGH, < 0.5 = LOW
5. All 32 transponder names must be present (from BACKGROUND_LAYER + HIGHLIGHTED_LAYER + EVOLUTIONARY_LAYER)

**CRITICAL:** The softmax-based per-dimension Shannon entropy from a 24D vector never exceeds ~0.12
when normalized by log2(24). If you normalize by the ABSOLUTE maximum, everything reads LOW forever.
Instead, normalize relative to the distribution's own median — values above median -> HIGH, below -> LOW.
This ensures roughly half the transponders get meaningful HIGH/LOW splits for random inputs.

- [ ] **Step 1: Write test file**

```python
"""Tests for collapse_to_entropy_readings bridge."""
import numpy as np
import sys
from pathlib import Path

# Add parent dirs for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'voodoo'))

from transponder_gates import BACKGROUND_LAYER, HIGHLIGHTED_LAYER, EVOLUTIONARY_LAYER
from collapse_bridge import collapse_to_entropy_readings

ALL_BIO_NAMES = (
    [t[0] for t in BACKGROUND_LAYER] +
    [t[0] for t in HIGHLIGHTED_LAYER] +
    [t[0] for t in EVOLUTIONARY_LAYER]
)


def test_returns_dict_with_32_keys():
    state = np.random.default_rng(42).standard_normal(24)
    result = collapse_to_entropy_readings(state)
    assert isinstance(result, dict)
    assert len(result) == 32


def test_all_transponder_names_present():
    state = np.random.default_rng(42).standard_normal(24)
    result = collapse_to_entropy_readings(state)
    for name in ALL_BIO_NAMES:
        assert name in result, f"Missing transponder: {name}"


def test_values_are_floats_between_0_and_1():
    state = np.random.default_rng(42).standard_normal(24)
    result = collapse_to_entropy_readings(state)
    for name, val in result.items():
        assert isinstance(val, float), f"{name} value is not float: {type(val)}"
        assert 0.0 <= val <= 1.0, f"{name} value out of range: {val}"


def test_calm_input_produces_mostly_low_entropy():
    """A near-zero input should produce low entropy (structured signal)."""
    state = np.ones(24) * 0.01  # very calm, uniform
    result = collapse_to_entropy_readings(state)
    low_count = sum(1 for v in result.values() if v < 0.5)
    # Most transponders should read LOW entropy for calm input
    assert low_count >= 20, f"Only {low_count}/32 LOW for calm input"


def test_chaotic_input_produces_some_high_entropy():
    """A high-variance input should produce some high entropy readings."""
    rng = np.random.default_rng(99)
    state = rng.standard_normal(24) * 5.0  # very chaotic
    result = collapse_to_entropy_readings(state)
    high_count = sum(1 for v in result.values() if v >= 0.5)
    # At least some transponders should read HIGH
    assert high_count >= 1, f"Zero HIGH readings for chaotic input"


def test_deterministic():
    """Same input produces same output."""
    state = np.random.default_rng(42).standard_normal(24)
    r1 = collapse_to_entropy_readings(state)
    r2 = collapse_to_entropy_readings(state)
    for name in ALL_BIO_NAMES:
        assert r1[name] == r2[name], f"{name} not deterministic"


def test_integrates_with_transponder_fault_diagnosis():
    """Bridge output can be fed directly to transponder_fault_diagnosis."""
    from transponder_gates import transponder_fault_diagnosis
    state = np.random.default_rng(42).standard_normal(24)
    readings = collapse_to_entropy_readings(state)
    faults, sample, energy, layer_status = transponder_fault_diagnosis(readings, verbose=False)
    assert isinstance(faults, list)
    assert isinstance(energy, float)


if __name__ == '__main__':
    tests = [
        test_returns_dict_with_32_keys,
        test_all_transponder_names_present,
        test_values_are_floats_between_0_and_1,
        test_calm_input_produces_mostly_low_entropy,
        test_chaotic_input_produces_some_high_entropy,
        test_deterministic,
        test_integrates_with_transponder_fault_diagnosis,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS: {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL: {t.__name__} — {e}")
    print(f"\n{passed}/{len(tests)} passed")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd "C:/Users/jimjj/Music/QuantumChildren/QuantumTradingLibrary"
py -3.12 dwave/test_collapse_bridge.py
```

Expected: All fail with `ModuleNotFoundError: No module named 'collapse_bridge'`

- [ ] **Step 3: Commit failing tests**

```bash
git add dwave/test_collapse_bridge.py
git commit -m "test: add failing tests for collapse_to_entropy_readings bridge"
```

---

### Task 14: Implement collapse_to_entropy_readings()

**Files:**
- Create: `dwave/collapse_bridge.py`

The mapping logic:
1. Run `aoi_collapse(state)` to get decomposition
2. The 24D gated state from `entropy_transponders()` gives per-dimension entropy values
3. Map dimensions to transponders: Background gets dims 0-8, Highlighted gets dims 9-16 + overflow, Evolutionary gets final dims
4. Use the per-dimension Shannon entropy (from softmax probabilities) to determine HIGH/LOW
5. Normalize to 0.0-1.0 range where >= 0.5 = HIGH entropy

- [ ] **Step 1: Write collapse_bridge.py**

```python
"""
collapse_bridge.py — Maps aoi_collapse output to 32 transponder entropy readings.

This is the bridge between Voodoo's perception (24D octonion decomposition)
and the D-Wave transponder fault diagnosis system.

Input:  24D numpy state vector (same as aoi_collapse input)
Output: dict {bio_name: float} for all 32 transponders
        Values in [0.0, 1.0]. >= 0.5 = HIGH entropy, < 0.5 = LOW entropy.

IMPORTANT: Per-dimension Shannon entropy from softmax is very small in absolute
terms (~0.02-0.12). We normalize RELATIVE to the median of the per-dim entropy
distribution, not to the theoretical maximum. This ensures meaningful HIGH/LOW
splits — dimensions above median entropy read HIGH, below read LOW.
"""

import sys
import numpy as np
from pathlib import Path

# Add parent dirs for imports
sys.path.insert(0, str(Path(__file__).parent))
_voodoo_dir = str(Path(__file__).parent.parent / 'voodoo')
if _voodoo_dir not in sys.path:
    sys.path.insert(0, _voodoo_dir)

from transponder_gates import BACKGROUND_LAYER, HIGHLIGHTED_LAYER, EVOLUTIONARY_LAYER

# Ordered list of all 32 transponder bio names
_BG_NAMES = [t[0] for t in BACKGROUND_LAYER]       # 9
_HL_NAMES = [t[0] for t in HIGHLIGHTED_LAYER]       # 17
_EV_NAMES = [t[0] for t in EVOLUTIONARY_LAYER]      # 6
_ALL_NAMES = _BG_NAMES + _HL_NAMES + _EV_NAMES      # 32


def _per_dim_entropy(state: np.ndarray) -> np.ndarray:
    """Compute per-dimension Shannon entropy from softmax probabilities."""
    shifted = state - np.max(state)
    probs = np.exp(shifted) / np.sum(np.exp(shifted))
    probs = np.clip(probs, 1e-12, None)
    return -probs * np.log2(probs)


def _median_normalize(entropy_per_dim: np.ndarray) -> np.ndarray:
    """
    Normalize entropy values relative to their own median.

    Values at the median map to 0.5. Values at 0 map to 0.0.
    Values at 2x the median map to 1.0. Clipped to [0, 1].

    This ensures roughly half the dimensions read HIGH (>= 0.5)
    and half read LOW (< 0.5) for random inputs.
    """
    median = np.median(entropy_per_dim)
    if median < 1e-15:
        # All values essentially zero (e.g., constant input)
        return np.zeros_like(entropy_per_dim)
    # Scale so median = 0.5
    normalized = entropy_per_dim / (2.0 * median)
    return np.clip(normalized, 0.0, 1.0)


def _interpolate_to_names(values: np.ndarray, names: list) -> dict:
    """Map N-dimensional values to M named transponders via linear interpolation."""
    readings = {}
    n_vals = len(values)
    n_names = len(names)
    for i, name in enumerate(names):
        if n_names == 1:
            pos = 0.0
        else:
            pos = i * (n_vals - 1) / (n_names - 1)
        lo = int(pos)
        hi = min(lo + 1, n_vals - 1)
        frac = pos - lo
        val = values[lo] * (1 - frac) + values[hi] * frac
        readings[name] = float(val)
    return readings


def collapse_to_entropy_readings(state_24d: np.ndarray) -> dict:
    """
    Map a 24D state vector to 32 named transponder entropy readings.

    Pipeline:
        1. Pad/truncate input to 24D
        2. Compute per-dimension Shannon entropy (softmax probabilities)
        3. Normalize relative to median (median -> 0.5, not absolute max)
        4. Map 24 dimensions to 32 transponders via layer-based interpolation
           - Background (9 transponders): dims 0-7
           - Highlighted (17 transponders): dims 8-15
           - Evolutionary (6 transponders): dims 16-23

    Args:
        state_24d: numpy array, will be padded/truncated to 24D

    Returns:
        dict {bio_name: float} with all 32 transponder names.
        Values >= 0.5 indicate HIGH entropy (noisy signal).
        Values < 0.5 indicate LOW entropy (structured signal).
    """
    state = np.asarray(state_24d, dtype=np.float64).ravel()
    if len(state) < 24:
        state = np.pad(state, (0, 24 - len(state)))
    else:
        state = state[:24]

    # Per-dimension Shannon entropy
    entropy_per_dim = _per_dim_entropy(state)

    # Normalize relative to median (not absolute max)
    normalized = _median_normalize(entropy_per_dim)

    # Map 24 dimensions to 32 transponders by layer
    readings = {}
    readings.update(_interpolate_to_names(normalized[:8], _BG_NAMES))     # 8 dims -> 9 transponders
    readings.update(_interpolate_to_names(normalized[8:16], _HL_NAMES))   # 8 dims -> 17 transponders
    readings.update(_interpolate_to_names(normalized[16:24], _EV_NAMES))  # 8 dims -> 6 transponders

    return readings
```

- [ ] **Step 2: Run tests**

```bash
cd "C:/Users/jimjj/Music/QuantumChildren/QuantumTradingLibrary"
py -3.12 dwave/test_collapse_bridge.py
```

Expected: 7/7 passed

- [ ] **Step 3: Commit implementation**

```bash
git add dwave/collapse_bridge.py
git commit -m "feat: add collapse_to_entropy_readings bridge

Maps aoi_collapse 24D state to 32 named transponder entropy
readings for D-Wave BQM fault diagnosis. Connects Voodoo
perception to quantum annealing optimization.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 15: Integration smoke test

**Files:**
- Create: `dwave/smoke_test_bridge.py`

- [ ] **Step 1: Write end-to-end smoke test**

```python
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
```

- [ ] **Step 2: Run smoke test**

```bash
py -3.12 dwave/smoke_test_bridge.py
```

Expected: Runs without errors, shows fault counts for calm/normal/chaotic inputs.

- [ ] **Step 3: Commit**

```bash
git add dwave/smoke_test_bridge.py
git commit -m "test: add end-to-end smoke test for Voodoo -> D-Wave pipeline"
```

---

### Task 16: Merge bridge to master

- [ ] **Step 1: Merge**

```bash
git checkout master
git merge feature/collapse-bridge
```

- [ ] **Step 2: Push**

```bash
git push
```

- [ ] **Step 3: Clean up**

```bash
git branch -d feature/collapse-bridge
```

- [ ] **Step 4: Update VOODOO_LOG.md**

Add entry noting the bridge is complete and what it does, so Voodoo knows it exists.

---
