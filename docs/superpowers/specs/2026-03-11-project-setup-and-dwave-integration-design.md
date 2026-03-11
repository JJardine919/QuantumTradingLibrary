# Project Setup & D-Wave Integration Design

**Date:** 2026-03-11
**Status:** Approved
**Scope:** Full QuantumTradingLibrary project setup + D-Wave integration build plan

---

## 1. Git Foundation

- Commit all current QTL files on `master` as a snapshot (safety net)
- Add `.gitignore` for `__pycache__/`, `*.pyc`, `.env`, `*.pt`, `*.onnx`
- Create public GitHub repo: `QuantumTradingLibrary`
- Push master to origin

## 2. Directory Reorganization

On `feature/reorganize` branch, restructure QTL from flat layout to:

```
QuantumTradingLibrary/
├── CLAUDE.md
├── MASTER_CONFIG.json
├── config_loader.py
├── VOODOO_LOG.md
├── trading/          # BRAIN scripts, watchdogs
├── voodoo/           # aoi_collapse, voodoo_agent, collapse_query, doodoo_chat
├── dwave/            # transponder_gates, demo_local, D-Wave project CLAUDE.md
│   └── circuit_fault_diagnosis/
├── proofs/           # millennium, collatz, cancer
├── DEPLOY/           # MQ5 Expert Advisors
└── archive/          # old/experimental
```

All import paths updated after moves. Test that nothing breaks before merging.

## 3. Branch Workflow

- `master` = stable. Never push directly.
- All new work on feature branches: `feature/<name>`
- Claude Code handles git operations. Jim says "merge it" when ready.
- Voodoo does NOT push to git. She logs to `VOODOO_LOG.md`. Claude Code reads it.

## 4. D-Wave Integration Builds (Priority Order)

Each gets its own feature branch:

1. **`collapse_to_entropy_readings()`** — Bridge function mapping aoi_collapse output (chaos, control, intent) to 32 transponder entropy readings. Connects Voodoo perception to D-Wave BQM. **Priority 1.**

2. **Penalty tuning** — Adjust FAULT_GAP and LAYER_ACTIVE_BIAS for SimulatedAnnealingSampler compatibility. Currently only works with analytical solver.

3. **Energy gap tracking** — Confidence metrics on fault diagnosis results.

4. **QPU swap** — When D-Wave LaunchPad access is granted, swap SimulatedAnnealingSampler for DWaveSampler/LeapHybridSampler. One-line change.

## 5. Coordination

- `VOODOO_LOG.md` — Voodoo's activity log, read by Claude Code each session
- `@CLAUDE:` prefix in log = message for Claude Code
- `@JIM:` prefix in log = message for Jim
- Voodoo has autonomy but does not touch: `MASTER_CONFIG.json`, `BRAIN_*.py`, `.mq5` files, `.env` files

## Existing Assets (Already Built)

- `transponder_gates.py` — D-Wave BQM for 32-transponder engine, all 4 tests passing
- `demo_local.py` — Circuit fault diagnosis demo, all 4 tests passing
- `voodoo_agent.py` — Standalone local agent (Ollama + aoi_collapse), running
- `aoi_collapse.py` — 24D octonion Jordan-Shadow decomposition core
- D-Wave Ocean SDK 9.3.0 installed on py -3.12
