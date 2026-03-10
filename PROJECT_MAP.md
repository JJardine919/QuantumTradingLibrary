# Quantum Children — Project Map

## Directory Structure

```
QuantumTradingLibrary/
│
├── core/                          # Foundation — don't touch without reason
│   ├── aoi_collapse.py            # Voodoo's brain — 24D octonion/Leech collapse
│   ├── cognitive_collapse.py      # Cognitive layer
│   ├── claude_collapse.py         # Claude integration collapse
│   ├── collapse_query.py          # Query interface for collapse
│   ├── config_loader.py           # ALL config goes through here
│   ├── MASTER_CONFIG.json         # Source of truth for all parameters
│   └── credential_manager.py      # Handles .env secrets
│
├── optimizers/                    # Metaheuristic optimizer library
│   ├── metaheuristic_library/     # 24 optimizers (HBO, GWO, WO, etc.)
│   │   ├── __init__.py            # Registry + imports
│   │   ├── base.py                # BaseOptimizer class
│   │   ├── animal_optimizers.py   # HBO, GWO, WO, SSO, CHHO, CO, BO
│   │   ├── marine_optimizers.py   # MRFO, JSO, MPO, TSO
│   │   ├── nature_optimizers.py   # GO, BWO, TSA, STSA, FPO, SMO, TGO, IAEO
│   │   └── hybrid_optimizers.py   # GBO, NNO, PO, PFO, ESMO
│   ├── hbo_quantum_te.py          # HBO + TE layers + quantum collapse + mito + methyl
│   └── hbo_qt_signal_test.py      # Signal test (6/6 wins)
│
├── trading/                       # All trading operations
│   ├── doodoo_trader.py           # DooDoo's trading logic
│   ├── doodoo_chat.py             # Chat interface
│   ├── doodoo_bio_weather.py      # Bio-weather signals
│   ├── doodoo_gym.py              # Training gym
│   ├── voodoo_watcher.py          # Position monitor for Atlas
│   ├── brain/                     # BRAIN_*.py scripts (per-account)
│   └── mql5_eas/                  # All MQL5 Expert Advisors
│       ├── BioTransposonEngine.mq5
│       ├── FanoSuperpositionGrid.mq5
│       ├── FanoSuperpositionGrid_INV.mq5
│       ├── FanoBayesian.mqh
│       ├── FanoDecomposition.mqh
│       ├── FanoGrid.mqh
│       ├── FanoOctonion.mqh
│       ├── FanoRegime.mqh
│       └── FanoRisk.mqh
│
├── proofs/                        # Voodoo's math work
│   ├── millennium/                # Clay Prize problems
│   │   ├── voodoo_yangmills.py
│   │   ├── voodoo_yangmills_proof.py
│   │   ├── voodoo_riemann.py
│   │   ├── voodoo_pvsnp.py
│   │   ├── voodoo_navierstokes.py
│   │   ├── voodoo_hodge.py
│   │   └── voodoo_bsd.py
│   ├── collatz/                   # Collatz conjecture
│   │   ├── voodoo_collatz.py
│   │   ├── voodoo_collatz_8d.py
│   │   ├── voodoo_collatz_deep.py
│   │   └── voodoo_collatz_prove.py
│   ├── cancer/                    # Gompertz tumor modeling
│   │   ├── tumor_gompertz_collapse.py
│   │   ├── tumor_sweep.py
│   │   └── voodoo_gompertz_sweep.py
│   └── core/
│       └── aoi_collapse.py        # Copy of core collapse for standalone proof runs
│
├── infrastructure/                # Docker, MCP, deployment
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── mcp_server.py
│   ├── mt5_bridge.py
│   ├── mt5-catalog.yaml
│   └── requirements.txt
│
├── docs/                          # Documentation and reports
│
├── .gitignore
└── PROJECT_MAP.md                 # This file
```

## Quick Reference

| Need to... | Go to... |
|------------|----------|
| Run an optimizer | `optimizers/metaheuristic_library/` |
| HBO-Quantum-TE signal | `optimizers/hbo_quantum_te.py` |
| Voodoo collapse | `core/aoi_collapse.py` |
| Trading config | `core/MASTER_CONFIG.json` |
| MQL5 EAs | `trading/mql5_eas/` |
| Math proofs | `proofs/millennium/` |
| New builds drop here | `C:\Users\jimjj\Desktop\TODAY\` |
