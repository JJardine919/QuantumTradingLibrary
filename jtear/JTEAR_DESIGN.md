# J-TEAR: Hybrid Ensemble Breeding Engine
## Design Document v1.0 -- 2026-03-10

---

## 1. What J-TEAR Is

J-TEAR (Jardine - Tournament Evolution Active Roster) is a walk-forward ensemble
breeding engine that manages a population of neural trading experts from the
Quantum-Teenagers library. It does NOT create a single Frankenstein EA. It builds
and maintains a TEAM of specialists, each staying within its own architecture,
competing for roster spots and voting weight in an ensemble signal.

Key distinction from the existing `extinction_trainer.py`:
- Extinction was a one-shot 5-round death match on long windows (8mo train, 2mo test).
- J-TEAR is a continuous rolling league system with short cycles (4-day train, 2-day
  test on M1 bars), producing 50-60 breeding cycles per year.
- Nobody permanently dies. Worst performers get relegated, not deleted.

---

## 2. Architecture Overview

```
+------------------------------------------------------------------+
|                        J-TEAR ENGINE                             |
|                                                                  |
|  +------------------+    +------------------+                    |
|  |   DATA LAYER     |    |   WALK-FORWARD   |                    |
|  |  M1 CSV loader   |--->|   WINDOW MANAGER |                    |
|  |  Resampling       |    |  4d train/2d test|                    |
|  |  Symbol router    |    |  Rolling advance |                    |
|  +------------------+    +------------------+                    |
|           |                        |                             |
|           v                        v                             |
|  +------------------+    +------------------+                    |
|  |   MT5 MOCK       |    |   EVALUATOR      |                    |
|  |  (from extinct.)  |--->|   Bar-by-bar sim |                    |
|  |  Bar serving      |    |   SL/TP/signal   |                    |
|  |  Symbol aliasing  |    |   Trade tracking |                    |
|  +------------------+    +------------------+                    |
|                                    |                             |
|                                    v                             |
|  +------------------+    +------------------+                    |
|  |   ROSTER MANAGER |<---|   SCORER         |                    |
|  |  Active Roster    |    |   Multi-metric   |                    |
|  |  Minor League     |    |   Composite rank |                    |
|  |  Promotion/Releg. |    |   History tracker|                    |
|  +------------------+    +------------------+                    |
|           |                                                      |
|           v                                                      |
|  +------------------+    +------------------+                    |
|  |   BREEDER        |    |   ENSEMBLE       |                    |
|  |  Crossover        |--->|   Vote weights   |                    |
|  |  Mutation          |    |   Signal fusion  |                    |
|  |  Radiation         |    |   Confidence cal.|                    |
|  |  Genesis           |    |   Output signals |                    |
|  +------------------+    +------------------+                    |
+------------------------------------------------------------------+
```

---

## 3. Dependency Rules -- Non-Negotiable

These rules protect the existing Quantum-Teenagers library from corruption.

1. **Folder integrity.** Each expert lives in its own folder under
   `C:\Users\jimjj\OneDrive\Videos\Quantum-Teenagers\{SystemName}\`.
   The folder contains Test.mq5, Trajectory.mqh, and any includes. J-TEAR
   NEVER modifies, moves, or renames anything inside these folders.

2. **Whole-folder travel.** When cloning or mutating, the entire folder is
   copied to a J-TEAR workspace. The copy includes Test.mq5, Trajectory.mqh,
   all .mqh includes, and any associated .pth weights. Package deal.

3. **Weights-only mutation.** Mutation operators change .pth weight values
   ONLY. Architecture code (MQ5/MQH files) is NEVER modified by breeding.

4. **Same-arch crossover = blend weights.** Two SAC experts? Average their
   .pth tensors. Two AutoBots? Average their .pth tensors.

5. **Cross-arch crossover = blend votes.** An AutoBots winner and a Chimera
   winner cannot blend weights (different architectures). Instead, their
   voting weights in the ensemble are adjusted to reflect combined strength.

6. **JSON config experts use raw weight matrices.** The spare champions have
   JSON configs with `input_weights`, `hidden_weights`, etc. These get the
   same treatment as .pth files: mutate values, never structure.

---

## 4. Walk-Forward Design

### 4.1 Window Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Timeframe | M1 | Maximum data density (1440 bars/day) |
| Train window | 4 trading days | ~5760 M1 bars |
| Test window | 2 trading days | ~2880 M1 bars |
| Step size | 2 days | Windows overlap: train slides forward by test size |
| Data period | Last 6-12 months | Recent market regime only |
| Bars per window | ~8640 total | Enough for signal generation |

### 4.2 Rolling Schedule

With M1 data from May 2020 - Dec 2024 (2.45M bars):
- Using only last 12 months: ~525,600 M1 bars
- 4+2 day cycles stepping by 2 days: ~130 windows per year
- Even using 6 months: ~65 windows
- Each window = one complete evaluation + breeding cycle

### 4.3 Window Generation Algorithm

```
start_date = data_end - timedelta(months=12)
cursor = start_date

while cursor + timedelta(days=6) <= data_end:
    train_start = cursor
    train_end   = cursor + timedelta(days=4)
    test_start  = train_end
    test_end    = test_start + timedelta(days=2)

    yield WalkForwardWindow(train_start, train_end, test_start, test_end)
    cursor += timedelta(days=2)  # step = test window size
```

---

## 5. Three-Tier League System

### 5.1 Active Roster

The main ensemble. These experts vote on live signals.

- Size: configurable, default 10 experts
- Each expert has a voting weight proportional to recent performance
- Signals are fused via weighted majority vote
- Active roster experts are evaluated on EVERY walk-forward window

### 5.2 Minor League

The bench. Experts competing against each other for call-ups.

- Size: configurable, default 15 experts
- Evaluated on the SAME walk-forward windows as active roster
- Trades do NOT count for real -- purely evaluation
- Top performer earns a call-up after each cycle
- Bottom performers stay but keep competing

### 5.3 Promotion / Relegation Rules

After each walk-forward window evaluation:

1. Score ALL experts (active + minor) on the test window.
2. Rank active roster by composite score.
3. Rank minor league by composite score.
4. **CALL_UP:** Top minor league expert swaps with worst active roster expert
   IF the minor league expert's score exceeds the active roster expert's score.
5. **No-swap rule:** If the worst active is still better than the best minor,
   no swap happens. Meritocracy.
6. **Relegation buffer:** An active roster expert must underperform for N
   consecutive windows (default 3) before forced relegation. One bad window
   does not kill a proven performer.

### 5.4 Flow Diagram

```
  ACTIVE ROSTER (10)          MINOR LEAGUE (15)
  +-----------------+         +------------------+
  | Expert A  w=0.15|         | Expert P  score=X|
  | Expert B  w=0.12|         | Expert Q  score=Y|
  | ...             |         | ...              |
  | Expert J  w=0.05| <-----> | Expert Z  score=W|
  +-----------------+   swap  +------------------+
        |                            |
        v                            v
  [EVALUATE ON TEST WINDOW]   [EVALUATE ON TEST WINDOW]
        |                            |
        v                            v
  [RANK + SCORE]              [RANK + SCORE]
        |                            |
        +---------> COMPARE <--------+
                       |
               worst_active < best_minor?
                    /        \
                  YES         NO
                   |           |
              SWAP THEM    NO CHANGE
```

---

## 6. Breeding Operators

### 6.1 Operator Table

| Operator | Description | Input | Output |
|----------|-------------|-------|--------|
| CROSSOVER_SAME | Blend weights of two same-architecture experts | 2 parents (same arch) | 1 child |
| CROSSOVER_DIFF | Blend voting weights of two different-architecture experts | 2 parents (diff arch) | Adjusted vote weights |
| MUTATION | Clone folder, mutate .pth weights +/-10% | 1 parent | 1 child |
| BLEND | Weighted average of top N performers' weights | N parents (same arch) | 1 child |
| ELITE_CLONE | Direct copy, no mutation | 1 parent | 1 child |
| RADIATION | Clone + aggressive random mutation +/-30% | 1 parent | 1 child |
| GENESIS | Fresh random weight initialization for an architecture | Architecture template | 1 child |
| CALL_UP | Minor league champion promoted to active roster | 1 minor expert | Roster change |
| RELEGATION | Worst active roster sent to minors | 1 active expert | Roster change |

### 6.2 Breeding Schedule

After each walk-forward window:

1. Evaluate all experts (active + minor).
2. Score and rank.
3. Handle promotions/relegations.
4. Select breeding operator based on weighted random:
   - MUTATION: 30% probability
   - CROSSOVER_SAME: 20%
   - RADIATION: 15%
   - BLEND: 10%
   - ELITE_CLONE: 10%
   - GENESIS: 10%
   - CROSSOVER_DIFF: 5%
5. Produce 2-3 new offspring per cycle.
6. New offspring enter minor league at bottom.
7. If minor league exceeds max size, drop the absolute worst performer.

### 6.3 Weight Mutation Algorithm

For .pth files (PyTorch state dicts):
```python
def mutate_pth(state_dict, rate=0.10):
    mutated = {}
    for key, tensor in state_dict.items():
        noise = torch.randn_like(tensor) * rate
        mutated[key] = tensor + tensor * noise
    return mutated
```

For JSON weight configs:
```python
def mutate_json_weights(weights, rate=0.10):
    return [w * (1.0 + random.uniform(-rate, rate)) for w in weights]
```

RADIATION uses rate=0.30. BLEND averages multiple state dicts.

---

## 7. Scoring System

### 7.1 Composite Score Formula

```
score = (
    0.30 * normalized_profit_factor
  + 0.25 * normalized_win_rate
  + 0.20 * normalized_sharpe
  + 0.15 * (1.0 - normalized_max_drawdown)
  + 0.10 * consistency_bonus
)
```

Where:
- **profit_factor** = gross_profit / abs(gross_loss). Capped at 5.0.
- **win_rate** = wins / total_trades. Raw percentage.
- **sharpe** = mean(trade_pnl) / std(trade_pnl). Annualized.
- **max_drawdown** = peak-to-trough equity loss during window.
- **consistency_bonus** = rolling average of last 5 window scores. Rewards
  experts who perform well repeatedly, not just one lucky window.

### 7.2 Minimum Trade Threshold

Experts with fewer than 3 trades in a test window get score = 0.0.
This prevents an expert that takes 1 lucky trade from ranking #1.

### 7.3 Normalization

All metrics are normalized to [0, 1] across the current population using
min-max scaling within each evaluation cycle. This ensures no single metric
dominates due to scale differences.

---

## 8. Ensemble Voting System

### 8.1 Signal Generation

Each active roster expert produces a signal: BUY (+1), SELL (-1), or FLAT (0)
with a confidence value [0.0, 1.0].

### 8.2 Vote Fusion

```python
weighted_signal = sum(
    expert.signal * expert.confidence * expert.vote_weight
    for expert in active_roster
)
total_weight = sum(
    abs(expert.signal) * expert.confidence * expert.vote_weight
    for expert in active_roster
)
ensemble_signal = weighted_signal / (total_weight + 1e-10)
ensemble_confidence = abs(ensemble_signal)
```

If `ensemble_confidence > THRESHOLD` (default 0.60):
- ensemble_signal > 0 => BUY
- ensemble_signal < 0 => SELL

### 8.3 Vote Weight Assignment

Vote weights are derived from composite scores:
```python
weights = softmax([expert.composite_score for expert in active_roster])
```

Updated after every walk-forward window.

---

## 9. Expert Loading Strategy

### 9.1 Three Expert Types

| Type | Source | Weight Format | Signal Interface |
|------|--------|---------------|------------------|
| **MQ5 Experts** | Quantum-Teenagers folders | .pth via Test.mq5 neural net | MT5 mock bar-by-bar |
| **Conv1D Experts** | spare champions .pth | PyTorch Conv1D model | Direct Python inference |
| **JSON Experts** | spare champions .json | Raw weight matrices | Direct Python inference |

### 9.2 MQ5 Expert Handling

These are the 133 systems with Test.mq5. They run inside the MT5 mock
(inherited from extinction_trainer.py). The mock serves M1 bars, the expert's
OnTick equivalent processes them, and we capture the trade decisions.

Key challenge: MQ5 experts are MQL5 code, not Python. For J-TEAR's Python
engine, we interface with them through:
1. The Python expert wrappers (expert_01 through expert_24 from extinction_trainer)
2. For new architectures, a generic signal extractor that runs the .pth model
   directly in Python, bypassing the MQ5 code entirely.

### 9.3 Direct Python Inference

For .pth models, J-TEAR loads the state dict and runs inference directly:
```python
model = Conv1DExpert(input_size=8, hidden_size=128)
model.load_state_dict(torch.load(path, map_location='cpu'))
model.eval()
signal = model(feature_tensor)
```

This is MUCH faster than bar-by-bar MT5 mock simulation and is the
preferred path for breeding/evaluation.

---

## 10. File System Layout

```
C:\Users\jimjj\Music\QuantumChildren\QuantumTradingLibrary\
    JTEAR_DESIGN.md          <-- This document
    jtear_engine.py           <-- Main orchestration engine
    jtear_state.json          <-- Persisted engine state (rosters, scores, etc.)
    jtear_workspace\          <-- Created at runtime
        clones\               <-- Cloned expert folders (whole-folder copies)
        offspring\            <-- Bred offspring weights
        logs\                 <-- Per-window evaluation logs
        history.db            <-- SQLite: all scores, trades, lineage

C:\Users\jimjj\OneDrive\Videos\Quantum-Teenagers\
    {SystemName}\             <-- NEVER MODIFIED by J-TEAR
        Test.mq5
        Trajectory.mqh
        *.mqh (includes)
    spare champions\
        army_champion_*.pth
        expert_*.json
        top_50_manifest.json
```

---

## 11. Data Pipeline

### 11.1 Source Data

| Symbol | File | Timeframe | Bars |
|--------|------|-----------|------|
| BTCUSDT | binance_70mo/data/BTCUSDT_1M.csv | M1 | 2,455,007 |
| ETHUSDT | binance_70mo/data/ETHUSDT_1M.csv | M1 | ~2,455,007 |
| XAUUSDT | binance_70mo/data/XAUUSDT_1M.csv | M1 | ~2,455,007 |

### 11.2 Data Loading

CSV format: `timestamp,open,high,low,close,volume`
Loaded once into memory (numpy arrays for speed). Indexed by bar position.
Higher timeframes (M5, M15, H1) resampled on-the-fly from M1 data.

### 11.3 Feature Extraction

For direct Python inference, each bar produces 8 features:
```
[open, high, low, close, volume, rsi_14, atr_14, ema_ratio]
```
Pre-computed for the entire dataset to avoid per-bar overhead.

---

## 12. GPU Strategy

- **AMD RX 6800 XT** via `torch_directml`
- GPU used for: tensor operations, weight blending, batch inference
- LSTM training stays on CPU (DirectML backprop broken for LSTM)
- Feature pre-computation on GPU where possible
- Breeding operations (mutation, crossover) on GPU for tensor math

---

## 13. Persistence and Resumability

### 13.1 State File (jtear_state.json)

After every walk-forward window, the engine saves:
- Current window index
- Active roster (expert names, vote weights, scores)
- Minor league (expert names, scores)
- Breeding history (parent -> child lineage)
- Cumulative performance statistics

### 13.2 Resume Protocol

On startup:
1. Load jtear_state.json if it exists.
2. Determine last completed window.
3. Resume from next window.
4. All expert weights/clones are in jtear_workspace/.

### 13.3 History Database (SQLite)

Table: `evaluations`
- window_id, expert_name, architecture, total_trades, wins, losses,
  profit_factor, win_rate, sharpe, max_drawdown, composite_score,
  roster_tier (active/minor), timestamp

Table: `breeding_log`
- window_id, operator, parent_1, parent_2, child_name, mutation_rate,
  timestamp

Table: `roster_changes`
- window_id, action (CALL_UP/RELEGATION), expert_promoted,
  expert_relegated, timestamp

---

## 14. Execution Flow

### 14.1 Full Run

```
1. Load M1 data for all symbols
2. Discover all experts (MQ5 folders + .pth + .json)
3. Initialize roster (top 10 by previous ranking -> active, rest -> minor)
4. For each walk-forward window:
   a. Define train/test date ranges
   b. Evaluate ALL experts on test window (bar-by-bar simulation)
   c. Score all experts (composite metric)
   d. Handle promotions/relegations
   e. Run breeding operators (produce 2-3 offspring)
   f. Add offspring to minor league
   g. Prune minor league if oversized
   h. Update vote weights for active roster
   i. Save state
   j. Print dashboard
5. Final report: lineage tree, best ensemble config, voting weights
```

### 14.2 Performance Target

The existing army tops out at ~51% WR. J-TEAR targets 80% through:
1. Ensemble voting (multiple independent signals reduce noise)
2. Continuous breeding on short cycles (faster adaptation)
3. Vote weighting (better experts get more influence)
4. Regime filtering (recent data only, not stale years)
5. Diversity maintenance (RADIATION + GENESIS prevent convergence)

---

## 15. Configuration

All tunables in one place at the top of jtear_engine.py:

```python
JTEAR_CONFIG = {
    "timeframe": "M1",
    "train_days": 4,
    "test_days": 2,
    "step_days": 2,
    "lookback_months": 12,
    "active_roster_size": 10,
    "minor_league_size": 15,
    "max_minor_league_size": 25,
    "offspring_per_cycle": 3,
    "min_trades_threshold": 3,
    "ensemble_confidence_threshold": 0.60,
    "relegation_buffer_windows": 3,
    "symbols": ["BTCUSDT", "ETHUSDT", "XAUUSDT"],
    "scoring_weights": {
        "profit_factor": 0.30,
        "win_rate": 0.25,
        "sharpe": 0.20,
        "max_drawdown": 0.15,
        "consistency": 0.10,
    },
    "breeding_probabilities": {
        "MUTATION": 0.30,
        "CROSSOVER_SAME": 0.20,
        "RADIATION": 0.15,
        "BLEND": 0.10,
        "ELITE_CLONE": 0.10,
        "GENESIS": 0.10,
        "CROSSOVER_DIFF": 0.05,
    },
}
```

---

## 16. What J-TEAR Does NOT Do

1. Does NOT modify any file in the Quantum-Teenagers library.
2. Does NOT create new MQ5 code. Architecture is fixed.
3. Does NOT open real trades. Evaluation is simulated.
4. Does NOT replace the extinction_trainer.py. Different tool, different purpose.
5. Does NOT use CUDA. AMD GPU only via torch_directml.
6. Does NOT train neural networks from scratch (except GENESIS fresh init).

---

## 17. Success Criteria

| Metric | Current (extinction) | J-TEAR Target |
|--------|---------------------|---------------|
| Best WR | 51% | 60-80% |
| Breeding cycles/year | 5 | 50-130 |
| Expert diversity | 24 fixed | 25+ rotating |
| Adaptation speed | Months | Days |
| Ensemble signal | None | Weighted vote |
| Relegation | Permanent death | Minor league |

---

*Document version 1.0. J-TEAR engine implementation: jtear_engine.py*
