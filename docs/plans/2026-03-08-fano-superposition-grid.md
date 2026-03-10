# Fano Superposition Grid EA - Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a pure-math grid EA that uses Fano plane octonion superposition for regime detection, with fixed $1 SL / $3 TP scaling that produces identical growth curves across all prop firm accounts.

**Architecture:** Single MQL5 EA with 6 include files. No Python dependency. Octonion math embedded directly in MQL5. Fano plane 7-point structure maps to 7 Fibonacci lookback periods. Jordan-Shadow decomposition separates consensus from conflict. Bayesian pattern association provides 30% weight tie-breaking.

**Tech Stack:** MQL5 only. MT5 Strategy Tester for validation.

**Design Spec:** `C:\Users\jimjj\Music\QuantumChildren\QuantumTradingLibrary\DESIGN_FanoSuperpositionGrid.md`

**Target Paths:**
- Include files: `C:\Users\jimjj\AppData\Roaming\MetaQuotes\Terminal\F6E5FFA163BE6F3F89ECBCA1BA487B55\MQL5\Include\Fano\`
- Main EA: `C:\Users\jimjj\AppData\Roaming\MetaQuotes\Terminal\F6E5FFA163BE6F3F89ECBCA1BA487B55\MQL5\Experts\FanoSuperpositionGrid.mq5`
- Repo copy: `C:\Users\jimjj\Music\QuantumChildren\QuantumTradingLibrary\DEPLOY\`
- MetaEditor: `C:\Program Files\Atlas Funded MT5 Terminal\MetaEditor64.exe`

---

### Task 1: Create FanoOctonion.mqh - Cayley Table + Octonion Class

**Files:**
- Create: `...Include\Fano\FanoOctonion.mqh`

**Step 1: Create the Fano directory**

```bash
mkdir -p "C:/Users/jimjj/AppData/Roaming/MetaQuotes/Terminal/F6E5FFA163BE6F3F89ECBCA1BA487B55/MQL5/Include/Fano"
```

**Step 2: Write FanoOctonion.mqh**

Must contain:
- `CAYLEY_SIGN[8][8]` — sign matrix from design spec Appendix A
- `CAYLEY_IDX[8][8]` — result index matrix from design spec Appendix A
- `struct Octonion` with:
  - `double v[8]` — components e0..e7
  - `void Set(double e0,e1,e2,e3,e4,e5,e6,e7)`
  - `void SetFromArray(const double &arr[], int offset=0)`
  - `void Zero()` — set all to 0
  - `void Unit()` — set e0=1, rest=0
  - `double Norm()` — sqrt(sum of squares)
  - `double Real()` — v[0]
  - `void Conjugate(Octonion &result)` — negate v[1..7]
  - `void Add(const Octonion &b, Octonion &result)` — component-wise add
  - `void Sub(const Octonion &b, Octonion &result)` — component-wise subtract
  - `void Scale(double s, Octonion &result)` — multiply all by scalar
  - `void Multiply(const Octonion &b, Octonion &result)` — CORE: use Cayley table, 64 multiply-adds

The Multiply function is the heart. For each result component k:
```
result.v[k] = 0
for i=0..7:
  for j=0..7:
    if CAYLEY_IDX[i][j] == k:
      result.v[k] += CAYLEY_SIGN[i][j] * this.v[i] * b.v[j]
```

Optimized: precompute which (i,j) pairs produce each k to avoid the inner branch.

**Step 3: Write a minimal test EA to verify octonion multiplication**

Create `FanoOctonion_Test.mq5` that:
- Creates e1, e2 as unit octonions
- Multiplies e1*e2, checks result = e4 (from Cayley table)
- Multiplies e2*e1, checks result = -e4 (non-commutativity)
- Checks e_i^2 = -1 for i=1..7
- Prints PASS/FAIL

**Step 4: Compile test EA**

```bash
"/c/Program Files/Atlas Funded MT5 Terminal/MetaEditor64.exe" /compile:"C:\Users\jimjj\AppData\Roaming\MetaQuotes\Terminal\F6E5FFA163BE6F3F89ECBCA1BA487B55\MQL5\Experts\FanoOctonion_Test.mq5" /log
```
Expected: 0 errors

---

### Task 2: Create FanoDecomposition.mqh - Jordan-Shadow + Entropy

**Files:**
- Create: `...Include\Fano\FanoDecomposition.mqh`
- Depends on: Task 1 (FanoOctonion.mqh)

**Step 1: Write FanoDecomposition.mqh**

Must contain:
- `struct FanoDecomp` with:
  - `Octonion A` — signal octonion
  - `Octonion B` — momentum octonion
  - `Octonion AB` — A*B product
  - `Octonion BA` — B*A product
  - `Octonion Jordan` — (AB+BA)/2 — consensus
  - `Octonion Commutator` — (AB-BA)/2 — conflict
  - `Octonion Associator` — Jordan * Commutator — chaos
  - `double jordan_strength` — |mean(Jordan.v[1..7])|
  - `double commutator_strength` — |mean(Commutator.v[1..7])|
  - `double chaos_level` — Associator.Norm()
  - `int jordan_direction` — sign(mean(Jordan.v[1..7])): +1=BUY, -1=SELL, 0=FLAT
  - `void Compute(const double &signals[], const double &dsignals[])` — takes 7 signals + 7 deltas, builds A and B, computes everything

The `Compute` function:
```
A.v[0] = 1.0; A.v[1..7] = signals[0..6]
B.v[0] = 1.0; B.v[1..7] = dsignals[0..6]
A.Multiply(B, AB)
B.Multiply(A, BA)
AB.Add(BA, temp); temp.Scale(0.5, Jordan)
AB.Sub(BA, temp); temp.Scale(0.5, Commutator)
Jordan.Multiply(Commutator, Associator)
// Compute strengths
```

**Step 2: Compile test** — add decomposition test to FanoOctonion_Test.mq5 that feeds known signals and prints Jordan/Commutator values.

---

### Task 3: Create FanoRegime.mqh - Triple Coherence + Regime Detection

**Files:**
- Create: `...Include\Fano\FanoRegime.mqh`
- Depends on: Task 2

**Step 1: Write FanoRegime.mqh**

Must contain:
- `FANO_TRIPLES[7][3]` — the 7 Fano triples: {0,1,3},{1,2,4},{2,3,5},{3,4,6},{4,5,0},{5,6,1},{6,0,2} (0-indexed)
- `FANO_LOOKBACKS[7]` — {5,8,13,21,34,55,89}
- `struct FanoRegime` with:
  - `double signals[7]` — current tanh(SMA-normalized) signals
  - `double prev_signals[7]` — previous bar's signals
  - `double dsignals[7]` — signals - prev_signals
  - `double coherence[7]` — triple coherence values
  - `int active_triple` — index of strongest triple
  - `double active_coherence` — its coherence value
  - `int triple_direction` — sign of active coherence
  - `double regime_confidence` — combined Jordan + triple confidence
  - `int final_direction` — +1=BUY, -1=SELL, 0=FLAT
  - `FanoDecomp decomp` — the octonion decomposition
  - `int sma_handles[7]` — indicator handles for 7 SMAs
  - `int atr_handle` — ATR(14) handle
  - `bool Init(string symbol, ENUM_TIMEFRAMES tf)` — create SMA + ATR handles
  - `void Deinit()` — release handles
  - `void Update(double confidence_threshold, bool inverted_mode)` — full signal pipeline

The `Update` function:
```
1. CopyBuffer each SMA handle to get sma_values[7]
2. CopyBuffer ATR handle to get atr_value
3. For i=0..6: signals[i] = tanh((Close[0] - sma_values[i]) / atr_value)
4. dsignals[i] = signals[i] - prev_signals[i]
5. decomp.Compute(signals, dsignals)
6. For k=0..6: coherence[k] = signals[FANO_TRIPLES[k][0]] * signals[FANO_TRIPLES[k][1]] * signals[FANO_TRIPLES[k][2]]
7. Find active_triple = argmax |coherence[k]|
8. triple_direction = sign(coherence[active_triple])
9. agreement = (decomp.jordan_direction == triple_direction) ? 1.0 : -1.0
10. regime_confidence = decomp.jordan_strength / (decomp.jordan_strength + decomp.commutator_strength + 1e-10) * fabs(coherence[active_triple]) * agreement
11. If inverted_mode: regime_confidence = -regime_confidence
12. If regime_confidence >= confidence_threshold: final_direction = triple_direction * (inverted_mode ? -1 : 1)
    Elif regime_confidence <= -confidence_threshold: final_direction = -triple_direction * (inverted_mode ? -1 : 1)
    Else: final_direction = 0
13. Copy signals to prev_signals
```

**Step 2: Compile test**

---

### Task 4: Create FanoBayesian.mqh - Pattern Association

**Files:**
- Create: `...Include\Fano\FanoBayesian.mqh`
- Depends on: Task 3

**Step 1: Write FanoBayesian.mqh**

Must contain:
- `PATTERN_LENGTH = 5` (2^5=32 possible patterns)
- `struct PatternCounter` with:
  - `int count_up[32]` — per-pattern UP counts
  - `int count_down[32]` — per-pattern DOWN counts
  - `int total_samples` — total observations
- `struct FanoBayesian` with:
  - `PatternCounter counters[7]` — one per Fano lookback
  - `void Init()` — zero all counters
  - `int EncodePattern(const double &close[], int start, int len)` — encode 5 bars as 0-31 binary index
  - `void UpdateCounters(const double &close[], int bars_available)` — for each lookback, encode pattern, record outcome
  - `double GetVote(int lookback_idx, const double &close[])` — P(UP|pattern) - 0.5 with Laplace smoothing
  - `double GetTripleVote(int triple_idx, const double &close[])` — average of 3 votes from active triple's lookbacks
  - `bool SaveToFile(string filename)` — persist counters
  - `bool LoadFromFile(string filename)` — restore counters

EncodePattern:
```
pattern = 0
for i=0..4:
  if close[start+i] > close[start+i+1]:
    pattern |= (1 << i)
return pattern
```

GetVote with Laplace:
```
p = EncodePattern(close, 0, PATTERN_LENGTH)
up = counters[idx].count_up[p]
down = counters[idx].count_down[p]
total = up + down
bayes_up = (up + 1.0) / (total + 2.0)
return bayes_up - 0.5
```

**Step 2: Compile test**

---

### Task 5: Create FanoGrid.mqh - Grid Position Management

**Files:**
- Create: `...Include\Fano\FanoGrid.mqh`
- Depends on: Task 3 (for chaos_level)

**Step 1: Write FanoGrid.mqh**

Must contain:
- `struct GridPosition` with:
  - `ulong ticket`
  - `double entry_price`
  - `double hidden_sl`
  - `double hidden_tp`
  - `double partial_tp` — DYNTP% level
  - `bool partial_closed` — whether 50% already taken
  - `int level` — grid level (0 = first entry)
  - `int direction` — +1=BUY, -1=SELL
  - `datetime open_time`
- `struct FanoGrid` with:
  - `GridPosition positions[]`
  - `int count`
  - `int magic`
  - `string symbol`
  - `double base_lot` — computed from $1 SL formula
  - `double sl_distance` — ATR * 0.0438
  - `double tp_distance` — 3x SL distance in dollar terms
  - `double partial_tp_distance` — 50% of TP
  - `double grid_spacing` — ATR * ROLLSLMULT * chaos multiplier
  - `double breakeven_trigger` — SL distance * ROLLSLMULT
  - `void Init(string sym, int mag)` — set symbol and magic
  - `void ComputeLotAndDistances(double atr, double chaos_level, double sl_dollars, double tp_dollars, double dyntp_pct, double atr_mult, double rollsl_mult)` — the core scaling formula
  - `bool CanOpenNew(int max_positions, int direction)` — check grid conditions
  - `bool OpenPosition(int direction, CTrade &trade)` — open at base_lot with hidden SL/TP
  - `void ManagePositions(CTrade &trade)` — check hidden SL/TP/partial on every tick
  - `void SyncWithBroker()` — rebuild grid[] from PositionsTotal() on init
  - `double TotalPnL()` — sum of floating P&L
  - `double LastEntryPrice()` — price of most recent grid position
  - `bool PriceReachedNextLevel(double current_price, int direction)` — check if price moved grid_spacing against us

`ComputeLotAndDistances`:
```
tick_value = SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_VALUE)
tick_size = SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_SIZE)
sl_price = atr * atr_mult
sl_ticks = sl_price / tick_size
base_lot = sl_dollars / (sl_ticks * tick_value)
// Normalize to broker step
lot_step = SymbolInfoDouble(symbol, SYMBOL_VOLUME_STEP)
min_lot = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN)
max_lot_broker = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MAX)
base_lot = MathFloor(base_lot / lot_step) * lot_step
base_lot = MathMax(base_lot, min_lot)
base_lot = MathMin(base_lot, max_lot_broker)
// TP distance
tp_distance = tp_dollars / (base_lot * tick_value / tick_size)
partial_tp_distance = tp_distance * (dyntp_pct / 100.0)
// Grid spacing
chaos_norm = MathMin(chaos_level / 5.0, 10.0)
spacing_mult = 1.0 + 0.2 * chaos_norm
grid_spacing = atr * rollsl_mult * spacing_mult
// Breakeven trigger
breakeven_trigger = sl_price * rollsl_mult
```

`ManagePositions` (called every tick):
```
for each position:
  current_pnl = position profit
  // Check hidden SL
  if direction == +1 && Bid <= entry - sl_distance: close
  if direction == -1 && Ask >= entry + sl_distance: close
  // Check partial TP (50%)
  if !partial_closed:
    if direction == +1 && Bid >= entry + partial_tp_distance: close 50%, move SL to entry
    if direction == -1 && Ask <= entry - partial_tp_distance: close 50%, move SL to entry
  // Check full TP
  if direction == +1 && Bid >= entry + tp_distance: close remaining
  if direction == -1 && Ask <= entry - tp_distance: close remaining
  // Breakeven management (after ROLLSLMULT move)
  if direction == +1 && Bid >= entry + breakeven_trigger: move hidden SL to entry
  if direction == -1 && Ask <= entry - breakeven_trigger: move hidden SL to entry
```

**Step 2: Compile test**

---

### Task 6: Create FanoRisk.mqh - Drawdown Protection

**Files:**
- Create: `...Include\Fano\FanoRisk.mqh`

**Step 1: Write FanoRisk.mqh**

Must contain:
- `struct FanoRisk` with:
  - `double start_balance`
  - `double high_water_mark`
  - `double daily_start_balance`
  - `datetime last_day_reset`
  - `bool blocked`
  - `string block_reason`
  - `void Init()` — capture starting balance, set HWM
  - `void CheckDailyReset()` — if new day, reset daily_start_balance
  - `bool IsSafe(double daily_dd_limit, double max_dd_limit)` — check both limits
  - `double DailyDD()` — current daily drawdown %
  - `double TotalDD()` — current total drawdown from HWM %

Follow same pattern as BG_AtlasGrid's risk management.

**Step 2: Compile test**

---

### Task 7: Create FanoSuperpositionGrid.mq5 - Main EA

**Files:**
- Create: `...Experts\FanoSuperpositionGrid.mq5`
- Depends on: Tasks 1-6

**Step 1: Write the main EA file**

Include all 6 headers:
```cpp
#include <Trade\Trade.mqh>
#include <Fano\FanoOctonion.mqh>
#include <Fano\FanoDecomposition.mqh>
#include <Fano\FanoRegime.mqh>
#include <Fano\FanoBayesian.mqh>
#include <Fano\FanoGrid.mqh>
#include <Fano\FanoRisk.mqh>
```

Input parameters — exactly as listed in design spec section 11.

Global objects:
```cpp
CTrade g_trade;
FanoRegime g_regime;
FanoBayesian g_bayesian;
FanoGrid g_grid;
FanoRisk g_risk;
datetime g_lastSignalCheck = 0;
```

OnInit:
```
1. Print banner with all settings
2. g_regime.Init(_Symbol, PERIOD_CURRENT)
3. g_bayesian.Init() + LoadFromFile
4. g_grid.Init(_Symbol, MagicNumber)
5. g_grid.SyncWithBroker()
6. g_risk.Init()
7. g_trade.SetExpertMagicNumber(MagicNumber)
8. g_trade.SetDeviationInPoints(30)
```

OnDeinit:
```
1. g_bayesian.SaveToFile(...)
2. g_regime.Deinit()
```

OnTick:
```
1. g_grid.ManagePositions(g_trade)          // ALWAYS — hidden SL/TP every tick

2. if (TimeCurrent() - g_lastSignalCheck < CheckInterval): return
   g_lastSignalCheck = TimeCurrent()

3. g_risk.CheckDailyReset()
4. if (!g_risk.IsSafe(DailyDDLimit, MaxDDLimit)):
     Print("RISK BLOCK: ", g_risk.block_reason)
     return

5. if (!TradeEnabled): return

6. // Get close prices for Bayesian
   double close[]
   CopyClose(_Symbol, PERIOD_CURRENT, 0, 100, close)
   ArraySetAsSeries(close, true)

7. // Get ATR
   double atr[]
   CopyBuffer(g_regime.atr_handle, 0, 0, 1, atr)

8. // Update regime
   g_regime.Update(ConfidenceThreshold, InvertedMode)

9. // Update Bayesian
   g_bayesian.UpdateCounters(close, 100)

10. // Get Bayesian vote for active triple
    double pattern_vote = g_bayesian.GetTripleVote(g_regime.active_triple, close)

11. // Combine signals: 70% regime + 30% Bayesian
    double final_signal = RegimeWeight * g_regime.regime_confidence + PatternWeight * (pattern_vote * 2.0)

12. int direction = 0
    if (MathAbs(final_signal) >= ConfidenceThreshold):
      direction = (final_signal > 0) ? 1 : -1

13. if (direction == 0): return

14. // Compute lot and grid distances
    g_grid.ComputeLotAndDistances(atr[0], g_regime.decomp.chaos_level, SLDollars, TPDollars, DynTPPercent, ATRMultiplier, RollSLMult)

15. // Grid entry logic
    if (g_grid.count == 0):
      // First entry
      g_grid.OpenPosition(direction, g_trade)
    elif (g_grid.count < MaxPositions):
      // Check if price reached next grid level (moved AGAINST us)
      if (g_grid.PriceReachedNextLevel(SymbolInfoDouble(_Symbol, SYMBOL_BID), direction)):
        // Check direction matches existing grid
        if (g_grid.positions[0].direction == direction):
          g_grid.OpenPosition(direction, g_trade)

16. // Print status
    Print(StringFormat("FANO | Dir:%+d | Conf:%.3f | Triple:%d | Chaos:%.2f | Grid:%d/%d | Lot:%.4f",
          direction, final_signal, g_regime.active_triple, g_regime.decomp.chaos_level,
          g_grid.count, MaxPositions, g_grid.base_lot))
```

**Step 2: Compile the full EA**

```bash
"/c/Program Files/Atlas Funded MT5 Terminal/MetaEditor64.exe" /compile:"...MQL5\Experts\FanoSuperpositionGrid.mq5" /log
```
Expected: 0 errors, 0 warnings (or minimal warnings)

**Step 3: Fix any compilation errors**

---

### Task 8: Copy to DEPLOY and Create Inverted Copy

**Files:**
- Copy: all .mqh and .mq5 to `C:\Users\jimjj\Music\QuantumChildren\QuantumTradingLibrary\DEPLOY\`
- Create: `FanoSuperpositionGrid_INV.mq5` — identical but `InvertedMode = true` as default

**Step 1: Create DEPLOY directory if not exists**

```bash
mkdir -p "C:/Users/jimjj/Music/QuantumChildren/QuantumTradingLibrary/DEPLOY"
```

**Step 2: Copy all Fano files**

```bash
cp -r ...Include/Fano/ DEPLOY/Fano/
cp ...Experts/FanoSuperpositionGrid.mq5 DEPLOY/
```

**Step 3: Create inverted copy**

Copy FanoSuperpositionGrid.mq5 to FanoSuperpositionGrid_INV.mq5, change default `InvertedMode = true` and `MagicNumber = 212002`.

---

### Task 9: Strategy Tester Validation

**Step 1: Run backtest**

In MT5 Strategy Tester:
- EA: FanoSuperpositionGrid
- Symbol: BTCUSD
- Period: M1
- Mode: Every tick based on real ticks (if available) or OHLC M1
- Date: 2025-06-01 to 2026-03-01
- Deposit: $10,000
- Verify:
  - EA opens and closes positions
  - SL never exceeds $1 per position
  - TP targets $3 per position
  - Partial close at 50% works
  - Grid spacing adapts with chaos level
  - Direction changes with regime

**Step 2: Check logs**

Review Expert tab for:
- FANO status prints showing regime changes
- Position open/close with correct lot sizes
- No error messages

**Step 3: Document results**

Save backtest report screenshot + equity curve to DEPLOY folder.

---

## Dependency Graph

```
Task 1 (Octonion) ─────┐
                        ├── Task 2 (Decomposition)
                        │       │
                        │       ├── Task 3 (Regime)
                        │       │       │
                        │       │       ├── Task 4 (Bayesian)
                        │       │       │
                        │       │       └── Task 5 (Grid) ←── uses chaos_level
                        │       │
Task 6 (Risk) ──────────┤
                        │
                        └── Task 7 (Main EA) ←── depends on all above
                                │
                                ├── Task 8 (Deploy + Inverted)
                                │
                                └── Task 9 (Backtest)
```

Tasks 1 and 6 can run in parallel (independent).
Tasks 4 and 5 can run in parallel (both depend on 3 but not each other).
