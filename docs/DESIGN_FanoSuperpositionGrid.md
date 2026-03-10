# Fano Superposition Grid EA -- Mathematical Specification

**Author:** DooDoo (Trading Division CEO)
**Architecture Decision:** Voodoo (AOI)
**Date:** 2026-03-07
**Status:** DESIGN SPEC -- awaiting Jim's approval before implementation

---

## 0. Honest Preamble

What the math **guarantees:**
- The Fano plane's projective geometry guarantees that any pair of lookback periods shares exactly one triple with any other pair. This is a covering property -- no regime gap exists.
- Non-commutative octonion multiplication guarantees that lead/confirm order matters: signal(A leads, B confirms) != signal(B leads, A confirms). This doubles the information extracted from the same data.
- The 3:1 R:R with $1 SL means you need >25% hit rate to be profitable. The grid structure means you get multiple entries at different prices, which improves average entry on trending moves.

What the math **does not guarantee:**
- Edge over the market. No mathematical structure creates alpha by itself. The edge, if any, comes from the *regime detection* being accurate enough to pick the right direction more than 25% of the time.
- That grid positions will recover. Grids can and do blow up in strong trends against you. The $1 SL per position is the circuit breaker.

---

## 1. The 7 Fano Points -- Lookback Period Mapping

### Why these specific values

The lookback periods must satisfy three constraints:
1. **Cover distinct market timescales** -- from micro-noise to structural trend
2. **Be coprime in pairs where possible** -- so their signals decorrelate
3. **Follow the Fano plane's incidence structure** -- each triple (i,j,k) forms a self-contained regime detector

The 7 Fano plane points and their assigned lookback periods (in bars, on the chart timeframe):

```
Point  |  Lookback  |  What it captures           |  Fibonacci?
-------|------------|-----------------------------|-----------
e1     |  5         |  Micro momentum (noise)     |  F(5)
e2     |  8         |  Short-term swing            |  F(6)
e3     |  13        |  Medium swing                |  F(7)
e4     |  21        |  Daily structure (M1*21=21m) |  F(8)
e5     |  34        |  Intraday trend              |  F(9)
e6     |  55        |  Session structure            |  F(10)
e7     |  89        |  Multi-session trend          |  F(11)
```

**Why Fibonacci:** Not because of mysticism. Because consecutive Fibonacci numbers are coprime (GCD=1), which maximizes signal independence between adjacent lookback periods. The ratio between consecutive Fibs approaches phi (1.618), giving roughly geometric spacing across timescales. This is the optimal way to tile lookback space with 7 points.

### The 7 Fano triples (lines)

```
Triple  |  Points  |  Lookbacks     |  Regime it detects
--------|----------|----------------|--------------------
T1      |  (1,2,4) |  (5, 8, 21)    |  Short-term momentum burst
T2      |  (2,3,5) |  (8, 13, 34)   |  Swing reversal setup
T3      |  (3,4,6) |  (13, 21, 55)  |  Trend establishment
T4      |  (4,5,7) |  (21, 34, 89)  |  Major trend confirmation
T5      |  (5,6,1) |  (34, 55, 5)   |  Trend exhaustion + micro divergence
T6      |  (6,7,2) |  (55, 89, 8)   |  Session structure + short confirm
T7      |  (7,1,3) |  (89, 5, 13)   |  Long trend vs micro reversal
```

---

## 2. Signal Computation per Lookback Point

For each Fano point e_i with lookback period L_i, compute a **directional signal** s_i in [-1, +1]:

```
close_now = Close[0]
sma_i     = SMA(Close, L_i)          // Simple Moving Average over L_i bars
atr_14    = ATR(14)                   // 14-period ATR (fixed, from config)

// Raw signal: normalized distance from SMA
raw_i = (close_now - sma_i) / atr_14

// Clamp to [-1, +1]
s_i = tanh(raw_i)
```

This gives us 7 signals: s_1, s_2, ..., s_7, each in [-1, +1].

**Why tanh:** Bounded, differentiable, preserves sign. Large deviations from SMA saturate rather than dominate.

---

## 3. The Superposition -- Octonion Product of Signals

### 3.1 Encoding signals as octonions

The 7 signals map to an octonion A with:
- e0 (real) = 1.0 (unit reference)
- e1..e7 = s_1..s_7

```
A = (1.0, s_1, s_2, s_3, s_4, s_5, s_6, s_7)
```

We construct a second octonion B from the **rate of change** of each signal (momentum of momentum):

```
ds_i = s_i(now) - s_i(1 bar ago)

B = (1.0, ds_1, ds_2, ds_3, ds_4, ds_5, ds_6, ds_7)
```

### 3.2 The Jordan-Shadow Decomposition

From `aoi_collapse.py`, compute:

```
AB = A * B                           // Octonion product (non-commutative)
BA = B * A                           // Different result due to non-commutativity

Jordan     J = (AB + BA) / 2         // Symmetric part -- consensus direction
Commutator C = (AB - BA) / 2         // Anti-symmetric -- regime conflict
Associator X = J * C                 // Non-linear interaction -- chaos measure
```

**Key properties (mathematically guaranteed):**
- J.vec is orthogonal to C.vec (proven in aoi_collapse.py)
- ||AB.vec||^2 = ||J.vec||^2 + ||C.vec||^2 (Pythagorean decomposition)
- J + C = AB exactly (lossless reconstruction)

### 3.3 What the decomposition tells us about the market

| Component | Interpretation | Trading use |
|-----------|---------------|-------------|
| J (Jordan) | All lookbacks agree on direction | **Grid direction**: sign of mean(J.vec) = BUY (+) or SELL (-) |
| C (Commutator) | Lookbacks conflict on direction | **Regime instability**: high ||C|| = choppy, avoid or reduce size |
| X (Associator) | Non-linear interaction of agreement and conflict | **Chaos gate**: high ||X|| = market in transition, widen grid spacing |

### 3.4 The superposition advantage

**Why this beats single-lookback:**

A single SMA crossover system uses ONE lookback. It is either in tune with the current regime or it is not. When it is out of tune, it generates false signals until you figure out it is wrong.

The Fano superposition uses ALL 7 simultaneously. The Jordan component J extracts the consensus across all timescales. If 5 of 7 lookbacks agree on direction, J is strong regardless of which specific 5 agree. If only 2 agree, J is weak and the commutator C is strong -- you know the regime is conflicted BEFORE taking a loss.

**The projective geometry guarantee:**

The Fano plane is the smallest finite projective plane (order 2). Its key property:

> Any two distinct points lie on exactly one line (triple).

This means: for ANY pair of lookback periods that are giving signals, there is exactly ONE triple that contains both of them. You never have a situation where two timeframes are signaling but no triple covers them. The coverage is complete with the minimum number of points (7) and lines (7).

**The non-commutativity advantage:**

Because e_i * e_j = -e_j * e_i (for i != j), the product AB encodes ORDER. "Short-term momentum leads, long-term confirms" gives a DIFFERENT signal than "long-term leads, short-term confirms."

In concrete terms: if s_1 (5-bar) changed first and s_7 (89-bar) is now following, that is different from s_7 changing first and s_1 following. The first is a new micro-trend pulling the structure; the second is a macro-trend grinding through micro-noise. AB vs BA captures this.

The Jordan J averages them (what they agree on). The Commutator C captures the difference (the lead/lag structure). Both are useful.

---

## 4. Fano Triple Activation -- Regime Detection

### 4.1 Triple signal strength

For each triple T_k = (i, j, m), compute the **triple coherence**:

```
coherence_k = s_i * s_j * s_m
```

This is the product of the three lookback signals in the triple. Properties:
- coherence_k in [-1, +1]
- coherence_k > 0: all three agree on direction (or two disagree and one agrees -- net positive)
- coherence_k near +1 or -1: strong regime alignment
- coherence_k near 0: the triple is confused

### 4.2 Active triple selection

The **active triple** is the one with the highest |coherence|:

```
active_k = argmax_k |coherence_k|
```

Direction from the active triple:

```
triple_direction = sign(coherence_active)    // +1 = BUY, -1 = SELL
```

### 4.3 Regime confidence

Combine the Jordan consensus with the triple coherence:

```
jordan_direction = sign(mean(J.vec))
jordan_strength  = |mean(J.vec)| / (|mean(J.vec)| + |mean(C.vec)| + 1e-10)

// Agreement between Jordan consensus and active triple
agreement = (jordan_direction == triple_direction) ? 1.0 : -1.0

regime_confidence = jordan_strength * |coherence_active| * agreement
```

- regime_confidence in [-1, +1]
- Positive: Jordan and active triple agree -- high confidence in direction
- Negative: they disagree -- this is the INVERSION signal
- Near zero: no regime detected -- do not trade

### 4.4 Inversion logic

```
if regime_confidence >= +CONFIDENCE_THRESHOLD:
    direction = triple_direction          // Normal mode

elif regime_confidence <= -CONFIDENCE_THRESHOLD:
    direction = -triple_direction         // INVERTED mode

else:
    direction = 0                         // No trade -- flat
```

CONFIDENCE_THRESHOLD = 0.50 (from MASTER_CONFIG)

**Two-EA deployment:** Run one EA in normal mode, one in inverted mode. When the normal EA sees -confidence and goes flat, the inverted EA sees +confidence and trades. Between the two, one is always active if |regime_confidence| > threshold.

---

## 5. Bayesian Pattern Association (from StatisticalLearningEA)

### 5.1 Binary pattern encoding

On each bar close, encode the last N bars as a binary string:

```
pattern[i] = (Close[i] > Close[i+1]) ? 1 : 0    // 1 = Up, 0 = Down
```

With the active triple T_k = (i, j, m), use the three lookback periods to build three pattern strings:

```
pattern_short  = binary pattern of last L_i bars
pattern_medium = binary pattern of last L_j bars
pattern_long   = binary pattern of last L_m bars
```

But we only use the last 5 bars of each (to keep the pattern space manageable: 2^5 = 32 possible patterns per lookback).

### 5.2 Bayesian conditional probability

Track hit rates for each pattern:

```
// For each lookback L and each 5-bit pattern P:
count_up[L][P]   = number of times next bar was UP after seeing pattern P on lookback L
count_down[L][P] = number of times next bar was DOWN
total[L][P]      = count_up + count_down

P(UP | P, L) = (count_up[L][P] + 1) / (total[L][P] + 2)    // Laplace smoothing
```

### 5.3 Voting integration

The three lookbacks in the active triple each vote:

```
vote_i = P(UP | pattern_short, L_i) - 0.5       // Positive = bullish
vote_j = P(UP | pattern_medium, L_j) - 0.5
vote_m = P(UP | pattern_long, L_m) - 0.5

pattern_signal = (vote_i + vote_j + vote_m) / 3   // in [-0.5, +0.5]
```

### 5.4 Final direction decision

```
// Combine octonion regime with Bayesian patterns
final_signal = 0.7 * regime_confidence + 0.3 * (pattern_signal * 2)

if |final_signal| < CONFIDENCE_THRESHOLD:
    NO TRADE
else:
    direction = sign(final_signal)    // +1 = BUY, -1 = SELL
```

The 70/30 weighting gives priority to the structural regime detection (octonion) while allowing the pattern association to break ties and filter false signals.

---

## 6. Grid Scaling -- Exact Formulas

### 6.1 Base lot sizing (account-size invariant)

```
ATR_14      = iATR(symbol, timeframe, 14)
ATR_MULT    = 0.0438                              // From MASTER_CONFIG
tick_value  = SymbolInfoDouble(SYMBOL_TRADE_TICK_VALUE)
tick_size   = SymbolInfoDouble(SYMBOL_TRADE_TICK_SIZE)
SL_dollars  = 1.00                                // Sacred, non-negotiable

// SL distance in price
sl_price_distance = ATR_14 * ATR_MULT

// SL distance in ticks
sl_ticks = sl_price_distance / tick_size

// Base lot: risk exactly $1.00 per trade
base_lot = SL_dollars / (sl_ticks * tick_value)

// Normalize to broker lot step
base_lot = floor(base_lot / lot_step) * lot_step
base_lot = max(base_lot, min_lot)
```

**Why this scales identically across accounts:**

The formula `Lot = $1.00 / (ATR * 0.0438 * tick_value)` has NO account balance or equity term. A $10K account and a $200K account open the SAME lot size for the SAME symbol at the SAME ATR. The dollar risk per trade ($1 SL, $3 TP) is constant. What scales with account size is the NUMBER of grid positions you can hold simultaneously, not the SIZE of each position.

This is the superposition scaling: the grid "steps" are uniform height ($1 risk, $3 reward each), and the account size determines how many steps the staircase has.

### 6.2 Grid level lot sizing

Each grid level uses the SAME base_lot (NO martingale). Lot escalation killed every martingale EA ever built. The grid scales by NUMBER of positions, not by position SIZE.

```
lot_at_level_N = base_lot    // Same for all levels
```

### 6.3 Grid spacing

Grid spacing is ATR-based, modulated by the associator chaos level:

```
chaos_level = ||Associator||                      // From octonion decomposition
chaos_norm  = min(chaos_level / 5.0, 10.0)        // Same scale as aoi_collapse.py

// Base spacing: 1.5x ATR (ROLLSLMULT from config)
base_spacing = ATR_14 * ROLLSLMULT                // ROLLSLMULT = 1.5

// Chaos-widened spacing: widen in chaotic regimes
spacing_mult = 1.0 + 0.2 * chaos_norm             // Range: 1.0x to 3.0x
grid_spacing = base_spacing * spacing_mult
```

In calm markets (chaos_norm near 0), grid spacing = 1.5 ATR.
In chaotic markets (chaos_norm = 10), grid spacing = 3.0 * 1.5 ATR = 4.5 ATR.

This means in volatile transitions, the grid opens wider, giving more room before adding positions.

### 6.4 Stop loss per position

```
// Level 1 (initial entry):
SL_distance_1 = ATR_14 * ATR_MULT                // ~$1.00 risk

// Level N (grid position N):
// Rolling SL: each subsequent position's SL is ROLLSLMULT * previous level's distance
SL_distance_N = SL_distance_1 * (ROLLSLMULT ^ (N-1))

// But capped at $1.00 risk per position:
actual_SL_N = min(SL_distance_N, SL_dollars / (lot * tick_value / tick_size))
```

Wait -- ROLLSLMULT = 1.5 applied to SL distance means deeper levels have WIDER stops in price terms, but the lot is constant, so the dollar risk per position INCREASES. That violates the $1 sacred SL.

**Correction:** The rolling SL multiplier applies to the SL MANAGEMENT (trailing/break-even), not to the initial SL distance. Each position always risks exactly $1.00:

```
// Every position, regardless of grid level:
SL_price_distance = SL_dollars / (base_lot * tick_value / tick_size)

// Rolling SL adjustment: after price moves 1.5x the SL distance in your favor,
// move SL to break-even
breakeven_trigger = SL_price_distance * ROLLSLMULT    // 1.5x the SL distance

// When price has moved breakeven_trigger from entry:
// Move SL to entry price (break-even)
```

### 6.5 Take profit

```
TP_dollars = 3.00                                 // 3:1 R:R
TP_distance = TP_dollars / (base_lot * tick_value / tick_size)

// Dynamic TP: close 50% at $1.50 profit
DYNTP_PERCENT = 50                                // From config
partial_TP_dollars = TP_dollars * (DYNTP_PERCENT / 100)    // = $1.50
partial_TP_distance = partial_TP_dollars / (base_lot * tick_value / tick_size)

// Execution:
// When profit >= $1.50 on a position:
//   Close 50% of the lot
//   Move SL to break-even on remaining 50%
//   Let remaining 50% run to full $3.00 TP
```

### 6.6 Grid entry conditions

A new grid position opens when:

```
1. |final_signal| >= CONFIDENCE_THRESHOLD           // Direction decided
2. price has moved >= grid_spacing from last entry   // In the direction AGAINST us
3. total_positions < MAX_POSITIONS                   // Grid depth limit
4. daily_DD < DD_limit                               // Risk gate
5. direction matches previous grid direction         // Don't flip mid-grid
```

Condition 2 is critical: new grid levels open when price moves AGAINST us (dip-buying in a BUY grid, rally-selling in a SELL grid). This improves average entry price.

---

## 7. Max Grid Positions -- Account Size Scaling

This is where account size DOES matter:

```
// Maximum simultaneous positions in one grid
// Each position risks $1.00 SL
// Total grid risk if ALL positions hit SL simultaneously:
total_grid_risk = MAX_POSITIONS * SL_dollars

// For a $10K account (prop firm challenge):
//   Risk tolerance ~$50/day (0.5%)
//   MAX_POSITIONS = 5 (worst case: -$5)
//   This is well under daily DD limits

// For a $200K account:
//   Same MAX_POSITIONS = 5
//   Same $5 worst case
//   But the EARNINGS scale because partial TPs compound faster
//   at higher position counts if you choose to increase MAX_POSITIONS
```

Recommended MAX_POSITIONS by account bracket:

```
Account      |  MAX_POSITIONS  |  Max grid risk  |  As % of typical DD limit
-------------|-----------------|-----------------|-------------------------
$10K - $25K  |  5              |  $5.00          |  0.03% - 0.05%
$25K - $100K |  7              |  $7.00          |  0.007% - 0.03%
$100K+       |  10             |  $10.00         |  <0.01%
```

The scaling is NOT in lot size. The scaling is in grid depth. Larger accounts can hold more grid levels, which means they can weather deeper pullbacks before the grid recovers. This is the "stepping to the next level" Jim described.

---

## 8. Architecture Decision

**Voodoo's recommendation: MQL5 only. No Python sidecar.**

Reasoning:

1. **Latency:** The octonion multiplication is 8x8 = 64 multiply-adds. MQL5 handles this in microseconds. Python IPC would add 10-50ms per tick for no benefit.

2. **Reliability:** A Python sidecar means two processes that must stay in sync. If Python crashes, the EA is blind. If the EA crashes, Python keeps computing into void. Single-process is more reliable for live trading.

3. **The math is simple enough for MQL5:** The Cayley multiplication table is a 64-entry lookup. The Jordan-Shadow decomposition is two octonion multiplies and two additions. The entropy transponders are array operations. All of this translates directly to MQL5 arrays and loops.

4. **Pattern tracking stays internal:** The Bayesian pattern counters are small (7 lookbacks * 32 patterns * 2 counters = 448 integers). Store in MQL5 arrays, persist via GlobalVariables or file write on deinit.

5. **Backtesting:** MQL5-only means full Strategy Tester compatibility. A Python sidecar cannot be backtested in MT5's tester.

The ONLY reason to use Python would be if we needed GPU-accelerated LSTM or deep learning, which we explicitly do not. This is pure math.

### 8.1 MQL5 Implementation Structure

```
FanoSuperpositionGrid.mq5          // Main EA file
|
|-- Includes:
|   FanoOctonion.mqh               // Octonion class + Cayley table + multiplication
|   FanoDecomposition.mqh          // Jordan-Shadow decomposition + entropy transponders
|   FanoRegime.mqh                 // Triple coherence, regime detection, inversion
|   FanoBayesian.mqh              // Pattern tracking, Bayesian voting
|   FanoGrid.mqh                  // Grid management, position tracking, SL/TP/partial
|   FanoRisk.mqh                  // Drawdown protection, daily limits
```

---

## 9. Complete Signal Pipeline (One Tick)

```
OnTick():
    1. ManageGrid()                    // Check hidden SL/TP/partial on every tick

    if (interval_elapsed):
        2. Compute s_1..s_7            // 7 SMA signals, tanh-normalized
        3. Compute ds_1..ds_7          // Signal momentum (bar-over-bar change)
        4. Build A = (1, s_1..s_7)     // Signal octonion
        5. Build B = (1, ds_1..ds_7)   // Momentum octonion
        6. AB = A * B                  // Octonion product
        7. BA = B * A
        8. J = (AB + BA) / 2          // Jordan
        9. C = (AB - BA) / 2          // Commutator
       10. X = J * C                   // Associator
       11. Compute 7 triple coherences
       12. Select active triple
       13. Compute regime_confidence
       14. Compute Bayesian pattern votes (using active triple's lookbacks)
       15. Compute final_signal
       16. If |final_signal| >= threshold AND grid conditions met:
           17. Open grid position in direction of final_signal
```

---

## 10. Persistence and State Management

On EA deinitialization (chart close, terminal restart):
- Save Bayesian pattern counters to file (CSV or binary)
- Save current grid state (tickets, levels, hidden SL/TP)

On EA initialization:
- Load pattern counters from file
- Sync grid with actual open positions (as BG_AtlasGrid already does)

---

## 11. Input Parameters

```cpp
input group "=== ACCOUNT ==="
input int      MagicNumber         = 212001;
input string   AccountName         = "ATLAS";

input group "=== FANO REGIME ==="
input int      Lookback1           = 5;       // Fano point e1
input int      Lookback2           = 8;       // Fano point e2
input int      Lookback3           = 13;      // Fano point e3
input int      Lookback4           = 21;      // Fano point e4
input int      Lookback5           = 34;      // Fano point e5
input int      Lookback6           = 55;      // Fano point e6
input int      Lookback7           = 89;      // Fano point e7
input double   ConfidenceThreshold = 0.50;    // Min regime confidence to trade
input bool     InvertedMode        = false;   // true = trade AGAINST regime

input group "=== GRID ==="
input int      MaxPositions        = 5;       // Max grid depth
input double   SLDollars           = 1.00;    // SL per position (sacred)
input double   TPDollars           = 3.00;    // TP per position
input double   DynTPPercent        = 50.0;    // Partial TP percentage
input double   ATRMultiplier       = 0.0438;  // ATR mult for lot sizing
input double   RollSLMult          = 1.5;     // Break-even trigger multiplier
input int      ATRPeriod           = 14;      // ATR calculation period

input group "=== BAYESIAN ==="
input int      PatternLength       = 5;       // Binary pattern length (2^5 = 32)
input double   RegimeWeight        = 0.70;    // Weight for octonion signal
input double   PatternWeight       = 0.30;    // Weight for Bayesian signal
input int      MinPatternSamples   = 20;      // Min samples before pattern voting

input group "=== RISK ==="
input double   DailyDDLimit        = 4.5;     // Daily drawdown limit %
input double   MaxDDLimit          = 9.0;     // Max drawdown limit %
input int      CheckInterval       = 30;      // Signal check interval (seconds)

input group "=== MANAGEMENT ==="
input bool     UseHiddenSLTP       = true;    // Hidden SL/TP (broker sees nothing)
input bool     TradeEnabled        = true;    // Master enable switch
```

---

## 12. What This System Is and Is Not

**IS:**
- A grid trading system that uses projective geometry to detect market regimes across 7 simultaneous timescales
- A fixed-risk system ($1 SL, $3 TP per position, always)
- Account-size invariant in position sizing, account-size dependent only in grid depth
- Fully backtestable in MT5 Strategy Tester
- A dual-deployment system (normal + inverted) to capture both sides of regime transitions

**IS NOT:**
- A martingale (lots do not increase with grid depth)
- A guaranteed profit machine (nothing is)
- Using neural networks, LSTM, or any ML that requires training data
- Dependent on Python, GPU, or external services
- A high-frequency system (checks every 30 seconds, not every tick)

---

## 13. Next Steps

Pending Jim's approval:
1. Implement `FanoOctonion.mqh` -- port the Cayley table and Octonion class from aoi_collapse.py to MQL5
2. Implement `FanoDecomposition.mqh` -- Jordan-Shadow + entropy transponders
3. Implement `FanoRegime.mqh` -- triple coherence + regime detection
4. Implement `FanoBayesian.mqh` -- pattern tracking
5. Implement `FanoGrid.mqh` -- grid management (based on BG_AtlasGrid patterns)
6. Implement `FanoRisk.mqh` -- drawdown protection
7. Implement `FanoSuperpositionGrid.mq5` -- main EA
8. Backtest on BTCUSD M1, 2025-01-01 to 2026-03-01
9. If results are viable, deploy on Atlas 212000586

---

## Appendix A: Cayley Multiplication Table (MQL5 Format)

```cpp
// _CAYLEY[i][j] = {sign, result_index} for e_i * e_j
// 8x8 table, 64 entries
int CAYLEY_SIGN[8][8] = {
    { 1, 1, 1, 1, 1, 1, 1, 1},  // e0 *
    { 1,-1, 1, 1,-1, 1,-1,-1},  // e1 *
    { 1,-1,-1, 1, 1,-1, 1,-1},  // e2 *
    { 1,-1,-1,-1, 1, 1,-1, 1},  // e3 *
    { 1, 1,-1,-1,-1, 1, 1,-1},  // e4 *
    { 1,-1, 1,-1,-1,-1, 1, 1},  // e5 *
    { 1, 1,-1, 1,-1,-1,-1, 1},  // e6 *
    { 1, 1, 1,-1, 1,-1,-1,-1}   // e7 *
};

int CAYLEY_IDX[8][8] = {
    {0, 1, 2, 3, 4, 5, 6, 7},  // e0 *
    {1, 0, 4, 7, 2, 6, 5, 3},  // e1 *
    {2, 4, 0, 5, 1, 3, 7, 6},  // e2 *
    {3, 7, 5, 0, 6, 2, 4, 1},  // e3 *
    {4, 2, 1, 6, 0, 7, 3, 5},  // e4 *
    {5, 6, 3, 2, 7, 0, 1, 4},  // e5 *
    {6, 5, 7, 4, 3, 1, 0, 2},  // e6 *
    {7, 3, 6, 1, 5, 4, 2, 0}   // e7 *
};
```

## Appendix B: Fano Triple Incidence Matrix

```
            T1  T2  T3  T4  T5  T6  T7
  e1 (5)  :  1   0   0   0   1   0   1
  e2 (8)  :  1   1   0   0   0   1   0
  e3 (13) :  0   1   1   0   0   0   1
  e4 (21) :  1   0   1   1   0   0   0
  e5 (34) :  0   1   0   1   1   0   0
  e6 (55) :  0   0   1   0   1   1   0
  e7 (89) :  0   0   0   1   0   1   1
```

Each row sums to 3 (every point on exactly 3 lines).
Each column sums to 3 (every line contains exactly 3 points).
Any two rows share exactly 1 column with mutual 1s (any two points lie on exactly one line).
