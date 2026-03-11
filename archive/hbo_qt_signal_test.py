"""
HBO-Quantum-TE Signal Test — Quick drop on price data.
Tests if the optimizer can find profitable entry/exit params on OHLC data.
Uses synthetic BTC-like price series (no terminal needed).
"""

import numpy as np
import sys
sys.path.insert(0, '.')

from hbo_quantum_te import HBOQuantumTE


def generate_btc_like_series(n_bars=2000, seed=42):
    """Generate realistic BTC-like M15 OHLC with trends + noise."""
    rng = np.random.default_rng(seed)

    # Base price walk with momentum
    returns = rng.standard_normal(n_bars) * 0.003  # ~0.3% per bar
    # Add trending regimes
    regime = np.zeros(n_bars)
    for i in range(0, n_bars, 200):
        regime[i:i+200] = rng.choice([-1, 0, 1]) * 0.001
    returns += regime

    close = np.zeros(n_bars)
    close[0] = 85000.0  # BTC starting price
    for i in range(1, n_bars):
        close[i] = close[i-1] * (1 + returns[i])

    # Generate OHLC from close
    spread = close * 0.002
    high = close + np.abs(rng.standard_normal(n_bars)) * spread
    low = close - np.abs(rng.standard_normal(n_bars)) * spread
    opn = close + rng.standard_normal(n_bars) * spread * 0.3

    return opn, high, low, close


def extract_features(opn, high, low, close, i, lookback=20):
    """Extract 10D feature vector from price bars ending at index i."""
    if i < lookback:
        return None

    c = close[i-lookback:i]
    h = high[i-lookback:i]
    l = low[i-lookback:i]
    o = opn[i-lookback:i]

    # Normalized features
    price_range = np.max(h) - np.min(l)
    if price_range < 1e-10:
        return None

    features = np.array([
        (c[-1] - np.mean(c)) / (np.std(c) + 1e-10),           # 0: z-score
        (c[-1] - c[0]) / (price_range + 1e-10),                # 1: trend
        np.std(np.diff(c)) / (np.mean(c) + 1e-10) * 100,      # 2: volatility
        (c[-1] - np.min(l)) / (price_range + 1e-10),           # 3: position in range
        np.mean(h - l) / (np.mean(c) + 1e-10) * 100,          # 4: avg bar range %
        (h[-1] - c[-1]) / (h[-1] - l[-1] + 1e-10),            # 5: upper wick ratio
        (c[-1] - l[-1]) / (h[-1] - l[-1] + 1e-10),            # 6: lower wick ratio
        np.corrcoef(np.arange(lookback), c)[0,1],              # 7: linear correlation
        (c[-1] - c[-5]) / (c[-5] + 1e-10) * 100,              # 8: 5-bar momentum %
        (np.max(c[-5:]) - np.min(c[-5:])) / (price_range + 1e-10),  # 9: recent range ratio
    ])

    return features


def trading_objective(params):
    """
    Objective function: optimize entry/exit thresholds.
    params[0]: z-score entry threshold (buy when z < -threshold)
    params[1]: z-score entry threshold (sell when z > threshold)
    params[2]: trend filter minimum
    params[3]: volatility filter max
    params[4]: momentum threshold
    params[5]: take-profit multiplier (of avg range)
    params[6]: stop-loss multiplier (of avg range)
    params[7]: position-in-range buy threshold
    params[8]: correlation filter
    params[9]: holding period (bars)
    """
    # These get optimized by HBO-Quantum-TE
    # Return value will be negative profit (since we minimize)
    # Actual backtest happens in run_backtest()
    return 0  # placeholder — real eval in backtest loop


def run_backtest(params, opn, high, low, close):
    """Run a simple backtest with the given parameters."""
    z_buy = abs(params[0])
    z_sell = abs(params[1])
    trend_min = params[2]
    vol_max = abs(params[3]) + 0.1
    mom_thresh = params[4]
    tp_mult = abs(params[5]) + 0.5
    sl_mult = abs(params[6]) + 0.3
    range_buy = params[7]
    corr_filter = params[8]
    hold_bars = max(3, int(abs(params[9]) * 10))

    n = len(close)
    lookback = 20
    trades = []
    position = 0  # 0=flat, 1=long, -1=short
    entry_price = 0
    entry_bar = 0
    tp_price = 0
    sl_price = 0

    for i in range(lookback, n):
        feat = extract_features(opn, high, low, close, i)
        if feat is None:
            continue

        z_score = feat[0]
        trend = feat[1]
        vol = feat[2]
        pos_in_range = feat[3]
        avg_range_pct = feat[4]
        momentum = feat[8]
        correlation = feat[7]

        avg_range = close[i] * avg_range_pct / 100

        if position == 0:
            # Entry logic
            if (z_score < -z_buy and trend > trend_min and vol < vol_max
                    and pos_in_range < range_buy and correlation > corr_filter):
                # BUY signal
                position = 1
                entry_price = close[i]
                entry_bar = i
                tp_price = entry_price + tp_mult * avg_range
                sl_price = entry_price - sl_mult * avg_range

            elif (z_score > z_sell and trend < -trend_min and vol < vol_max
                  and momentum < -mom_thresh):
                # SELL signal
                position = -1
                entry_price = close[i]
                entry_bar = i
                tp_price = entry_price - tp_mult * avg_range
                sl_price = entry_price + sl_mult * avg_range

        elif position == 1:  # Long
            # Check exits
            if high[i] >= tp_price:
                trades.append(tp_price - entry_price)
                position = 0
            elif low[i] <= sl_price:
                trades.append(sl_price - entry_price)
                position = 0
            elif i - entry_bar >= hold_bars:
                trades.append(close[i] - entry_price)
                position = 0

        elif position == -1:  # Short
            if low[i] <= tp_price:
                trades.append(entry_price - tp_price)
                position = 0
            elif high[i] >= sl_price:
                trades.append(entry_price - sl_price)
                position = 0
            elif i - entry_bar >= hold_bars:
                trades.append(entry_price - close[i])
                position = 0

    # Close any open position
    if position == 1:
        trades.append(close[-1] - entry_price)
    elif position == -1:
        trades.append(entry_price - close[-1])

    return trades


def main():
    print("=" * 70)
    print("HBO-Quantum-TE Signal Test — Quick Drop")
    print("=" * 70)

    # Generate test data
    opn, high, low, close = generate_btc_like_series(n_bars=3000, seed=42)
    print(f"Generated {len(close)} bars of BTC-like M15 data")
    print(f"Price range: ${np.min(low):.0f} - ${np.max(high):.0f}")

    # Split: train on first 2000, test on last 1000
    train_end = 2000

    opn_train, high_train, low_train, close_train = opn[:train_end], high[:train_end], low[:train_end], close[:train_end]
    opn_test, high_test, low_test, close_test = opn[train_end:], high[train_end:], low[train_end:], close[train_end:]

    print(f"Train: {train_end} bars | Test: {len(close_test)} bars")

    # Define objective: negative total profit (minimize = maximize profit)
    def objective(params):
        trades = run_backtest(params, opn_train, high_train, low_train, close_train)
        if len(trades) == 0:
            return 1e6  # no trades = bad
        total_pnl = sum(trades)
        n_trades = len(trades)
        win_rate = sum(1 for t in trades if t > 0) / n_trades if n_trades > 0 else 0

        # Penalize too few trades or terrible win rate
        if n_trades < 5:
            return 1e6
        if win_rate < 0.2:
            return 1e5

        # Objective: negative profit-factor-adjusted PnL
        avg_win = np.mean([t for t in trades if t > 0]) if any(t > 0 for t in trades) else 0
        avg_loss = abs(np.mean([t for t in trades if t <= 0])) if any(t <= 0 for t in trades) else 1
        profit_factor = avg_win / (avg_loss + 1e-10)

        return -(total_pnl * profit_factor * win_rate)

    # Run HBO-Quantum-TE optimization
    print("\nOptimizing trading parameters with HBO-Quantum-TE...")
    print("-" * 70)

    opt = HBOQuantumTE(
        obj_func=objective,
        dim=10,
        lb=-3.0,
        ub=3.0,
        pop_size=40,
        max_iter=150,
        seed=123
    )
    best_params, best_obj = opt.optimize(verbose=True)

    print(f"\nBest objective: {best_obj:.2f}")
    print(f"Best params: {np.round(best_params, 4)}")

    # Evaluate on TRAIN
    print("\n" + "=" * 70)
    print("TRAIN RESULTS")
    print("=" * 70)
    train_trades = run_backtest(best_params, opn_train, high_train, low_train, close_train)
    if train_trades:
        total = sum(train_trades)
        wins = [t for t in train_trades if t > 0]
        losses = [t for t in train_trades if t <= 0]
        wr = len(wins) / len(train_trades) * 100
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        pf = abs(sum(wins) / (sum(losses) + 1e-10)) if losses else float('inf')
        print(f"  Trades:        {len(train_trades)}")
        print(f"  Total PnL:     ${total:.2f}")
        print(f"  Win Rate:      {wr:.1f}%")
        print(f"  Avg Win:       ${avg_win:.2f}")
        print(f"  Avg Loss:      ${avg_loss:.2f}")
        print(f"  Profit Factor: {pf:.2f}")
    else:
        print("  No trades generated.")

    # Evaluate on TEST (out-of-sample)
    print("\n" + "=" * 70)
    print("TEST RESULTS (Out-of-Sample)")
    print("=" * 70)
    test_trades = run_backtest(best_params, opn_test, high_test, low_test, close_test)
    if test_trades:
        total = sum(test_trades)
        wins = [t for t in test_trades if t > 0]
        losses = [t for t in test_trades if t <= 0]
        wr = len(wins) / len(test_trades) * 100
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        pf = abs(sum(wins) / (sum(losses) + 1e-10)) if losses else float('inf')
        print(f"  Trades:        {len(test_trades)}")
        print(f"  Total PnL:     ${total:.2f}")
        print(f"  Win Rate:      {wr:.1f}%")
        print(f"  Avg Win:       ${avg_win:.2f}")
        print(f"  Avg Loss:      ${avg_loss:.2f}")
        print(f"  Profit Factor: {pf:.2f}")

        # Verdict
        print("\n" + "=" * 70)
        if total > 0 and pf > 1.2 and wr > 40:
            print("VERDICT: Looks viable. Worth converting to MQL5.")
        elif total > 0 and pf > 1.0:
            print("VERDICT: Marginal. Needs more tuning before MQL5.")
        else:
            print("VERDICT: Not ready. Overfits train, fails OOS.")
        print("=" * 70)
    else:
        print("  No trades on test set.")
        print("\nVERDICT: Dead signal — no trades generated OOS.")


if __name__ == '__main__':
    main()
