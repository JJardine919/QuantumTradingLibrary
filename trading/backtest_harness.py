"""
J-TEAR Backtest Harness — Walk-Forward Comparison
Train 4 days / Test 2 days / 6 months of BTCUSDT 5M data

Compares 5 converted neural experts against ETARE baseline.
Features per bar: close-open, high-open, low-open, vol/1000, RSI(14), CCI(14), ATR(14), MACD, MACD_signal

Usage:
    python backtest_harness.py
"""

import sys
import os
import time
import json
import traceback
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch

# Add QTL to path
QTL = Path(r"C:\Users\jimjj\Music\QuantumChildren\QuantumTradingLibrary")
sys.path.insert(0, str(QTL))

# ============================================================
# Config
# ============================================================
DATA_PATH = Path(r"C:\Users\jimjj\Downloads\binance_70mo\data\BTCUSDT_5M.csv")
TRAIN_DAYS = 4
TEST_DAYS = 2
STEP_DAYS = 2  # Slide by test window size
LOOKBACK_MONTHS = 6
BARS_PER_DAY = 288  # 24h * 60min / 5min
CONTEXT_BARS = 120  # Most experts need 120 bars context (Conformer uses 100)
MIN_CONFIDENCE = 0.1  # Minimum confidence to open a trade
TRAIN_EPOCHS = 3  # Light training per window (keep it fast)
RESULTS_DIR = QTL / "backtest_results"

# Trade simulation
SPREAD_POINTS = 2.0  # Spread in price units
MAX_HOLD_BARS = 60  # Close after 5 hours max hold (60 * 5min)


# ============================================================
# Feature Engineering
# ============================================================
def compute_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """RSI(14) normalized 0-1."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    avg_gain[period] = np.mean(gain[1:period + 1])
    avg_loss[period] = np.mean(loss[1:period + 1])

    for i in range(period + 1, len(close)):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period

    rs = np.where(avg_loss > 0, avg_gain / (avg_loss + 1e-10), 100.0)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    return rsi / 100.0  # Normalize 0-1


def compute_cci(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """CCI(14) normalized by /500."""
    tp = (high + low + close) / 3.0
    cci = np.zeros_like(close)
    for i in range(period, len(close)):
        window = tp[i - period + 1:i + 1]
        sma = np.mean(window)
        mad = np.mean(np.abs(window - sma)) + 1e-10
        cci[i] = (tp[i] - sma) / (0.015 * mad)
    return cci / 500.0  # Rough normalization


def compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """ATR(14)."""
    tr = np.zeros_like(close)
    tr[0] = high[0] - low[0]
    for i in range(1, len(close)):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))

    atr = np.zeros_like(close)
    atr[period - 1] = np.mean(tr[:period])
    for i in range(period, len(close)):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def compute_macd(close: np.ndarray) -> tuple:
    """MACD(12,26,9). Returns (macd_line, signal_line)."""
    ema12 = np.zeros_like(close)
    ema26 = np.zeros_like(close)
    ema12[0] = close[0]
    ema26[0] = close[0]
    m12 = 2.0 / 13.0
    m26 = 2.0 / 27.0

    for i in range(1, len(close)):
        ema12[i] = close[i] * m12 + ema12[i - 1] * (1 - m12)
        ema26[i] = close[i] * m26 + ema26[i - 1] * (1 - m26)

    macd_line = ema12 - ema26

    signal = np.zeros_like(close)
    signal[0] = macd_line[0]
    m9 = 2.0 / 10.0
    for i in range(1, len(close)):
        signal[i] = macd_line[i] * m9 + signal[i - 1] * (1 - m9)

    return macd_line, signal


def build_features(df: pd.DataFrame) -> np.ndarray:
    """Build the 9-feature array from OHLCV data. Returns [N, 9]."""
    o = df['open'].values.astype(np.float64)
    h = df['high'].values.astype(np.float64)
    l = df['low'].values.astype(np.float64)
    c = df['close'].values.astype(np.float64)
    v = df['volume'].values.astype(np.float64)

    rsi = compute_rsi(c, 14)
    cci = compute_cci(h, l, c, 14)
    atr = compute_atr(h, l, c, 14)
    macd_main, macd_sig = compute_macd(c)

    features = np.column_stack([
        c - o,         # [0] close - open
        h - o,         # [1] high - open
        l - o,         # [2] low - open
        v / 1000.0,    # [3] volume / 1000
        rsi,           # [4] RSI(14) normalized
        cci,           # [5] CCI(14) normalized
        atr,           # [6] ATR(14)
        macd_main,     # [7] MACD main
        macd_sig,      # [8] MACD signal
    ])
    return features.astype(np.float32)


# ============================================================
# Trade Simulator
# ============================================================
class TradeSimulator:
    """Simple trade-by-trade simulation on bar data."""

    def __init__(self, close_prices: np.ndarray, spread: float = SPREAD_POINTS):
        self.close = close_prices
        self.spread = spread
        self.trades = []

    def simulate(self, signals: list) -> dict:
        """
        signals: list of (bar_idx, direction, confidence) tuples
        direction: 1=buy, -1=sell, 0=hold
        Returns dict with WR, PF, trades, max_dd, net_pnl
        """
        self.trades = []
        position = None  # (entry_bar, entry_price, direction)

        for bar_idx, direction, confidence in signals:
            if bar_idx >= len(self.close):
                break

            price = self.close[bar_idx]

            # Close existing position if opposite signal or max hold
            if position is not None:
                entry_bar, entry_price, pos_dir = position
                bars_held = bar_idx - entry_bar

                should_close = False
                if direction != 0 and direction != pos_dir:
                    should_close = True
                elif bars_held >= MAX_HOLD_BARS:
                    should_close = True

                if should_close:
                    if pos_dir == 1:  # Long
                        pnl = (price - entry_price) - self.spread
                    else:  # Short
                        pnl = (entry_price - price) - self.spread
                    self.trades.append({
                        'entry_bar': entry_bar,
                        'exit_bar': bar_idx,
                        'direction': pos_dir,
                        'entry_price': entry_price,
                        'exit_price': price,
                        'pnl': pnl,
                        'bars_held': bars_held,
                    })
                    position = None

            # Open new position
            if position is None and direction != 0 and confidence >= MIN_CONFIDENCE:
                position = (bar_idx, price, direction)

        # Close any open position at end
        if position is not None:
            entry_bar, entry_price, pos_dir = position
            price = self.close[-1]
            if pos_dir == 1:
                pnl = (price - entry_price) - self.spread
            else:
                pnl = (entry_price - price) - self.spread
            self.trades.append({
                'entry_bar': entry_bar,
                'exit_bar': len(self.close) - 1,
                'direction': pos_dir,
                'entry_price': entry_price,
                'exit_price': price,
                'pnl': pnl,
                'bars_held': len(self.close) - 1 - entry_bar,
            })

        return self._compute_metrics()

    def _compute_metrics(self) -> dict:
        if not self.trades:
            return {'wr': 0.0, 'pf': 0.0, 'trades': 0, 'net_pnl': 0.0, 'max_dd': 0.0}

        wins = [t for t in self.trades if t['pnl'] > 0]
        losses = [t for t in self.trades if t['pnl'] <= 0]

        wr = len(wins) / len(self.trades) * 100.0
        gross_profit = sum(t['pnl'] for t in wins) if wins else 0.0
        gross_loss = abs(sum(t['pnl'] for t in losses)) if losses else 0.0001
        pf = gross_profit / gross_loss if gross_loss > 0 else 99.0
        net_pnl = sum(t['pnl'] for t in self.trades)

        # Max drawdown from cumulative PnL
        cum_pnl = np.cumsum([t['pnl'] for t in self.trades])
        peak = np.maximum.accumulate(cum_pnl)
        dd = peak - cum_pnl
        max_dd = np.max(dd) if len(dd) > 0 else 0.0

        return {
            'wr': round(wr, 2),
            'pf': round(pf, 3),
            'trades': len(self.trades),
            'net_pnl': round(net_pnl, 2),
            'max_dd': round(max_dd, 2),
        }


# ============================================================
# Expert Wrappers — Unified Interface
# ============================================================
class ExpertWrapper:
    """Base class for wrapping each expert with a common interface."""
    name = "base"

    def init_model(self):
        raise NotImplementedError

    def predict(self, features_window: np.ndarray) -> tuple:
        """Returns (direction, confidence). features_window shape: [N, 9]"""
        raise NotImplementedError

    def train_on_window(self, features: np.ndarray, close_prices: np.ndarray):
        """Train on a window of data. Default: no-op for untrained baselines."""
        pass

    def reset(self):
        """Reset model to fresh random weights for each walk-forward window."""
        self.init_model()


class DACGLSTMWrapper(ExpertWrapper):
    name = "DACGLSTM"

    def init_model(self):
        from expert_dacglstm import DACGLSTMExpert
        self.model = DACGLSTMExpert()

    def predict(self, features_window: np.ndarray) -> tuple:
        window = features_window[-120:]
        if len(window) < 120:
            return (0, 0.0)
        return self.model.generate_signal(window)

    def train_on_window(self, features: np.ndarray, close_prices: np.ndarray):
        self.model.to_train_mode()
        opt_enc = torch.optim.Adam(self.model.encoder.parameters(), lr=1e-3)
        opt_task = torch.optim.Adam(self.model.task.parameters(), lr=1e-3)
        opt_actor = torch.optim.Adam(self.model.actor.parameters(), lr=1e-3)
        opt_critic = torch.optim.Adam(self.model.critic.parameters(), lr=1e-3)
        for epoch in range(TRAIN_EPOCHS):
            for i in range(CONTEXT_BARS, len(features) - 1, 10):
                state = features[i - CONTEXT_BARS:i]
                next_state = features[i - CONTEXT_BARS + 1:i + 1]
                reward = 1.0 if close_prices[i] > close_prices[i - 1] else -1.0
                batch = {
                    'states': torch.tensor(state, dtype=torch.float32).unsqueeze(0),
                    'accounts': torch.zeros(1, 12, dtype=torch.float32),
                    'actions': torch.zeros(1, 6, dtype=torch.float32),
                    'rewards': torch.tensor([[reward, 0.0, 0.0]], dtype=torch.float32),
                    'next_states': torch.tensor(next_state, dtype=torch.float32).unsqueeze(0),
                    'next_accounts': torch.zeros(1, 12, dtype=torch.float32),
                }
                try:
                    self.model.train_step(batch, opt_enc, opt_task, opt_actor, opt_critic)
                except Exception:
                    pass
        self.model.to_gpu()


class MambaWrapper(ExpertWrapper):
    name = "Mamba"

    def init_model(self):
        from expert_mamba import MambaExpert
        self.model = MambaExpert()

    def predict(self, features_window: np.ndarray) -> tuple:
        window = features_window[-120:]
        if len(window) < 120:
            return (0, 0.0)
        return self.model.generate_signal(window)

    def train_on_window(self, features: np.ndarray, close_prices: np.ndarray):
        self.model.to_cpu()
        for epoch in range(TRAIN_EPOCHS):
            for i in range(CONTEXT_BARS, len(features) - 2, 10):
                state = features[i - CONTEXT_BARS:i].flatten()
                reward_curr = [1.0 if close_prices[i] > close_prices[i - 1] else -1.0, 0.0, 0.0]
                reward_next = [1.0 if close_prices[i + 1] > close_prices[i] else -1.0, 0.0, 0.0]
                profitable = close_prices[i] > close_prices[i - 5] if i >= 5 else False
                batch = {
                    'state': torch.tensor(state, dtype=torch.float32).unsqueeze(0),
                    'account': torch.zeros(1, 12, dtype=torch.float32),
                    'action': torch.zeros(1, 6, dtype=torch.float32),
                    'reward_curr': torch.tensor([reward_curr], dtype=torch.float32),
                    'reward_next': torch.tensor([reward_next], dtype=torch.float32),
                    'profitable': torch.tensor([profitable], dtype=torch.bool),
                }
                try:
                    self.model.train_step(batch)
                except Exception:
                    pass
        self.model.to_gpu()


class ConformerWrapper(ExpertWrapper):
    name = "Conformer"

    def init_model(self):
        from expert_conformer import ConformerExpert
        self.model = ConformerExpert()

    def predict(self, features_window: np.ndarray) -> tuple:
        window = features_window[-100:]  # Conformer uses 100 bars
        if len(window) < 100:
            return (0, 0.0)
        bars_t = torch.tensor(window, dtype=torch.float32)
        return self.model.generate_signal(bars_t)

    def train_on_window(self, features: np.ndarray, close_prices: np.ndarray):
        self.model.to(torch.device("cpu"))
        self.model.train()
        opt_enc = torch.optim.Adam(self.model.encoder.parameters(), lr=1e-3)
        opt_actor = torch.optim.Adam(self.model.actor.parameters(), lr=1e-3)
        opt_critic = torch.optim.Adam(self.model.critic.parameters(), lr=1e-3)
        for epoch in range(TRAIN_EPOCHS):
            for i in range(100, len(features) - 2, 10):
                bars = features[i - 100:i]
                reward_now = [1.0 if close_prices[i] > close_prices[i - 1] else -1.0, 0.0, 0.0]
                reward_next = [1.0 if close_prices[i + 1] > close_prices[i] else -1.0, 0.0, 0.0]
                batch = {
                    'bars': torch.tensor(bars, dtype=torch.float32).unsqueeze(0),
                    'account': torch.zeros(1, 12, dtype=torch.float32),
                    'actions': torch.zeros(1, 6, dtype=torch.float32),
                    'rewards_now': torch.tensor([reward_now], dtype=torch.float32),
                    'rewards_next': torch.tensor([reward_next], dtype=torch.float32),
                }
                try:
                    self.model.train_step(batch, opt_actor, opt_critic, opt_enc)
                except Exception:
                    pass
        self.model.eval()


class TimeMoEWrapper(ExpertWrapper):
    name = "TimeMoE"

    def init_model(self):
        from expert_timemoe import TimeMoESystem, generate_signal as tmoe_gen
        self.model = TimeMoESystem()
        self._gen = tmoe_gen

    def predict(self, features_window: np.ndarray) -> tuple:
        window = features_window[-120:]
        if len(window) < 120:
            return (0, 0.0)
        feat_t = torch.tensor(window, dtype=torch.float32)
        return self._gen(feat_t)

    def train_on_window(self, features: np.ndarray, close_prices: np.ndarray):
        from expert_timemoe import train_step as tmoe_train
        self.model.to(torch.device("cpu"))
        self.model.train()
        params = list(self.model.parameters())
        optimizer = torch.optim.Adam(params, lr=1e-3)
        optimizers = {'encoder': optimizer, 'actor': optimizer, 'critic': optimizer, 'director': optimizer}
        for epoch in range(TRAIN_EPOCHS):
            for i in range(CONTEXT_BARS, len(features) - 1, 10):
                state = features[i - CONTEXT_BARS:i].flatten()
                next_state = features[i - CONTEXT_BARS + 1:i + 1].flatten()
                reward = 1.0 if close_prices[i] > close_prices[i - 1] else -1.0
                batch = {
                    'state': torch.tensor(state, dtype=torch.float32).unsqueeze(0),
                    'next_state': torch.tensor(next_state, dtype=torch.float32).unsqueeze(0),
                    'account': torch.zeros(1, 13, dtype=torch.float32),
                    'next_account': torch.zeros(1, 13, dtype=torch.float32),
                    'action': torch.zeros(1, 6, dtype=torch.float32),
                    'reward': torch.tensor([[reward]], dtype=torch.float32),
                    'profitable': torch.tensor([reward > 0], dtype=torch.bool),
                }
                try:
                    tmoe_train(batch, optimizers, device=torch.device("cpu"))
                except Exception:
                    pass
        self.model.eval()


class StockformerWrapper(ExpertWrapper):
    name = "Stockformer"

    def init_model(self):
        from expert_stockformer import MultiTaskStockformer, generate_signal as sf_gen
        self.model = MultiTaskStockformer()
        self._gen = sf_gen

    def predict(self, features_window: np.ndarray) -> tuple:
        window = features_window[-120:]
        if len(window) < 120:
            return (0, 0.0)
        return self._gen(window, self.model)

    def train_on_window(self, features: np.ndarray, close_prices: np.ndarray):
        from expert_stockformer import train_step as sf_train
        self.model.to(torch.device("cpu"))
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        for epoch in range(TRAIN_EPOCHS):
            for i in range(CONTEXT_BARS, len(features) - 1, 10):
                state = features[i - CONTEXT_BARS:i].flatten()
                next_state = features[i - CONTEXT_BARS + 1:i + 1].flatten()
                reward = 1.0 if close_prices[i] > close_prices[i - 1] else -1.0
                batch = {
                    'state': torch.tensor(state, dtype=torch.float32).unsqueeze(0),
                    'next_state': torch.tensor(next_state, dtype=torch.float32).unsqueeze(0),
                    'account': torch.zeros(1, 12, dtype=torch.float32),
                    'next_account': torch.zeros(1, 12, dtype=torch.float32),
                    'action': torch.zeros(1, 6, dtype=torch.float32),
                    'reward': torch.tensor([[reward, 0.0, 0.0]], dtype=torch.float32),
                    'profitable': torch.tensor([reward > 0], dtype=torch.bool),
                }
                try:
                    sf_train(batch, self.model, optimizer, device=torch.device("cpu"))
                except Exception:
                    pass
        self.model.eval()


# ============================================================
# Walk-Forward Engine
# ============================================================
def run_walk_forward(expert: ExpertWrapper, features: np.ndarray, close: np.ndarray,
                     train_bars: int, test_bars: int, step_bars: int) -> list:
    """Run walk-forward backtest for one expert. Returns list of window results."""
    results = []
    total_bars = len(features)
    window_num = 0

    start = 0
    while start + train_bars + test_bars <= total_bars:
        window_num += 1
        train_end = start + train_bars
        test_end = train_end + test_bars

        train_features = features[start:train_end]
        train_close = close[start:train_end]
        test_features = features[train_end:test_end]
        test_close = close[train_end:test_end]

        # Reset and train
        expert.reset()
        t0 = time.time()
        try:
            expert.train_on_window(train_features, train_close)
        except Exception as e:
            print(f"  [WARN] Training failed window {window_num}: {e}")
        train_time = time.time() - t0

        # Generate signals on test window
        signals = []
        for i in range(CONTEXT_BARS, len(test_features)):
            try:
                ctx_start = max(0, i - CONTEXT_BARS)
                window = test_features[ctx_start:i]
                direction, confidence = expert.predict(window)
                if direction != 0:
                    signals.append((i, direction, confidence))
            except Exception:
                pass

        # Simulate trades
        sim = TradeSimulator(test_close)
        metrics = sim.simulate(signals)
        metrics['window'] = window_num
        metrics['train_time'] = round(train_time, 1)
        results.append(metrics)

        print(f"  W{window_num:3d} | {metrics['trades']:3d}t | "
              f"WR {metrics['wr']:5.1f}% | PF {metrics['pf']:5.2f} | "
              f"PnL ${metrics['net_pnl']:+8.2f} | DD ${metrics['max_dd']:7.2f} | "
              f"{train_time:.0f}s")

        start += step_bars

    return results


def aggregate_results(results: list) -> dict:
    """Aggregate window results into overall metrics."""
    if not results:
        return {'wr': 0, 'pf': 0, 'trades': 0, 'net_pnl': 0, 'max_dd': 0}

    total_trades = sum(r['trades'] for r in results)
    if total_trades == 0:
        return {'wr': 0, 'pf': 0, 'trades': 0, 'net_pnl': 0, 'max_dd': 0}

    # Weighted average WR by trade count
    wr = sum(r['wr'] * r['trades'] for r in results) / total_trades
    net_pnl = sum(r['net_pnl'] for r in results)
    max_dd = max(r['max_dd'] for r in results)

    # Overall PF
    total_profit = sum(r['net_pnl'] for r in results if r['net_pnl'] > 0)
    total_loss = abs(sum(r['net_pnl'] for r in results if r['net_pnl'] < 0)) + 0.0001
    pf = total_profit / total_loss

    # WR consistency (how many windows > 50%)
    above_50 = sum(1 for r in results if r['wr'] > 50 and r['trades'] > 0)
    total_active = sum(1 for r in results if r['trades'] > 0)
    consistency = above_50 / total_active * 100 if total_active > 0 else 0

    return {
        'wr': round(wr, 2),
        'pf': round(pf, 3),
        'trades': total_trades,
        'net_pnl': round(net_pnl, 2),
        'max_dd': round(max_dd, 2),
        'windows': len(results),
        'active_windows': total_active,
        'consistency': round(consistency, 1),
    }


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 70)
    print("J-TEAR BACKTEST HARNESS")
    print(f"Train {TRAIN_DAYS}d / Test {TEST_DAYS}d / Walk-Forward")
    print("=" * 70)

    # Load data
    print(f"\nLoading {DATA_PATH.name}...")
    df = pd.read_csv(DATA_PATH, parse_dates=['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Use last 6 months
    cutoff = df['timestamp'].max() - timedelta(days=LOOKBACK_MONTHS * 30)
    df = df[df['timestamp'] >= cutoff].reset_index(drop=True)
    print(f"Data: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
    print(f"Bars: {len(df):,} ({len(df) / BARS_PER_DAY:.0f} days)")

    # Build features
    print("Computing features (RSI, CCI, ATR, MACD)...")
    features = build_features(df)
    close = df['close'].values.astype(np.float32)
    print(f"Features shape: {features.shape}")

    # Walk-forward params
    train_bars = TRAIN_DAYS * BARS_PER_DAY
    test_bars = TEST_DAYS * BARS_PER_DAY
    step_bars = STEP_DAYS * BARS_PER_DAY
    n_windows = (len(features) - train_bars) // step_bars
    print(f"Windows: ~{n_windows} (train={train_bars} bars, test={test_bars} bars)")

    # Create results directory
    RESULTS_DIR.mkdir(exist_ok=True)

    # Run each expert
    experts = [
        DACGLSTMWrapper(),
        MambaWrapper(),
        ConformerWrapper(),
        TimeMoEWrapper(),
        StockformerWrapper(),
    ]

    all_results = {}
    for exp in experts:
        print(f"\n{'=' * 70}")
        print(f"EXPERT: {exp.name}")
        print(f"{'=' * 70}")

        try:
            exp.init_model()
            results = run_walk_forward(exp, features, close, train_bars, test_bars, step_bars)
            agg = aggregate_results(results)
            all_results[exp.name] = {
                'aggregate': agg,
                'windows': results,
            }

            print(f"\n--- {exp.name} SUMMARY ---")
            print(f"  Win Rate:    {agg['wr']:.1f}%")
            print(f"  Profit Fac:  {agg['pf']:.3f}")
            print(f"  Total Trades: {agg['trades']}")
            print(f"  Net PnL:     ${agg['net_pnl']:+,.2f}")
            print(f"  Max DD:      ${agg['max_dd']:,.2f}")
            print(f"  Consistency: {agg.get('consistency', 0):.1f}% windows > 50% WR")
        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            all_results[exp.name] = {'error': str(e)}

    # ETARE baseline comparison
    print(f"\n{'=' * 70}")
    print("ETARE BASELINE (from extinction_results.json)")
    print("=" * 70)
    print("  VDJ_05 avg:     WR 43.3%, PF 1.22, ~1150 trades/round")
    print("  Syncytin best:  WR 36.9%, PF 1.65, 260 trades")
    print("  TEQA_04 avg:    WR 35.7%, PF 1.09, ~2100 trades/round")

    # Final comparison table
    print(f"\n{'=' * 70}")
    print("COMPARISON TABLE")
    print("=" * 70)
    print(f"{'Expert':<20} {'WR%':>6} {'PF':>7} {'Trades':>7} {'Net PnL':>10} {'MaxDD':>8} {'Consist':>8}")
    print("-" * 70)

    for name, data in all_results.items():
        if 'error' in data:
            print(f"{name:<20} FAILED: {data['error'][:40]}")
        else:
            a = data['aggregate']
            print(f"{name:<20} {a['wr']:5.1f}% {a['pf']:6.3f} {a['trades']:7d} "
                  f"${a['net_pnl']:+9.2f} ${a['max_dd']:7.2f} {a.get('consistency', 0):6.1f}%")

    # ETARE baselines for reference
    print("-" * 70)
    print(f"{'ETARE VDJ_05':<20} {'43.3%':>6} {'1.220':>7} {'~1150':>7} {'N/A':>10} {'N/A':>8} {'80.0%':>8}")
    print(f"{'ETARE Syncytin':<20} {'36.9%':>6} {'1.650':>7} {'~260':>7} {'N/A':>10} {'N/A':>8} {'60.0%':>8}")
    print(f"{'QT Army Best':<20} {'50.6%':>6} {'N/A':>7} {'~2000':>7} {'N/A':>10} {'N/A':>8} {'N/A':>8}")

    # Save results
    out_path = RESULTS_DIR / f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, 'w') as f:
        # Convert numpy types for JSON serialization
        def convert(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            return obj

        json.dump(all_results, f, indent=2, default=convert)
    print(f"\nResults saved: {out_path}")


if __name__ == '__main__':
    main()
