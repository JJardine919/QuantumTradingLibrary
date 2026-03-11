"""
jtear_engine.py - J-TEAR: Hybrid Ensemble Breeding Engine
==========================================================
Walk-forward ensemble breeding for the Quantum-Teenagers army.
Three-tier league system: Active Roster, Minor League, and breeding pipeline.

Uses M1 data, 4-day train / 2-day test rolling windows.
Manages roster promotions/relegations and breeding operators.
Does NOT modify existing Quantum-Teenagers folder contents.

Copyright 2026, QuantumChildren Trading Systems
"""

import sys
import os
import json
import copy
import time
import math
import random
import sqlite3
import shutil
import hashlib
import logging
import traceback
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple, Optional, Any, Set
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path

import types
import tempfile
import importlib

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# GPU Setup
# ---------------------------------------------------------------------------
try:
    import torch
    import torch_directml
    GPU_DEVICE = torch_directml.device()
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    GPU_DEVICE = None
    try:
        import torch
    except ImportError:
        torch = None

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
)
logger = logging.getLogger("J-TEAR")
logger.setLevel(logging.INFO)

# ===========================================================================
# CONFIGURATION
# ===========================================================================

JTEAR_CONFIG = {
    # Walk-forward parameters
    "timeframe": "M1",
    "train_days": 4,
    "test_days": 2,
    "step_days": 2,
    "lookback_months": 12,

    # Roster sizes
    "active_roster_size": 10,
    "minor_league_size": 15,
    "max_minor_league_size": 25,

    # Breeding
    "offspring_per_cycle": 3,
    "mutation_rate": 0.10,
    "radiation_rate": 0.30,

    # Evaluation
    "min_trades_threshold": 3,
    "ensemble_confidence_threshold": 0.60,
    "relegation_buffer_windows": 3,
    "signal_stride": 3,          # Check signals every N bars (M1)
    "signal_timeout": 2.0,       # Max seconds per generate_signal()
    "expert_timeout": 60.0,      # Max seconds per expert-symbol window
    "skip_bars": 100,            # Warmup bars at window start

    # Symbols
    "symbols": ["BTCUSDT", "ETHUSDT", "XAUUSDT"],

    # Scoring weights
    "scoring_weights": {
        "profit_factor": 0.30,
        "win_rate": 0.25,
        "sharpe": 0.20,
        "max_drawdown": 0.15,
        "consistency": 0.10,
    },

    # Breeding operator probabilities (must sum to 1.0)
    "breeding_probabilities": {
        "MUTATION": 0.30,
        "CROSSOVER_SAME": 0.20,
        "RADIATION": 0.15,
        "BLEND": 0.10,
        "ELITE_CLONE": 0.10,
        "GENESIS": 0.10,
        "CROSSOVER_DIFF": 0.05,
    },

    # Paths
    "qt_library": r"C:\Users\jimjj\OneDrive\Videos\Quantum-Teenagers",
    "spare_champions": r"C:\Users\jimjj\OneDrive\Videos\Quantum-Teenagers\spare champions",
    "data_dir": r"C:\Users\jimjj\Downloads\binance_70mo\data",
    "qtl_dir": r"C:\Users\jimjj\Music\QuantumChildren\QuantumTradingLibrary",
    "workspace": r"C:\Users\jimjj\Music\QuantumChildren\QuantumTradingLibrary\jtear_workspace",
    "state_file": r"C:\Users\jimjj\Music\QuantumChildren\QuantumTradingLibrary\jtear_state.json",
}


# ===========================================================================
# ENUMS
# ===========================================================================

class Tier(Enum):
    ACTIVE = "active"
    MINOR = "minor"


class BreedingOp(Enum):
    MUTATION = "MUTATION"
    CROSSOVER_SAME = "CROSSOVER_SAME"
    CROSSOVER_DIFF = "CROSSOVER_DIFF"
    BLEND = "BLEND"
    ELITE_CLONE = "ELITE_CLONE"
    RADIATION = "RADIATION"
    GENESIS = "GENESIS"


class ExpertType(Enum):
    PTH_CONV1D = "pth_conv1d"          # .pth files (Conv1D architecture)
    PTH_ARMY = "pth_army"              # army_champion .pth files
    JSON_WEIGHTS = "json_weights"       # JSON raw weight configs
    MQ5_SYSTEM = "mq5_system"           # Full MQ5 folder with Test.mq5
    PYTHON_EXPERT = "python_expert"    # Python ExpertBase subclass (#1 rankers)


# ===========================================================================
# DATA CLASSES
# ===========================================================================

@dataclass
class WalkForwardWindow:
    """One walk-forward evaluation window."""
    window_id: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str


@dataclass
class SimulatedTrade:
    """A single simulated trade."""
    ticket: int
    symbol: str
    direction: int           # 1=BUY, -1=SELL
    entry_price: float
    entry_time: str
    sl_price: float
    tp_price: float
    sl_distance: float
    volume: float
    status: str = "OPEN"     # OPEN, WIN, LOSS, SIGNAL_CLOSE, WINDOW_END
    exit_price: float = 0.0
    exit_time: str = ""
    pnl: float = 0.0


@dataclass
class ExpertPerformance:
    """Performance metrics for one expert on one window."""
    expert_name: str
    window_id: int = 0
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    net_pnl: float = 0.0
    max_drawdown: float = 0.0
    peak_equity: float = 0.0
    current_equity: float = 0.0
    trade_pnls: List[float] = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        return self.wins / self.total_trades if self.total_trades > 0 else 0.0

    @property
    def profit_factor(self) -> float:
        if self.gross_loss == 0:
            return 5.0 if self.gross_profit > 0 else 0.0
        return min(5.0, self.gross_profit / abs(self.gross_loss))

    @property
    def sharpe(self) -> float:
        if len(self.trade_pnls) < 2:
            return 0.0
        mean_pnl = np.mean(self.trade_pnls)
        std_pnl = np.std(self.trade_pnls)
        if std_pnl == 0:
            return 0.0
        # Annualize: assume ~250 trading days, ~20 trades per window
        return float((mean_pnl / std_pnl) * np.sqrt(250))

    def record_trade(self, trade: SimulatedTrade):
        self.total_trades += 1
        self.net_pnl += trade.pnl
        self.current_equity += trade.pnl
        self.trade_pnls.append(trade.pnl)
        if trade.pnl > 0:
            self.wins += 1
            self.gross_profit += trade.pnl
        else:
            self.losses += 1
            self.gross_loss += trade.pnl
        if self.current_equity > self.peak_equity:
            self.peak_equity = self.current_equity
        dd = self.peak_equity - self.current_equity
        if dd > self.max_drawdown:
            self.max_drawdown = dd


@dataclass
class ExpertEntry:
    """An expert in the J-TEAR system."""
    name: str
    architecture: str            # e.g., "AutoBots", "SAC", "Chimera", "conv1d"
    expert_type: ExpertType
    source_path: str             # Original folder or file path
    weight_path: Optional[str]   # Path to .pth or .json weights
    tier: Tier = Tier.MINOR
    vote_weight: float = 0.0
    composite_score: float = 0.0
    score_history: List[float] = field(default_factory=list)
    consecutive_underperform: int = 0
    lineage: List[str] = field(default_factory=list)  # Parent names
    breeding_op: str = "ORIGINAL"  # How this expert was created

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "architecture": self.architecture,
            "expert_type": self.expert_type.value,
            "source_path": self.source_path,
            "weight_path": self.weight_path,
            "tier": self.tier.value,
            "vote_weight": self.vote_weight,
            "composite_score": self.composite_score,
            "score_history": self.score_history[-20:],  # Keep last 20
            "consecutive_underperform": self.consecutive_underperform,
            "lineage": self.lineage,
            "breeding_op": self.breeding_op,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ExpertEntry":
        return cls(
            name=d["name"],
            architecture=d["architecture"],
            expert_type=ExpertType(d["expert_type"]),
            source_path=d["source_path"],
            weight_path=d.get("weight_path"),
            tier=Tier(d["tier"]),
            vote_weight=d.get("vote_weight", 0.0),
            composite_score=d.get("composite_score", 0.0),
            score_history=d.get("score_history", []),
            consecutive_underperform=d.get("consecutive_underperform", 0),
            lineage=d.get("lineage", []),
            breeding_op=d.get("breeding_op", "ORIGINAL"),
        )


# ===========================================================================
# DATA LAYER
# ===========================================================================

class DataLayer:
    """
    Loads and serves M1 CSV data. Provides slicing by date range and
    resampling to higher timeframes.
    """

    def __init__(self, data_dir: str, symbols: List[str]):
        self.data_dir = data_dir
        self.symbols = symbols
        self._data: Dict[str, pd.DataFrame] = {}
        self._np_cache: Dict[str, Dict[str, np.ndarray]] = {}

    def load(self):
        """Load M1 CSVs for all symbols."""
        for symbol in self.symbols:
            csv_path = os.path.join(self.data_dir, f"{symbol}_1M.csv")
            if not os.path.exists(csv_path):
                logger.warning(f"M1 data not found for {symbol}: {csv_path}")
                continue

            logger.info(f"Loading {csv_path}...")
            t0 = time.time()
            df = pd.read_csv(csv_path, parse_dates=["timestamp"])
            df = df.sort_values("timestamp").reset_index(drop=True)

            # Standard column names
            col_map = {"open": "open", "high": "high", "low": "low",
                        "close": "close"}
            if "volume" in df.columns:
                col_map["volume"] = "tick_volume"
            df = df.rename(columns=col_map)
            if "tick_volume" not in df.columns:
                df["tick_volume"] = 0.0

            self._data[symbol] = df

            # Pre-cache numpy arrays for fast access
            self._np_cache[symbol] = {
                "open": df["open"].values.astype(np.float64),
                "high": df["high"].values.astype(np.float64),
                "low": df["low"].values.astype(np.float64),
                "close": df["close"].values.astype(np.float64),
                "volume": df["tick_volume"].values.astype(np.float64),
                "timestamp": df["timestamp"].values,
            }

            elapsed = time.time() - t0
            logger.info(f"  {symbol}: {len(df):,} bars loaded in {elapsed:.1f}s "
                        f"({df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]})")

    def get_date_range(self) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """Return the min/max dates across all loaded data."""
        starts, ends = [], []
        for sym, df in self._data.items():
            starts.append(df["timestamp"].iloc[0])
            ends.append(df["timestamp"].iloc[-1])
        return min(starts), max(ends)

    def get_bar_indices(self, symbol: str, start_date: str,
                        end_date: str) -> Tuple[int, int]:
        """Return (start_idx, end_idx) for a date range."""
        if symbol not in self._data:
            # Try alias
            for s in self._data:
                if s.replace("USDT", "") == symbol.replace("USD", "").replace("USDT", ""):
                    symbol = s
                    break
        if symbol not in self._data:
            return (0, 0)

        df = self._data[symbol]
        start_dt = pd.Timestamp(start_date)
        end_dt = pd.Timestamp(end_date)
        mask = (df["timestamp"] >= start_dt) & (df["timestamp"] < end_dt)
        indices = df.index[mask].tolist()
        if not indices:
            return (0, 0)
        return (indices[0], indices[-1])

    def get_numpy(self, symbol: str) -> Optional[Dict[str, np.ndarray]]:
        """Return pre-cached numpy arrays for a symbol."""
        if symbol in self._np_cache:
            return self._np_cache[symbol]
        # Try alias
        for s in self._np_cache:
            if s.replace("USDT", "") == symbol.replace("USD", "").replace("USDT", ""):
                return self._np_cache[s]
        return None

    def compute_features(self, symbol: str, start_idx: int,
                         end_idx: int) -> Optional[np.ndarray]:
        """
        Compute feature matrix for a symbol over a bar range.
        Returns shape (num_bars, 8): [open, high, low, close, volume,
                                       rsi_14, atr_14, ema_ratio]
        """
        np_data = self.get_numpy(symbol)
        if np_data is None:
            return None

        closes = np_data["close"][start_idx:end_idx+1]
        highs = np_data["high"][start_idx:end_idx+1]
        lows = np_data["low"][start_idx:end_idx+1]
        opens = np_data["open"][start_idx:end_idx+1]
        volumes = np_data["volume"][start_idx:end_idx+1]

        n = len(closes)
        if n < 20:
            return None

        # RSI-14
        rsi = self._compute_rsi(closes, 14)

        # ATR-14
        atr = self._compute_atr(highs, lows, closes, 14)

        # EMA ratio (fast/slow)
        ema_fast = self._compute_ema(closes, 9)
        ema_slow = self._compute_ema(closes, 21)
        ema_ratio = np.where(ema_slow != 0, ema_fast / ema_slow, 1.0)

        # Normalize OHLCV for model input
        close_mean = np.mean(closes)
        close_std = np.std(closes) + 1e-10
        norm_open = (opens - close_mean) / close_std
        norm_high = (highs - close_mean) / close_std
        norm_low = (lows - close_mean) / close_std
        norm_close = (closes - close_mean) / close_std
        norm_vol = volumes / (np.mean(volumes) + 1e-10)

        features = np.column_stack([
            norm_open, norm_high, norm_low, norm_close,
            norm_vol, rsi / 100.0, atr / (close_mean + 1e-10), ema_ratio
        ])

        return features

    @staticmethod
    def _compute_rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
        """Vectorized RSI computation."""
        n = len(closes)
        rsi = np.full(n, 50.0)
        if n < period + 1:
            return rsi
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)

        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])

        for i in range(period, len(deltas)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            if avg_loss == 0:
                rsi[i + 1] = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi[i + 1] = 100.0 - (100.0 / (1.0 + rs))
        return rsi

    @staticmethod
    def _compute_atr(highs: np.ndarray, lows: np.ndarray,
                     closes: np.ndarray, period: int = 14) -> np.ndarray:
        """Vectorized ATR computation."""
        n = len(closes)
        atr = np.full(n, 0.0)
        if n < period + 1:
            atr[:] = np.abs(closes) * 0.002
            return atr

        tr = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(
                np.abs(highs[1:] - closes[:-1]),
                np.abs(lows[1:] - closes[:-1])
            )
        )
        tr = np.concatenate([[tr[0] if len(tr) > 0 else 0.0], tr])
        atr[period] = np.mean(tr[:period + 1])
        alpha = 1.0 / period
        for i in range(period + 1, n):
            atr[i] = alpha * tr[i] + (1.0 - alpha) * atr[i - 1]
        atr[:period + 1] = np.where(
            atr[:period + 1] == 0,
            np.abs(closes[:period + 1]) * 0.002,
            atr[:period + 1]
        )
        return atr

    @staticmethod
    def _compute_ema(data: np.ndarray, period: int) -> np.ndarray:
        """Exponential moving average."""
        ema = np.zeros_like(data)
        if len(data) < period:
            return ema
        k = 2.0 / (period + 1)
        ema[period - 1] = np.mean(data[:period])
        for i in range(period, len(data)):
            ema[i] = data[i] * k + ema[i - 1] * (1 - k)
        return ema


# ===========================================================================
# WALK-FORWARD WINDOW MANAGER
# ===========================================================================

class WindowManager:
    """Generates rolling walk-forward windows from the data."""

    def __init__(self, config: dict):
        self.train_days = config["train_days"]
        self.test_days = config["test_days"]
        self.step_days = config["step_days"]
        self.lookback_months = config["lookback_months"]

    def generate_windows(self, data_start: pd.Timestamp,
                         data_end: pd.Timestamp) -> List[WalkForwardWindow]:
        """Generate all walk-forward windows."""
        # Start from lookback_months before data_end
        period_start = data_end - pd.DateOffset(months=self.lookback_months)
        if period_start < data_start:
            period_start = data_start

        windows = []
        cursor = period_start
        window_id = 0

        total_days = self.train_days + self.test_days
        while cursor + timedelta(days=total_days) <= data_end:
            train_start = cursor
            train_end = cursor + timedelta(days=self.train_days)
            test_start = train_end
            test_end = test_start + timedelta(days=self.test_days)

            windows.append(WalkForwardWindow(
                window_id=window_id,
                train_start=str(train_start),
                train_end=str(train_end),
                test_start=str(test_start),
                test_end=str(test_end),
            ))

            cursor += timedelta(days=self.step_days)
            window_id += 1

        return windows


# ===========================================================================
# SIMPLE INFERENCE MODELS (Python-native, no MQ5 needed)
# ===========================================================================

class Conv1DModel:
    """
    Lightweight Conv1D signal model matching the spare champions architecture.
    input_size=8, hidden_size=128.
    Runs inference only -- no training here.
    """

    def __init__(self, input_size: int = 8, hidden_size: int = 128):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self._state_dict = None
        self._model = None

    def load_pth(self, path: str) -> bool:
        """Load a .pth state dict."""
        if torch is None:
            logger.warning("PyTorch not available, cannot load .pth")
            return False
        try:
            self._state_dict = torch.load(
                path, map_location="cpu", weights_only=False)
            return True
        except Exception as e:
            logger.warning(f"Failed to load {path}: {e}")
            return False

    def get_state_dict(self) -> Optional[dict]:
        return self._state_dict

    def set_state_dict(self, sd: dict):
        self._state_dict = sd

    def predict(self, features: np.ndarray) -> Tuple[int, float]:
        """
        Run inference on feature array.
        Returns (direction, confidence): direction in {-1, 0, 1}, confidence [0,1]

        For .pth models: uses the stored state dict for a forward pass.
        Falls back to a simple weight-matrix multiplication if the architecture
        is not a standard PyTorch module.
        """
        if self._state_dict is None:
            return (0, 0.0)

        try:
            # Try direct matrix multiplication with stored weights
            if isinstance(self._state_dict, dict):
                return self._predict_from_dict(features)
            else:
                return (0, 0.0)
        except Exception:
            return (0, 0.0)

    def _predict_from_dict(self, features: np.ndarray) -> Tuple[int, float]:
        """
        Predict using raw state dict weights via matrix multiplication.
        Handles both PyTorch state dicts and raw numpy weight matrices.
        """
        if torch is None:
            return (0, 0.0)

        sd = self._state_dict

        # If this is a PyTorch state dict with named layers
        if any(isinstance(v, torch.Tensor) for v in sd.values()):
            return self._predict_torch(features)

        # If this is a raw weight dict (from JSON format)
        return self._predict_raw(features)

    def _predict_torch(self, features: np.ndarray) -> Tuple[int, float]:
        """Forward pass through PyTorch state dict layers."""
        # Take last N bars of features as input
        lookback = min(50, len(features))
        x = features[-lookback:]

        # Flatten to 1D input
        x_flat = x.flatten()
        x_tensor = torch.tensor(x_flat, dtype=torch.float32).unsqueeze(0)

        # Simple linear pass through available weight layers
        output = x_tensor
        layer_keys = sorted([k for k in self._state_dict.keys()
                            if "weight" in k.lower()])

        for key in layer_keys[:3]:  # Max 3 layers to keep it fast
            w = self._state_dict[key]
            if isinstance(w, torch.Tensor) and w.dim() >= 2:
                # Reshape input to match weight matrix
                in_features = w.shape[-1]
                if output.shape[-1] != in_features:
                    # Adaptive reshape: truncate or pad
                    if output.shape[-1] > in_features:
                        output = output[:, :in_features]
                    else:
                        pad = torch.zeros(1, in_features - output.shape[-1])
                        output = torch.cat([output, pad], dim=-1)

                if w.dim() == 2:
                    output = torch.mm(output, w.t())
                elif w.dim() == 1:
                    output = output * w

                # ReLU activation between layers
                output = torch.relu(output)

                # Apply bias if exists
                bias_key = key.replace("weight", "bias")
                if bias_key in self._state_dict:
                    b = self._state_dict[bias_key]
                    if isinstance(b, torch.Tensor):
                        if output.shape[-1] == b.shape[0]:
                            output = output + b

        # Final output -> signal
        val = output.mean().item()
        confidence = min(1.0, abs(val))

        if val > 0.05:
            return (1, confidence)
        elif val < -0.05:
            return (-1, confidence)
        else:
            return (0, confidence)

    def _predict_raw(self, features: np.ndarray) -> Tuple[int, float]:
        """Predict using raw numpy weight matrices (from JSON experts)."""
        lookback = min(50, len(features))
        x = features[-lookback:].flatten()

        for key in ["input_weights", "hidden_weights", "output_weights"]:
            if key not in self._state_dict:
                continue
            w = np.array(self._state_dict[key])
            if w.ndim < 2:
                continue
            in_size = w.shape[1] if w.ndim == 2 else w.shape[0]
            if len(x) > in_size:
                x = x[:in_size]
            elif len(x) < in_size:
                x = np.pad(x, (0, in_size - len(x)))
            x = x @ w.T if w.ndim == 2 else x * w
            x = np.maximum(x, 0)  # ReLU

        val = np.mean(x)
        confidence = min(1.0, abs(float(val)))
        if val > 0.05:
            return (1, confidence)
        elif val < -0.05:
            return (-1, confidence)
        return (0, confidence)


class JSONWeightModel:
    """Model that loads weights from JSON config files."""

    def __init__(self):
        self._weights = None

    def load_json(self, path: str) -> bool:
        try:
            with open(path, "r") as f:
                self._weights = json.load(f)
            return True
        except Exception as e:
            logger.warning(f"Failed to load JSON weights from {path}: {e}")
            return False

    def get_weights(self) -> Optional[dict]:
        return self._weights

    def set_weights(self, w: dict):
        self._weights = w

    def predict(self, features: np.ndarray) -> Tuple[int, float]:
        if self._weights is None:
            return (0, 0.0)
        model = Conv1DModel()
        model.set_state_dict(self._weights)
        return model.predict(features)


# ===========================================================================
# MT5 MOCK ENGINE (adapted from extinction_trainer.py)
# ===========================================================================

class _SymbolInfo:
    """Mock MT5 symbol info."""
    def __init__(self, symbol):
        self.name = symbol
        if "BTC" in symbol:
            self.digits = 2
            self.trade_tick_value = 1.0
            self.trade_tick_size = 0.01
            self.volume_min = 0.01
            self.volume_max = 100.0
            self.volume_step = 0.01
            self.point = 0.01
        elif "ETH" in symbol:
            self.digits = 2
            self.trade_tick_value = 1.0
            self.trade_tick_size = 0.01
            self.volume_min = 0.01
            self.volume_max = 100.0
            self.volume_step = 0.01
            self.point = 0.01
        elif "XAU" in symbol:
            self.digits = 2
            self.trade_tick_value = 1.0
            self.trade_tick_size = 0.01
            self.volume_min = 0.01
            self.volume_max = 100.0
            self.volume_step = 0.01
            self.point = 0.01
        else:
            self.digits = 5
            self.trade_tick_value = 1.0
            self.trade_tick_size = 0.00001
            self.volume_min = 0.01
            self.volume_max = 100.0
            self.volume_step = 0.01
            self.point = 0.00001


class _TickInfo:
    """Mock MT5 tick."""
    def __init__(self, bid, ask):
        self.bid = bid
        self.ask = ask
        self.last = bid
        self.time = int(time.time())


class _AccountInfo:
    """Mock MT5 account info."""
    def __init__(self):
        self.login = 999999
        self.balance = 100000.0
        self.equity = 100000.0
        self.margin = 0.0
        self.margin_free = 100000.0


class MT5DataEngine:
    """
    Core data engine that replaces MetaTrader5 module for backtest.
    Loads data from the DataLayer and serves it bar-by-bar to experts.
    Adapted from extinction_trainer.py to share data with DataLayer.
    """

    # MT5 timeframe constants
    TIMEFRAME_M1 = 1
    TIMEFRAME_M5 = 5
    TIMEFRAME_M15 = 15
    TIMEFRAME_M30 = 30
    TIMEFRAME_H1 = 60
    TIMEFRAME_H4 = 240
    TIMEFRAME_D1 = 1440
    TIMEFRAME_W1 = 10080

    # Order type constants
    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1
    ORDER_TIME_GTC = 0
    ORDER_FILLING_IOC = 1
    TRADE_ACTION_DEAL = 1
    TRADE_ACTION_SLTP = 6
    TRADE_RETCODE_DONE = 10009

    def __init__(self):
        self._data: Dict[str, pd.DataFrame] = {}
        self._current_bar_idx: Dict[str, int] = {}
        self._symbol_map: Dict[str, str] = {}
        self._initialized = False
        self._resampled_cache: Dict[str, Dict[str, pd.DataFrame]] = {}

    def load_from_data_layer(self, data_layer: "DataLayer"):
        """
        Import data from the existing DataLayer so both systems share
        the same underlying data. Converts DataLayer format to what
        the MT5 mock expects (adds 'time' as unix seconds, etc).
        """
        for symbol, df in data_layer._data.items():
            mock_df = df.copy()
            # MT5 expects 'time' as unix timestamp (seconds)
            mock_df["time"] = mock_df["timestamp"].astype(np.int64) // 10**9
            # Ensure required columns exist
            if "tick_volume" not in mock_df.columns:
                mock_df["tick_volume"] = 0
            if "spread" not in mock_df.columns:
                mock_df["spread"] = 0
            if "real_volume" not in mock_df.columns:
                mock_df["real_volume"] = 0

            self._data[symbol] = mock_df
            self._current_bar_idx[symbol] = 0

            # Map common symbol variants (BTCUSDT <-> BTCUSD, etc)
            for variant in [symbol,
                            symbol.replace("USDT", "USD"),
                            symbol.replace("USD", "USDT"),
                            symbol.replace("USDT", "")]:
                self._symbol_map[variant] = symbol

    def _resolve_symbol(self, symbol: str) -> str:
        """Map symbol variants to canonical names."""
        if symbol in self._data:
            return symbol
        mapped = self._symbol_map.get(symbol)
        if mapped and mapped in self._data:
            return mapped
        for canonical in self._data:
            if canonical.replace("USDT", "") == symbol.replace("USD", "").replace("USDT", ""):
                self._symbol_map[symbol] = canonical
                return canonical
        return symbol

    def set_bar_index(self, symbol: str, idx: int):
        """Set the current bar position for a symbol."""
        canonical = self._resolve_symbol(symbol)
        if canonical in self._data:
            self._current_bar_idx[canonical] = min(
                idx, len(self._data[canonical]) - 1)

    def get_bar_index(self, symbol: str) -> int:
        canonical = self._resolve_symbol(symbol)
        return self._current_bar_idx.get(canonical, 0)

    def _resample_data(self, symbol: str, target_tf_minutes: int) -> pd.DataFrame:
        """Resample data to higher timeframes."""
        canonical = self._resolve_symbol(symbol)
        df = self._data.get(canonical)
        if df is None:
            return pd.DataFrame()

        cache_key = f"{canonical}_{target_tf_minutes}"
        if canonical in self._resampled_cache:
            if cache_key in self._resampled_cache[canonical]:
                return self._resampled_cache[canonical][cache_key]

        df_ts = df.set_index("timestamp")
        rule = f"{target_tf_minutes}min"
        resampled = df_ts.resample(rule).agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "tick_volume": "sum",
            "time": "first", "spread": "first", "real_volume": "sum"
        }).dropna()
        resampled = resampled.reset_index()

        if canonical not in self._resampled_cache:
            self._resampled_cache[canonical] = {}
        self._resampled_cache[canonical][cache_key] = resampled
        return resampled

    # --- MT5 API Mimics ---
    def initialize(self):
        self._initialized = True
        return True

    def shutdown(self):
        self._initialized = False

    def last_error(self):
        return (0, "No error")

    def account_info(self):
        return _AccountInfo()

    def symbol_info(self, symbol: str):
        return _SymbolInfo(symbol)

    def symbol_info_tick(self, symbol: str):
        canonical = self._resolve_symbol(symbol)
        if canonical not in self._data:
            return None
        idx = self._current_bar_idx.get(canonical, 0)
        row = self._data[canonical].iloc[idx]
        close = float(row["close"])
        spread = close * 0.0001
        return _TickInfo(close, close + spread)

    def copy_rates_from_pos(self, symbol: str, timeframe: int,
                            start_pos: int, count: int):
        """
        Return the last `count` bars up to (and including) the current bar.
        Handles timeframe resampling for higher timeframes.
        """
        canonical = self._resolve_symbol(symbol)
        if canonical not in self._data:
            return None

        current_idx = self._current_bar_idx.get(canonical, 0)

        # Determine which dataframe based on timeframe
        tf_map = {
            1: None, 5: None,  # Use base data directly
            15: 15, 30: 30, 60: 60, 240: 240, 1440: 1440, 10080: 10080,
        }
        resample_tf = tf_map.get(timeframe)

        if resample_tf is not None:
            df = self._resample_data(canonical, resample_tf)
            # Find the bar in resampled data that corresponds to current timestamp
            if len(df) == 0:
                return None
            current_ts = self._data[canonical]["timestamp"].iloc[current_idx]
            mask = df["timestamp"] <= current_ts
            current_idx = mask.sum() - 1
            if current_idx < 0:
                return None
        else:
            df = self._data[canonical]

        # Slice: last `count` bars ending at current_idx
        start = max(0, current_idx - count + 1)
        end = current_idx + 1
        subset = df.iloc[start:end]

        if len(subset) == 0:
            return None

        # Return as structured numpy array (what MT5 returns)
        cols = ["time", "open", "high", "low", "close",
                "tick_volume", "spread", "real_volume"]
        available_cols = [c for c in cols if c in subset.columns]
        records = subset[available_cols].to_records(index=False)
        return records

    def positions_get(self, **kwargs):
        """No real positions in backtest."""
        return None

    def order_send(self, request):
        """Mock order send - always returns None (we handle trades ourselves)."""
        return None

    def history_deals_get(self, *args, **kwargs):
        return None


# Global MT5 mock engine instance (shared by all Python experts)
_mt5_engine = MT5DataEngine()
_mt5_mock_installed = False


def _install_mt5_mock():
    """
    Replace the MetaTrader5 module in sys.modules with our mock engine.
    Must be called BEFORE any expert module imports.
    """
    global _mt5_mock_installed
    if _mt5_mock_installed:
        return

    mock_module = types.ModuleType("MetaTrader5")

    # Copy all constants
    mock_module.TIMEFRAME_M1 = MT5DataEngine.TIMEFRAME_M1
    mock_module.TIMEFRAME_M5 = MT5DataEngine.TIMEFRAME_M5
    mock_module.TIMEFRAME_M15 = MT5DataEngine.TIMEFRAME_M15
    mock_module.TIMEFRAME_M30 = MT5DataEngine.TIMEFRAME_M30
    mock_module.TIMEFRAME_H1 = MT5DataEngine.TIMEFRAME_H1
    mock_module.TIMEFRAME_H4 = MT5DataEngine.TIMEFRAME_H4
    mock_module.TIMEFRAME_D1 = MT5DataEngine.TIMEFRAME_D1
    mock_module.TIMEFRAME_W1 = MT5DataEngine.TIMEFRAME_W1
    mock_module.ORDER_TYPE_BUY = MT5DataEngine.ORDER_TYPE_BUY
    mock_module.ORDER_TYPE_SELL = MT5DataEngine.ORDER_TYPE_SELL
    mock_module.ORDER_TIME_GTC = MT5DataEngine.ORDER_TIME_GTC
    mock_module.ORDER_FILLING_IOC = MT5DataEngine.ORDER_FILLING_IOC
    mock_module.TRADE_ACTION_DEAL = MT5DataEngine.TRADE_ACTION_DEAL
    mock_module.TRADE_ACTION_SLTP = MT5DataEngine.TRADE_ACTION_SLTP
    mock_module.TRADE_RETCODE_DONE = MT5DataEngine.TRADE_RETCODE_DONE

    # Bind all functions to the global engine
    mock_module.initialize = _mt5_engine.initialize
    mock_module.shutdown = _mt5_engine.shutdown
    mock_module.last_error = _mt5_engine.last_error
    mock_module.account_info = _mt5_engine.account_info
    mock_module.symbol_info = _mt5_engine.symbol_info
    mock_module.symbol_info_tick = _mt5_engine.symbol_info_tick
    mock_module.copy_rates_from_pos = _mt5_engine.copy_rates_from_pos
    mock_module.positions_get = _mt5_engine.positions_get
    mock_module.order_send = _mt5_engine.order_send
    mock_module.history_deals_get = _mt5_engine.history_deals_get

    sys.modules["MetaTrader5"] = mock_module
    _mt5_mock_installed = True
    logger.info("MT5 mock module installed in sys.modules")


# ===========================================================================
# EXPERT BASE PATCH (adapted from extinction_trainer.py)
# ===========================================================================

_expert_base_patched = False


def _patch_expert_base():
    """
    Patch ExpertBase.__init__ to work in backtest mode:
    1. Accept 'version' kwarg (experts 15-24 pass it)
    2. Make 'name' and 'magic' optional with defaults
    3. Use temp SQLite DB instead of real adaptive DB
    Must be called AFTER _install_mt5_mock() and BEFORE loading experts.
    """
    global _expert_base_patched
    if _expert_base_patched:
        return

    from expert_base import ExpertBase

    def _patched_init(self, name: str = "Unknown", magic: int = 0,
                      symbols=None, timeframe: str = "M5",
                      max_loss_dollars: float = 1.00,
                      initial_sl_dollars: float = 0.60,
                      tp_multiplier: float = 3.0,
                      rolling_sl_divider: float = 1.5,
                      dynamic_tp_pct: int = 50,
                      confidence_threshold: float = 0.70,
                      daily_dd_limit_pct: float = 4.5,
                      max_dd_limit_pct: float = 9.0,
                      db_dir=None,
                      version: str = "1.0",
                      **extra_kwargs):

        if name == "Unknown":
            name = getattr(self.__class__, "EXPERT_NAME", None) or \
                   getattr(self.__class__, "NAME", None) or \
                   self.__class__.__name__
        if magic == 0:
            magic = getattr(self.__class__, "MAGIC_NUMBER", None) or \
                    getattr(self.__class__, "MAGIC", None) or \
                    random.randint(100000, 999999)

        if symbols is None:
            symbols = ["BTCUSD", "ETHUSD"]

        self.name = name
        self.magic = magic
        self.symbols = symbols
        self.timeframe_str = timeframe
        tf_map = {"M1": 1, "M5": 5, "M15": 15, "M30": 30,
                  "H1": 60, "H4": 240, "D1": 1440, "W1": 10080}
        self.timeframe = tf_map.get(timeframe, 5)

        self.max_loss_dollars = max_loss_dollars
        self.initial_sl_dollars = initial_sl_dollars
        self.tp_multiplier = tp_multiplier
        self.rolling_sl_divider = rolling_sl_divider
        self.dynamic_tp_pct = dynamic_tp_pct
        self.confidence_threshold = confidence_threshold
        self.daily_dd_limit_pct = daily_dd_limit_pct
        self.max_dd_limit_pct = max_dd_limit_pct
        self.daily_start_balance = 100000.0
        self.peak_balance = 100000.0
        self.last_dd_reset_day = None
        self.tracks = {}

        # Temp directory for adaptive DB (avoid clobbering real DBs)
        _bt_db_dir = os.path.join(tempfile.gettempdir(), "jtear_expert_dbs")
        os.makedirs(_bt_db_dir, exist_ok=True)
        self.db_path = os.path.join(_bt_db_dir, f"adaptive_{name}_{magic}.db")
        self._init_adaptive_db()

        self.logger = logging.getLogger(f"QC.{name}")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                "%(asctime)s [%(name)s] %(levelname)s: %(message)s"))
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.ERROR)  # Quiet during backtest

        self._entropy_cache = {}
        self._running = False
        self.tick_interval = 1.0

    ExpertBase.__init__ = _patched_init
    _expert_base_patched = True
    logger.info("ExpertBase patched for backtest mode")


# ===========================================================================
# EXPERT REGISTRY (all 24 Python experts)
# ===========================================================================

EXPERT_REGISTRY = {
    "expert_01_foundation": "FoundationExpert",
    "expert_02_jardines_gate": "JardinesGateExpert",
    "expert_03_etare": "ETAREExpert",
    "expert_04_teqa": "TEQAExpert",
    "expert_05_vdj": "VDJExpert",
    "expert_06_te_domestication": "TEDomesticationExpert",
    "expert_07_crispr": "CRISPRCasExpert",
    "expert_08_electric_organs": "ElectricOrgansExpert",
    "expert_09_protective_deletion": "ProtectiveDeletionExpert",
    "expert_10_toxoplasma": "ToxoplasmaExpert",
    "expert_11_bdelloid": "BdelloidExpert",
    "expert_12_syncytin": "SyncytinExpert",
    "expert_13_korv": "KorvExpert",
    "expert_14_cancer_cell": "CancerCellExpert",
    "expert_15_qnif": "Expert15QNIF",
    "expert_16_hgh": "Expert16HGH",
    "expert_17_stanozolol": "Expert17Stanozolol",
    "expert_18_testosterone": "Expert18Testosterone",
    "expert_19_mushroom_rabies": "Expert19MushroomRabies",
    "expert_20_blueguardian_quantum": "Expert20BlueGuardianQuantum",
    "expert_21_blueguardian_elite": "Expert21BlueGuardianElite",
    "expert_22_blueguardian_dynamic": "Expert22BlueGuardianDynamic",
    "expert_23_nociception": "Expert23Nociception",
    "expert_24_quantum_teenager": "Expert24QuantumTeenager",
}

# Experts 01-14 need symbols kwarg; 15-24 take no constructor args
EXPERTS_WITH_SYMBOLS = {
    "expert_01_foundation", "expert_02_jardines_gate",
    "expert_03_etare", "expert_04_teqa",
    "expert_05_vdj", "expert_06_te_domestication",
    "expert_07_crispr", "expert_08_electric_organs",
    "expert_09_protective_deletion", "expert_10_toxoplasma",
    "expert_11_bdelloid", "expert_12_syncytin",
    "expert_13_korv", "expert_14_cancer_cell",
}


def _load_python_expert(module_name: str, class_name: str,
                        expert_dir: str,
                        symbols: List[str]) -> Optional[Any]:
    """
    Import and instantiate a single Python expert from #1 rankers.
    Returns the expert instance or None on failure.
    """
    try:
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)

        if module_name in EXPERTS_WITH_SYMBOLS:
            instance = cls(symbols=symbols, confidence_threshold=0.50)
        else:
            instance = cls()

        # Force low confidence threshold for backtesting
        if hasattr(instance, "confidence_threshold"):
            instance.confidence_threshold = 0.50
        if hasattr(instance, "CONFIDENCE_THRESH"):
            instance.CONFIDENCE_THRESH = 0.20

        return instance
    except Exception as e:
        logger.warning(f"Failed to load {module_name}.{class_name}: {e}")
        return None


# ===========================================================================
# PYTHON EXPERT MODEL (wrapper for BacktestEngine compatibility)
# ===========================================================================

class PythonExpertModel:
    """
    Wraps a Python ExpertBase subclass to provide the same predict()
    interface as Conv1DModel and JSONWeightModel.

    The MT5 mock's bar index must be set by the caller before predict().
    """

    def __init__(self, expert_instance, mt5_engine: MT5DataEngine,
                 symbol: str):
        self.expert = expert_instance
        self.engine = mt5_engine
        self.symbol = symbol

    def predict(self, features_up_to_idx) -> Tuple[int, float]:
        """
        Call the expert's generate_signal() method.
        The MT5 engine's bar index must already be set by the caller.
        Returns (direction, confidence).
        """
        try:
            result = self.expert.generate_signal(self.symbol)
            if result is None:
                return (0, 0.0)
            direction, confidence, _fingerprint = result
            return (direction, confidence)
        except Exception:
            return (0, 0.0)


# ===========================================================================
# BACKTEST ENGINE (simplified from extinction_trainer.py)
# ===========================================================================

class BacktestEngine:
    """
    Bar-by-bar trade simulation for evaluating experts.
    Uses pre-cached numpy arrays for speed.
    """

    FIXED_RISK_DOLLARS = 1.00
    TP_MULTIPLIER = 3.0

    def __init__(self, data_layer: DataLayer, config: dict):
        self._data = data_layer
        self._config = config
        self._ticket_counter = 1000

    def _next_ticket(self) -> int:
        self._ticket_counter += 1
        return self._ticket_counter

    def evaluate_expert(self, expert_entry: ExpertEntry,
                        symbol: str,
                        test_start: str, test_end: str,
                        features: Optional[np.ndarray] = None
                        ) -> ExpertPerformance:
        """
        Evaluate an expert on a test window.
        For PTH/JSON experts, uses direct Python inference.
        Returns performance metrics.
        """
        perf = ExpertPerformance(
            expert_name=expert_entry.name,
            window_id=0,
        )

        np_data = self._data.get_numpy(symbol)
        if np_data is None:
            return perf

        start_idx, end_idx = self._data.get_bar_indices(
            symbol, test_start, test_end)
        if start_idx >= end_idx:
            return perf

        # Load the model
        model = self._load_model(expert_entry)
        if model is None:
            return perf

        # For Python experts, set the correct symbol on the model
        is_python_expert = isinstance(model, PythonExpertModel)
        if is_python_expert:
            model.symbol = self._resolve_expert_symbol(symbol)

        # Compute features if not provided (Python experts don't need features
        # but the simulation loop still needs price data and ATR)
        if features is None and not is_python_expert:
            features = self._data.compute_features(symbol, start_idx, end_idx)
        if features is None and not is_python_expert:
            return perf

        # Extract price arrays for SL/TP checking
        highs = np_data["high"][start_idx:end_idx + 1]
        lows = np_data["low"][start_idx:end_idx + 1]
        closes = np_data["close"][start_idx:end_idx + 1]
        timestamps = np_data["timestamp"][start_idx:end_idx + 1]

        # ATR for position sizing
        atr = DataLayer._compute_atr(highs, lows, closes, 14)

        # Simulate
        skip = self._config["skip_bars"]
        stride = self._config["signal_stride"]
        n_bars = len(closes)
        open_trade: Optional[SimulatedTrade] = None
        idx = skip
        wall_start = time.time()
        timeout = self._config["expert_timeout"]

        while idx < n_bars:
            # Wall clock timeout
            if time.time() - wall_start > timeout:
                break

            if open_trade is not None:
                # Check SL/TP
                scan_end = min(idx + stride, n_bars)
                trade_closed = False
                for si in range(idx, scan_end):
                    h, l = highs[si], lows[si]
                    if open_trade.direction == 1:  # BUY
                        if l <= open_trade.sl_price:
                            self._close_trade(open_trade, open_trade.sl_price,
                                              str(timestamps[si]), "LOSS")
                            perf.record_trade(open_trade)
                            open_trade = None
                            idx = si + 1
                            trade_closed = True
                            break
                        if h >= open_trade.tp_price:
                            self._close_trade(open_trade, open_trade.tp_price,
                                              str(timestamps[si]), "WIN")
                            perf.record_trade(open_trade)
                            open_trade = None
                            idx = si + 1
                            trade_closed = True
                            break
                    else:  # SELL
                        if h >= open_trade.sl_price:
                            self._close_trade(open_trade, open_trade.sl_price,
                                              str(timestamps[si]), "LOSS")
                            perf.record_trade(open_trade)
                            open_trade = None
                            idx = si + 1
                            trade_closed = True
                            break
                        if l <= open_trade.tp_price:
                            self._close_trade(open_trade, open_trade.tp_price,
                                              str(timestamps[si]), "WIN")
                            perf.record_trade(open_trade)
                            open_trade = None
                            idx = si + 1
                            trade_closed = True
                            break

                if trade_closed:
                    continue

                # Check for opposing signal
                can_predict = is_python_expert or idx < len(features)
                if can_predict:
                    if is_python_expert:
                        model.engine.set_bar_index(
                            model.symbol, start_idx + idx)
                    direction, confidence = model.predict(
                        features[:idx] if not is_python_expert else None)
                    if direction != 0 and direction != open_trade.direction:
                        if confidence >= 0.50:
                            ci = min(scan_end - 1, n_bars - 1)
                            self._close_trade(
                                open_trade, closes[ci],
                                str(timestamps[ci]), "SIGNAL_CLOSE")
                            perf.record_trade(open_trade)
                            open_trade = None
                idx = scan_end

            else:
                # No open trade -- generate signal
                can_predict = is_python_expert or idx < len(features)
                if can_predict:
                    if is_python_expert:
                        model.engine.set_bar_index(
                            model.symbol, start_idx + idx)
                    direction, confidence = model.predict(
                        features[:idx] if not is_python_expert else None)

                    if direction != 0 and confidence >= 0.50:
                        atr_val = atr[idx] if idx < len(atr) else closes[idx] * 0.002
                        if atr_val == 0 or np.isnan(atr_val):
                            atr_val = closes[idx] * 0.002

                        sl_dist = max(atr_val * 1.5, closes[idx] * 0.0005)
                        entry = closes[idx]

                        if direction == 1:
                            sl = entry - sl_dist
                            tp = entry + sl_dist * self.TP_MULTIPLIER
                        else:
                            sl = entry + sl_dist
                            tp = entry - sl_dist * self.TP_MULTIPLIER

                        volume = self.FIXED_RISK_DOLLARS / (sl_dist + 1e-10)

                        open_trade = SimulatedTrade(
                            ticket=self._next_ticket(),
                            symbol=symbol,
                            direction=direction,
                            entry_price=entry,
                            entry_time=str(timestamps[idx]),
                            sl_price=sl,
                            tp_price=tp,
                            sl_distance=sl_dist,
                            volume=volume,
                        )

                idx += stride

        # Close remaining trade at window end
        if open_trade is not None and n_bars > 0:
            self._close_trade(open_trade, closes[-1],
                              str(timestamps[-1]), "WINDOW_END")
            perf.record_trade(open_trade)

        return perf

    def _load_model(self, entry: ExpertEntry):
        """Load the appropriate model for an expert entry."""
        if entry.expert_type == ExpertType.PTH_CONV1D or \
           entry.expert_type == ExpertType.PTH_ARMY:
            model = Conv1DModel()
            if entry.weight_path and model.load_pth(entry.weight_path):
                return model
            return None

        elif entry.expert_type == ExpertType.JSON_WEIGHTS:
            model = JSONWeightModel()
            if entry.weight_path and model.load_json(entry.weight_path):
                return model
            return None

        elif entry.expert_type == ExpertType.MQ5_SYSTEM:
            # MQ5 systems need the full MT5 mock pipeline.
            # For now, return None -- these require the extinction_trainer
            # infrastructure to evaluate. Future: integrate MT5 mock here.
            logger.debug(f"MQ5 system {entry.name} skipped (not yet integrated)")
            return None

        elif entry.expert_type == ExpertType.PYTHON_EXPERT:
            # Python ExpertBase subclass from #1 rankers
            return self._load_python_expert_model(entry)

        return None

    def _load_python_expert_model(self, entry: ExpertEntry
                                  ) -> Optional[PythonExpertModel]:
        """
        Load a Python expert from #1 rankers and wrap it in PythonExpertModel.
        Uses the cached instance from entry._expert_instance if available.
        """
        # Check if we already have a cached instance
        instance = getattr(entry, "_expert_instance", None)
        if instance is not None:
            # Return a model for the first symbol (caller handles multi-symbol)
            symbols = self._config.get("symbols", ["BTCUSDT"])
            # Resolve to a symbol the expert can handle
            sym = self._resolve_expert_symbol(symbols[0] if symbols else "BTCUSDT")
            return PythonExpertModel(instance, _mt5_engine, sym)

        # Need to load the expert module
        module_name = entry.source_path  # e.g., "expert_01_foundation"
        class_name = EXPERT_REGISTRY.get(module_name)
        if class_name is None:
            logger.warning(f"No registry entry for {module_name}")
            return None

        # Ensure MT5 mock and ExpertBase patch are in place
        _install_mt5_mock()
        _mt5_engine.load_from_data_layer(self._data)

        rankers_dir = os.path.join(
            self._config["qt_library"], "#1 rankers")
        spare_dir = self._config["spare_champions"]

        # Ensure both directories are on sys.path
        if rankers_dir not in sys.path:
            sys.path.insert(0, rankers_dir)
        if spare_dir not in sys.path:
            sys.path.insert(0, spare_dir)

        _patch_expert_base()

        symbols_for_expert = ["BTCUSD", "ETHUSD", "XAUUSD"]
        instance = _load_python_expert(
            module_name, class_name, rankers_dir, symbols_for_expert)
        if instance is None:
            return None

        # Cache on the entry for reuse across symbols
        entry._expert_instance = instance

        sym = self._resolve_expert_symbol(
            self._config.get("symbols", ["BTCUSDT"])[0])
        return PythonExpertModel(instance, _mt5_engine, sym)

    @staticmethod
    def _resolve_expert_symbol(symbol: str) -> str:
        """Convert J-TEAR symbol names to what experts expect (no T suffix)."""
        # Experts typically use BTCUSD, ETHUSD, XAUUSD (without T)
        return symbol.replace("USDT", "USD")

    def _close_trade(self, trade: SimulatedTrade, exit_price: float,
                     exit_time: str, status: str):
        trade.exit_price = exit_price
        trade.exit_time = exit_time
        trade.status = status
        if trade.direction == 1:
            price_diff = exit_price - trade.entry_price
        else:
            price_diff = trade.entry_price - exit_price
        trade.pnl = price_diff * trade.volume


# ===========================================================================
# SCORER
# ===========================================================================

class Scorer:
    """Computes composite scores for experts."""

    def __init__(self, config: dict):
        self.weights = config["scoring_weights"]
        self.min_trades = config["min_trades_threshold"]

    def compute_composite(self, perf: ExpertPerformance,
                          score_history: List[float]) -> float:
        """Compute weighted composite score."""
        if perf.total_trades < self.min_trades:
            return 0.0

        # Raw metrics (not yet normalized -- normalization happens in batch)
        pf = min(5.0, perf.profit_factor)
        wr = perf.win_rate
        sharpe = perf.sharpe
        dd = perf.max_drawdown

        # Consistency: average of last 5 scores
        recent = score_history[-5:] if score_history else []
        consistency = np.mean(recent) if recent else 0.0

        return (
            self.weights["profit_factor"] * (pf / 5.0) +
            self.weights["win_rate"] * wr +
            self.weights["sharpe"] * min(1.0, max(0.0, (sharpe + 2.0) / 4.0)) +
            self.weights["max_drawdown"] * (1.0 - min(1.0, dd / 10.0)) +
            self.weights["consistency"] * consistency
        )

    def rank_population(self, scores: Dict[str, float]) -> List[Tuple[str, float]]:
        """Rank experts by composite score, descending."""
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ===========================================================================
# BREEDER
# ===========================================================================

class Breeder:
    """Handles all breeding operations."""

    def __init__(self, config: dict, workspace: str):
        self.config = config
        self.workspace = workspace
        self.offspring_dir = os.path.join(workspace, "offspring")
        os.makedirs(self.offspring_dir, exist_ok=True)
        self._child_counter = 0

    def select_operator(self) -> BreedingOp:
        """Weighted random selection of breeding operator."""
        probs = self.config["breeding_probabilities"]
        ops = list(probs.keys())
        weights = list(probs.values())
        chosen = random.choices(ops, weights=weights, k=1)[0]
        return BreedingOp(chosen)

    def breed(self, operator: BreedingOp, parents: List[ExpertEntry],
              window_id: int) -> Optional[ExpertEntry]:
        """
        Execute a breeding operation.
        Returns a new ExpertEntry or None if breeding fails.
        """
        if not parents:
            return None

        self._child_counter += 1
        child_id = f"w{window_id}_{operator.value}_{self._child_counter}"

        if operator == BreedingOp.MUTATION:
            return self._mutate(parents[0], child_id,
                                self.config["mutation_rate"])

        elif operator == BreedingOp.RADIATION:
            return self._mutate(parents[0], child_id,
                                self.config["radiation_rate"])

        elif operator == BreedingOp.ELITE_CLONE:
            return self._elite_clone(parents[0], child_id)

        elif operator == BreedingOp.CROSSOVER_SAME:
            if len(parents) >= 2:
                return self._crossover_same(parents[0], parents[1], child_id)
            return self._mutate(parents[0], child_id,
                                self.config["mutation_rate"])

        elif operator == BreedingOp.CROSSOVER_DIFF:
            # Cross-architecture: returns None (handled at roster level as
            # vote weight adjustment, not a new expert)
            return None

        elif operator == BreedingOp.BLEND:
            return self._blend(parents, child_id)

        elif operator == BreedingOp.GENESIS:
            return self._genesis(parents[0], child_id)

        return None

    def _mutate(self, parent: ExpertEntry, child_id: str,
                rate: float) -> Optional[ExpertEntry]:
        """Clone parent and mutate weights."""
        if parent.weight_path is None or not os.path.exists(parent.weight_path):
            return None

        child_name = f"{parent.architecture}_{child_id}"
        child_weight_path = os.path.join(
            self.offspring_dir, f"{child_name}.pth")

        if parent.expert_type in (ExpertType.PTH_CONV1D, ExpertType.PTH_ARMY):
            if torch is None:
                return None
            try:
                sd = torch.load(parent.weight_path, map_location="cpu",
                                weights_only=False)
                mutated = {}
                for key, tensor in sd.items():
                    if isinstance(tensor, torch.Tensor) and tensor.is_floating_point():
                        noise = torch.randn_like(tensor) * rate
                        mutated[key] = tensor + tensor * noise
                    else:
                        mutated[key] = tensor
                torch.save(mutated, child_weight_path)
            except Exception as e:
                logger.warning(f"Mutation failed for {parent.name}: {e}")
                return None

        elif parent.expert_type == ExpertType.JSON_WEIGHTS:
            try:
                with open(parent.weight_path, "r") as f:
                    weights = json.load(f)
                mutated = self._mutate_json_recursive(weights, rate)
                child_weight_path = child_weight_path.replace(".pth", ".json")
                with open(child_weight_path, "w") as f:
                    json.dump(mutated, f)
            except Exception as e:
                logger.warning(f"JSON mutation failed for {parent.name}: {e}")
                return None
        else:
            return None

        return ExpertEntry(
            name=child_name,
            architecture=parent.architecture,
            expert_type=parent.expert_type,
            source_path=parent.source_path,
            weight_path=child_weight_path,
            tier=Tier.MINOR,
            lineage=[parent.name],
            breeding_op=BreedingOp.MUTATION.value if rate <= 0.15 \
                else BreedingOp.RADIATION.value,
        )

    def _elite_clone(self, parent: ExpertEntry,
                     child_id: str) -> Optional[ExpertEntry]:
        """Direct copy of parent weights."""
        if parent.weight_path is None or not os.path.exists(parent.weight_path):
            return None

        child_name = f"{parent.architecture}_{child_id}"
        ext = ".json" if parent.expert_type == ExpertType.JSON_WEIGHTS else ".pth"
        child_weight_path = os.path.join(
            self.offspring_dir, f"{child_name}{ext}")

        try:
            shutil.copy2(parent.weight_path, child_weight_path)
        except Exception as e:
            logger.warning(f"Elite clone failed for {parent.name}: {e}")
            return None

        return ExpertEntry(
            name=child_name,
            architecture=parent.architecture,
            expert_type=parent.expert_type,
            source_path=parent.source_path,
            weight_path=child_weight_path,
            tier=Tier.MINOR,
            lineage=[parent.name],
            breeding_op=BreedingOp.ELITE_CLONE.value,
        )

    def _crossover_same(self, parent1: ExpertEntry, parent2: ExpertEntry,
                        child_id: str) -> Optional[ExpertEntry]:
        """Blend weights of two same-architecture experts."""
        if parent1.weight_path is None or parent2.weight_path is None:
            return None
        if not os.path.exists(parent1.weight_path) or \
           not os.path.exists(parent2.weight_path):
            return None

        child_name = f"{parent1.architecture}_{child_id}"
        child_weight_path = os.path.join(
            self.offspring_dir, f"{child_name}.pth")

        if parent1.expert_type in (ExpertType.PTH_CONV1D, ExpertType.PTH_ARMY):
            if torch is None:
                return None
            try:
                sd1 = torch.load(parent1.weight_path, map_location="cpu",
                                 weights_only=False)
                sd2 = torch.load(parent2.weight_path, map_location="cpu",
                                 weights_only=False)
                # Blend: 50/50 average, with slight random bias
                alpha = random.uniform(0.3, 0.7)
                blended = {}
                for key in sd1:
                    if key in sd2 and isinstance(sd1[key], torch.Tensor) \
                       and isinstance(sd2[key], torch.Tensor):
                        if sd1[key].shape == sd2[key].shape:
                            blended[key] = alpha * sd1[key] + (1 - alpha) * sd2[key]
                        else:
                            blended[key] = sd1[key]
                    else:
                        blended[key] = sd1[key]
                torch.save(blended, child_weight_path)
            except Exception as e:
                logger.warning(f"Crossover failed: {e}")
                return None

        elif parent1.expert_type == ExpertType.JSON_WEIGHTS:
            try:
                with open(parent1.weight_path, "r") as f:
                    w1 = json.load(f)
                with open(parent2.weight_path, "r") as f:
                    w2 = json.load(f)
                alpha = random.uniform(0.3, 0.7)
                blended = self._blend_json_recursive(w1, w2, alpha)
                child_weight_path = child_weight_path.replace(".pth", ".json")
                with open(child_weight_path, "w") as f:
                    json.dump(blended, f)
            except Exception as e:
                logger.warning(f"JSON crossover failed: {e}")
                return None
        else:
            return None

        return ExpertEntry(
            name=child_name,
            architecture=parent1.architecture,
            expert_type=parent1.expert_type,
            source_path=parent1.source_path,
            weight_path=child_weight_path,
            tier=Tier.MINOR,
            lineage=[parent1.name, parent2.name],
            breeding_op=BreedingOp.CROSSOVER_SAME.value,
        )

    def _blend(self, parents: List[ExpertEntry],
               child_id: str) -> Optional[ExpertEntry]:
        """Weighted average of multiple parents' weights."""
        # Filter to parents with loadable weights of same type
        valid = [p for p in parents
                 if p.weight_path and os.path.exists(p.weight_path)]
        if not valid:
            return None

        # Use the first parent as the architecture template
        template = valid[0]
        child_name = f"{template.architecture}_{child_id}"

        if template.expert_type in (ExpertType.PTH_CONV1D, ExpertType.PTH_ARMY):
            if torch is None:
                return None
            try:
                state_dicts = []
                for p in valid:
                    sd = torch.load(p.weight_path, map_location="cpu",
                                    weights_only=False)
                    state_dicts.append(sd)

                # Average all state dicts
                blended = {}
                keys = state_dicts[0].keys()
                for key in keys:
                    tensors = [sd[key] for sd in state_dicts
                               if key in sd and isinstance(sd[key], torch.Tensor)]
                    if tensors and all(t.shape == tensors[0].shape for t in tensors):
                        blended[key] = sum(tensors) / len(tensors)
                    else:
                        blended[key] = state_dicts[0][key]

                child_weight_path = os.path.join(
                    self.offspring_dir, f"{child_name}.pth")
                torch.save(blended, child_weight_path)

                return ExpertEntry(
                    name=child_name,
                    architecture=template.architecture,
                    expert_type=template.expert_type,
                    source_path=template.source_path,
                    weight_path=child_weight_path,
                    tier=Tier.MINOR,
                    lineage=[p.name for p in valid],
                    breeding_op=BreedingOp.BLEND.value,
                )
            except Exception as e:
                logger.warning(f"Blend failed: {e}")
                return None

        return None

    def _genesis(self, template: ExpertEntry,
                 child_id: str) -> Optional[ExpertEntry]:
        """Fresh random weight initialization using same architecture shape."""
        if template.weight_path is None or not os.path.exists(template.weight_path):
            return None

        child_name = f"{template.architecture}_{child_id}"

        if template.expert_type in (ExpertType.PTH_CONV1D, ExpertType.PTH_ARMY):
            if torch is None:
                return None
            try:
                sd = torch.load(template.weight_path, map_location="cpu",
                                weights_only=False)
                genesis = {}
                for key, tensor in sd.items():
                    if isinstance(tensor, torch.Tensor) and tensor.is_floating_point():
                        # Xavier-style initialization
                        std = 1.0 / math.sqrt(tensor.numel())
                        genesis[key] = torch.randn_like(tensor) * std
                    else:
                        genesis[key] = tensor

                child_weight_path = os.path.join(
                    self.offspring_dir, f"{child_name}.pth")
                torch.save(genesis, child_weight_path)

                return ExpertEntry(
                    name=child_name,
                    architecture=template.architecture,
                    expert_type=template.expert_type,
                    source_path=template.source_path,
                    weight_path=child_weight_path,
                    tier=Tier.MINOR,
                    lineage=[],
                    breeding_op=BreedingOp.GENESIS.value,
                )
            except Exception as e:
                logger.warning(f"Genesis failed: {e}")
                return None

        return None

    @staticmethod
    def _mutate_json_recursive(obj, rate: float):
        """Recursively mutate numeric values in a JSON structure."""
        if isinstance(obj, dict):
            return {k: Breeder._mutate_json_recursive(v, rate)
                    for k, v in obj.items()}
        elif isinstance(obj, list):
            return [Breeder._mutate_json_recursive(v, rate) for v in obj]
        elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
            factor = 1.0 + random.uniform(-rate, rate)
            return obj * factor
        return obj

    @staticmethod
    def _blend_json_recursive(obj1, obj2, alpha: float):
        """Recursively blend two JSON structures."""
        if isinstance(obj1, dict) and isinstance(obj2, dict):
            result = {}
            for k in obj1:
                if k in obj2:
                    result[k] = Breeder._blend_json_recursive(
                        obj1[k], obj2[k], alpha)
                else:
                    result[k] = obj1[k]
            return result
        elif isinstance(obj1, list) and isinstance(obj2, list):
            min_len = min(len(obj1), len(obj2))
            return [Breeder._blend_json_recursive(obj1[i], obj2[i], alpha)
                    for i in range(min_len)]
        elif isinstance(obj1, (int, float)) and isinstance(obj2, (int, float)):
            return alpha * obj1 + (1 - alpha) * obj2
        return obj1


# ===========================================================================
# ROSTER MANAGER
# ===========================================================================

class RosterManager:
    """Manages Active Roster and Minor League tiers."""

    def __init__(self, config: dict):
        self.config = config
        self.active_size = config["active_roster_size"]
        self.minor_size = config["minor_league_size"]
        self.max_minor = config["max_minor_league_size"]
        self.relegation_buffer = config["relegation_buffer_windows"]

        self.experts: Dict[str, ExpertEntry] = {}

    def add_expert(self, entry: ExpertEntry):
        """Add an expert to the system."""
        self.experts[entry.name] = entry

    def remove_expert(self, name: str):
        """Remove an expert (when minor league is pruned)."""
        if name in self.experts:
            del self.experts[name]

    def get_active(self) -> List[ExpertEntry]:
        """Get all active roster experts."""
        return [e for e in self.experts.values() if e.tier == Tier.ACTIVE]

    def get_minor(self) -> List[ExpertEntry]:
        """Get all minor league experts."""
        return [e for e in self.experts.values() if e.tier == Tier.MINOR]

    def get_all(self) -> List[ExpertEntry]:
        return list(self.experts.values())

    def initialize_roster(self, entries: List[ExpertEntry]):
        """
        Initialize from a flat list. Top N by composite_score go to active,
        rest go to minor.
        """
        # Sort by composite_score descending
        sorted_entries = sorted(entries, key=lambda e: e.composite_score,
                                reverse=True)

        for i, entry in enumerate(sorted_entries):
            if i < self.active_size:
                entry.tier = Tier.ACTIVE
                entry.vote_weight = 1.0 / self.active_size
            else:
                entry.tier = Tier.MINOR
                entry.vote_weight = 0.0
            self.experts[entry.name] = entry

    def update_scores(self, scores: Dict[str, float]):
        """Update composite scores for all experts."""
        for name, score in scores.items():
            if name in self.experts:
                self.experts[name].composite_score = score
                self.experts[name].score_history.append(score)

    def handle_promotions(self) -> List[Tuple[str, str]]:
        """
        Promote top minor leaguer, relegate worst active, if warranted.
        Returns list of (promoted_name, relegated_name) swaps.
        """
        active = self.get_active()
        minor = self.get_minor()

        if not active or not minor:
            return []

        # Sort active by score ascending (worst first)
        active_sorted = sorted(active, key=lambda e: e.composite_score)
        # Sort minor by score descending (best first)
        minor_sorted = sorted(minor, key=lambda e: e.composite_score,
                              reverse=True)

        swaps = []
        worst_active = active_sorted[0]
        best_minor = minor_sorted[0]

        # Check relegation buffer: worst active must have underperformed
        # for N consecutive windows
        if best_minor.composite_score > worst_active.composite_score:
            worst_active.consecutive_underperform += 1
        else:
            worst_active.consecutive_underperform = 0

        if worst_active.consecutive_underperform >= self.relegation_buffer:
            # SWAP
            worst_active.tier = Tier.MINOR
            worst_active.vote_weight = 0.0
            worst_active.consecutive_underperform = 0

            best_minor.tier = Tier.ACTIVE
            best_minor.consecutive_underperform = 0

            swaps.append((best_minor.name, worst_active.name))
            logger.info(f"  CALL-UP: {best_minor.name} replaces "
                        f"{worst_active.name}")
        elif best_minor.composite_score > worst_active.composite_score:
            logger.info(f"  {worst_active.name} underperforming "
                        f"({worst_active.consecutive_underperform}/"
                        f"{self.relegation_buffer} windows)")

        return swaps

    def update_vote_weights(self):
        """Update vote weights for active roster using softmax of scores."""
        active = self.get_active()
        if not active:
            return

        scores = [e.composite_score for e in active]
        max_score = max(scores) if scores else 0
        # Softmax with temperature
        exp_scores = [math.exp(s - max_score) for s in scores]
        total = sum(exp_scores)
        if total == 0:
            total = 1.0

        for entry, exp_s in zip(active, exp_scores):
            entry.vote_weight = exp_s / total

    def prune_minor_league(self) -> List[str]:
        """Remove worst minor league experts if over max size."""
        minor = self.get_minor()
        pruned = []

        if len(minor) > self.max_minor:
            # Sort by score ascending (worst first)
            minor_sorted = sorted(minor, key=lambda e: e.composite_score)
            n_prune = len(minor) - self.max_minor
            for entry in minor_sorted[:n_prune]:
                pruned.append(entry.name)
                self.remove_expert(entry.name)
                logger.info(f"  PRUNED from minor league: {entry.name}")

        return pruned

    def get_same_arch_experts(self, architecture: str) -> List[ExpertEntry]:
        """Get all experts of a specific architecture."""
        return [e for e in self.experts.values()
                if e.architecture == architecture]

    def to_dict(self) -> dict:
        return {
            name: entry.to_dict()
            for name, entry in self.experts.items()
        }

    def from_dict(self, d: dict):
        self.experts = {
            name: ExpertEntry.from_dict(entry_dict)
            for name, entry_dict in d.items()
        }


# ===========================================================================
# ENSEMBLE VOTER
# ===========================================================================

class EnsembleVoter:
    """Fuses signals from active roster experts."""

    def __init__(self, config: dict):
        self.threshold = config["ensemble_confidence_threshold"]

    def vote(self, signals: List[Tuple[ExpertEntry, int, float]]
             ) -> Tuple[int, float]:
        """
        Fuse signals from active roster.
        signals: list of (expert_entry, direction, confidence)
        Returns (ensemble_direction, ensemble_confidence).
        """
        if not signals:
            return (0, 0.0)

        weighted_sum = 0.0
        total_weight = 0.0

        for entry, direction, confidence in signals:
            w = entry.vote_weight * confidence
            weighted_sum += direction * w
            total_weight += abs(direction) * w

        if total_weight == 0:
            return (0, 0.0)

        ensemble_signal = weighted_sum / total_weight
        ensemble_confidence = abs(ensemble_signal)

        if ensemble_confidence >= self.threshold:
            if ensemble_signal > 0:
                return (1, ensemble_confidence)
            else:
                return (-1, ensemble_confidence)

        return (0, ensemble_confidence)


# ===========================================================================
# HISTORY DATABASE
# ===========================================================================

class HistoryDB:
    """SQLite database for tracking all J-TEAR history."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path)
        self._create_tables()

    def _create_tables(self):
        c = self._conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                window_id INTEGER,
                expert_name TEXT,
                architecture TEXT,
                symbol TEXT,
                total_trades INTEGER,
                wins INTEGER,
                losses INTEGER,
                profit_factor REAL,
                win_rate REAL,
                sharpe REAL,
                max_drawdown REAL,
                net_pnl REAL,
                composite_score REAL,
                tier TEXT,
                timestamp TEXT
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS breeding_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                window_id INTEGER,
                operator TEXT,
                parent_1 TEXT,
                parent_2 TEXT,
                child_name TEXT,
                mutation_rate REAL,
                timestamp TEXT
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS roster_changes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                window_id INTEGER,
                action TEXT,
                expert_promoted TEXT,
                expert_relegated TEXT,
                timestamp TEXT
            )
        """)
        self._conn.commit()

    def log_evaluation(self, window_id: int, entry: ExpertEntry,
                       perf: ExpertPerformance, symbol: str):
        c = self._conn.cursor()
        c.execute("""
            INSERT INTO evaluations
            (window_id, expert_name, architecture, symbol, total_trades,
             wins, losses, profit_factor, win_rate, sharpe, max_drawdown,
             net_pnl, composite_score, tier, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            window_id, entry.name, entry.architecture, symbol,
            perf.total_trades, perf.wins, perf.losses,
            round(perf.profit_factor, 4), round(perf.win_rate, 4),
            round(perf.sharpe, 4), round(perf.max_drawdown, 4),
            round(perf.net_pnl, 4), round(entry.composite_score, 4),
            entry.tier.value,
            datetime.now(timezone.utc).isoformat(),
        ))
        self._conn.commit()

    def log_breeding(self, window_id: int, operator: str,
                     parent1: str, parent2: str, child: str,
                     rate: float):
        c = self._conn.cursor()
        c.execute("""
            INSERT INTO breeding_log
            (window_id, operator, parent_1, parent_2, child_name,
             mutation_rate, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (window_id, operator, parent1, parent2, child, rate,
              datetime.now(timezone.utc).isoformat()))
        self._conn.commit()

    def log_roster_change(self, window_id: int, action: str,
                          promoted: str, relegated: str):
        c = self._conn.cursor()
        c.execute("""
            INSERT INTO roster_changes
            (window_id, action, expert_promoted, expert_relegated, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """, (window_id, action, promoted, relegated,
              datetime.now(timezone.utc).isoformat()))
        self._conn.commit()

    def close(self):
        self._conn.close()


# ===========================================================================
# EXPERT DISCOVERY
# ===========================================================================

class ExpertDiscovery:
    """Discovers and catalogs all available experts."""

    def __init__(self, config: dict):
        self.qt_library = config["qt_library"]
        self.spare_champions = config["spare_champions"]

    def discover_all(self) -> List[ExpertEntry]:
        """
        Scan the Quantum-Teenagers library and spare champions.
        Returns a list of ExpertEntry objects for all discoverable experts.
        """
        entries = []

        # 1. Discover MQ5 system folders (133 with Test.mq5)
        mq5_entries = self._discover_mq5_systems()
        entries.extend(mq5_entries)

        # 2. Discover army champion .pth files
        pth_entries = self._discover_army_champions()
        entries.extend(pth_entries)

        # 3. Discover conv1d .pth files
        conv1d_entries = self._discover_conv1d_models()
        entries.extend(conv1d_entries)

        # 4. Discover JSON weight configs
        json_entries = self._discover_json_experts()
        entries.extend(json_entries)

        # 5. Discover Python experts from #1 rankers (24 QC experts)
        py_entries = self._discover_python_experts()
        entries.extend(py_entries)

        logger.info(f"Discovered {len(entries)} total experts:")
        logger.info(f"  MQ5 systems: {len(mq5_entries)}")
        logger.info(f"  Army champions: {len(pth_entries)}")
        logger.info(f"  Conv1D models: {len(conv1d_entries)}")
        logger.info(f"  Python experts: {len(py_entries)}")
        logger.info(f"  JSON experts: {len(json_entries)}")

        return entries

    def _discover_mq5_systems(self) -> List[ExpertEntry]:
        """Find all folders with Test.mq5."""
        entries = []
        if not os.path.exists(self.qt_library):
            return entries

        for folder_name in os.listdir(self.qt_library):
            folder_path = os.path.join(self.qt_library, folder_name)
            if not os.path.isdir(folder_path):
                continue
            test_mq5 = os.path.join(folder_path, "Test.mq5")
            if os.path.exists(test_mq5):
                entries.append(ExpertEntry(
                    name=f"MQ5_{folder_name}",
                    architecture=folder_name,
                    expert_type=ExpertType.MQ5_SYSTEM,
                    source_path=folder_path,
                    weight_path=None,  # MQ5 weights are inside the .mq5/mqh
                    tier=Tier.MINOR,
                ))

        return entries

    def _discover_army_champions(self) -> List[ExpertEntry]:
        """Find army_champion_*.pth files."""
        entries = []
        if not os.path.exists(self.spare_champions):
            return entries

        for fname in os.listdir(self.spare_champions):
            if not fname.startswith("army_champion_") or \
               not fname.endswith(".pth"):
                continue

            fpath = os.path.join(self.spare_champions, fname)

            # Parse: army_champion_01_BTCUSD_CROSSOVER.pth
            parts = fname.replace(".pth", "").split("_")
            # Extract breeding method (last part)
            breeding = parts[-1] if len(parts) > 3 else "UNKNOWN"
            # Extract symbol
            symbol = parts[-2] if len(parts) > 3 else "UNKNOWN"

            entries.append(ExpertEntry(
                name=fname.replace(".pth", ""),
                architecture="army_champion",
                expert_type=ExpertType.PTH_ARMY,
                source_path=self.spare_champions,
                weight_path=fpath,
                tier=Tier.MINOR,
                breeding_op=breeding,
            ))

        return entries

    def _discover_conv1d_models(self) -> List[ExpertEntry]:
        """Find expert_*_conv1d.pth files."""
        entries = []
        if not os.path.exists(self.spare_champions):
            return entries

        for fname in os.listdir(self.spare_champions):
            if "conv1d" in fname and fname.endswith(".pth"):
                fpath = os.path.join(self.spare_champions, fname)
                sym = "UNKNOWN"
                if "BTCUSD" in fname:
                    sym = "BTCUSD"
                elif "ETHUSD" in fname:
                    sym = "ETHUSD"
                elif "XAUUSD" in fname:
                    sym = "XAUUSD"

                entries.append(ExpertEntry(
                    name=fname.replace(".pth", ""),
                    architecture="conv1d",
                    expert_type=ExpertType.PTH_CONV1D,
                    source_path=self.spare_champions,
                    weight_path=fpath,
                    tier=Tier.MINOR,
                ))

        return entries

    def _discover_json_experts(self) -> List[ExpertEntry]:
        """Find expert_*.json weight config files."""
        entries = []
        if not os.path.exists(self.spare_champions):
            return entries

        for fname in os.listdir(self.spare_champions):
            if fname.startswith("expert_") and fname.endswith(".json") \
               and "manifest" not in fname:
                fpath = os.path.join(self.spare_champions, fname)

                # Parse: expert_BTCUSD_M1_c1_10.json
                parts = fname.replace(".json", "").split("_")
                sym = parts[1] if len(parts) > 1 else "UNKNOWN"
                tf = parts[2] if len(parts) > 2 else "M1"

                entries.append(ExpertEntry(
                    name=fname.replace(".json", ""),
                    architecture=f"json_{tf}",
                    expert_type=ExpertType.JSON_WEIGHTS,
                    source_path=self.spare_champions,
                    weight_path=fpath,
                    tier=Tier.MINOR,
                ))

        return entries

    def _discover_python_experts(self) -> List[ExpertEntry]:
        """
        Discover the 24 Python ExpertBase experts from #1 rankers.
        These are the QuantumChildren Python experts that use the MT5 mock.
        """
        entries = []
        rankers_dir = os.path.join(self.qt_library, "#1 rankers")
        if not os.path.exists(rankers_dir):
            logger.warning(f"#1 rankers directory not found: {rankers_dir}")
            return entries

        for module_name, class_name in EXPERT_REGISTRY.items():
            py_file = os.path.join(rankers_dir, f"{module_name}.py")
            if os.path.exists(py_file):
                entries.append(ExpertEntry(
                    name=f"PY_{class_name}",
                    architecture="QuantumChildren",
                    expert_type=ExpertType.PYTHON_EXPERT,
                    source_path=module_name,  # module name for importlib
                    weight_path=None,  # Weights are internal to the expert
                    tier=Tier.MINOR,
                ))

        logger.info(f"  Found {len(entries)} Python experts in #1 rankers")
        return entries


# ===========================================================================
# DASHBOARD
# ===========================================================================

class Dashboard:
    """Console output for J-TEAR progress."""

    @staticmethod
    def print_header():
        print()
        print("=" * 90)
        print("       _   _____ _____    _    ____")
        print("      | | |_   _| ____|  / \\  |  _ \\")
        print("   _  | |   | | |  _|   / _ \\ | |_) |")
        print("  | |_| |   | | | |___ / ___ \\|  _ <")
        print("   \\___/    |_| |_____/_/   \\_\\_| \\_\\")
        print()
        print("  J-TEAR: Hybrid Ensemble Breeding Engine")
        print("  Walk-Forward League System | Continuous Evolution")
        print("=" * 90)

    @staticmethod
    def print_window_header(window: WalkForwardWindow, n_windows: int):
        print()
        print("+" + "-" * 88 + "+")
        pct = (window.window_id + 1) / n_windows * 100
        print(f"|  Window {window.window_id + 1}/{n_windows} "
              f"({pct:.0f}%)"
              f"{'':>60}|")
        print(f"|  Train: {window.train_start[:10]} to {window.train_end[:10]}"
              f"   Test: {window.test_start[:10]} to {window.test_end[:10]}"
              f"{'':>15}|")
        print("+" + "-" * 88 + "+")

    @staticmethod
    def print_evaluation_results(active_ranked: List[Tuple[str, float]],
                                  minor_ranked: List[Tuple[str, float]],
                                  perfs: Dict[str, ExpertPerformance]):
        print()
        print("  ACTIVE ROSTER:")
        print(f"  {'Rank':<5} {'Expert':<35} {'Score':>7} {'Trades':>7} "
              f"{'WR%':>6} {'PF':>6} {'PnL':>9} {'Weight':>7}")
        print("  " + "-" * 85)

        for rank, (name, score) in enumerate(active_ranked, 1):
            perf = perfs.get(name)
            if perf and perf.total_trades > 0:
                wr = f"{perf.win_rate * 100:.1f}%"
                pf = f"{perf.profit_factor:.2f}"
                pnl = f"${perf.net_pnl:+.2f}"
            else:
                wr = "N/A"
                pf = "N/A"
                pnl = "$0.00"
            trades = perf.total_trades if perf else 0
            print(f"  {rank:<5} {name:<35} {score:>7.3f} {trades:>7} "
                  f"{wr:>6} {pf:>6} {pnl:>9} {'':>7}")

        print()
        print("  MINOR LEAGUE (top 5):")
        for rank, (name, score) in enumerate(minor_ranked[:5], 1):
            perf = perfs.get(name)
            trades = perf.total_trades if perf else 0
            wr = f"{perf.win_rate * 100:.1f}%" if perf and perf.total_trades > 0 else "N/A"
            print(f"    {rank}. {name:<35} score={score:.3f} "
                  f"trades={trades} WR={wr}")

    @staticmethod
    def print_breeding_event(offspring: List[ExpertEntry]):
        if not offspring:
            return
        print()
        print("  BREEDING:")
        for child in offspring:
            parents = " x ".join(child.lineage) if child.lineage else "genesis"
            print(f"    {child.breeding_op}: {parents} -> {child.name}")

    @staticmethod
    def print_roster_changes(swaps: List[Tuple[str, str]],
                              pruned: List[str]):
        if swaps:
            print()
            print("  ROSTER CHANGES:")
            for promoted, relegated in swaps:
                print(f"    CALL-UP:    {promoted}")
                print(f"    RELEGATED:  {relegated}")
        if pruned:
            for name in pruned:
                print(f"    PRUNED:     {name}")

    @staticmethod
    def print_final_summary(roster: RosterManager, total_windows: int):
        print()
        print("=" * 90)
        print("  J-TEAR FINAL REPORT")
        print("=" * 90)

        active = sorted(roster.get_active(),
                        key=lambda e: e.composite_score, reverse=True)
        print()
        print(f"  ACTIVE ROSTER ({len(active)} experts):")
        print(f"  {'Name':<35} {'Arch':<15} {'Score':>7} "
              f"{'Vote%':>7} {'Origin':<15}")
        print("  " + "-" * 82)
        for e in active:
            print(f"  {e.name:<35} {e.architecture:<15} "
                  f"{e.composite_score:>7.3f} {e.vote_weight*100:>6.1f}% "
                  f"{e.breeding_op:<15}")

        minor = roster.get_minor()
        print()
        print(f"  MINOR LEAGUE: {len(minor)} experts")
        print(f"  Total windows evaluated: {total_windows}")
        print("=" * 90)


# ===========================================================================
# J-TEAR ENGINE (main orchestrator)
# ===========================================================================

class JTEAREngine:
    """
    Main orchestration engine for J-TEAR.
    Ties together all components: data, walk-forward, evaluation,
    scoring, breeding, roster management, and persistence.
    """

    def __init__(self, config: dict = None):
        self.config = config or JTEAR_CONFIG
        self.workspace = self.config["workspace"]
        os.makedirs(self.workspace, exist_ok=True)
        os.makedirs(os.path.join(self.workspace, "offspring"), exist_ok=True)
        os.makedirs(os.path.join(self.workspace, "logs"), exist_ok=True)

        # Components
        self.data_layer = DataLayer(
            self.config["data_dir"], self.config["symbols"])
        self.window_mgr = WindowManager(self.config)
        self.scorer = Scorer(self.config)
        self.breeder = Breeder(self.config, self.workspace)
        self.roster = RosterManager(self.config)
        self.voter = EnsembleVoter(self.config)
        self.db = HistoryDB(
            os.path.join(self.workspace, "history.db"))
        self.discovery = ExpertDiscovery(self.config)

        # State
        self.last_completed_window: int = -1
        self.windows: List[WalkForwardWindow] = []

    def run(self):
        """Main execution loop."""
        Dashboard.print_header()

        # --- Step 1: Load Data ---
        print()
        print("[1/5] LOADING MARKET DATA...")
        t0 = time.time()
        self.data_layer.load()
        data_start, data_end = self.data_layer.get_date_range()
        print(f"  Data range: {data_start} to {data_end}")
        print(f"  Loaded in {time.time() - t0:.1f}s")

        # --- Step 2: Generate Walk-Forward Windows ---
        print()
        print("[2/5] GENERATING WALK-FORWARD WINDOWS...")
        self.windows = self.window_mgr.generate_windows(data_start, data_end)
        print(f"  Generated {len(self.windows)} windows")
        if self.windows:
            print(f"  First: {self.windows[0].test_start[:10]} to "
                  f"{self.windows[0].test_end[:10]}")
            print(f"  Last:  {self.windows[-1].test_start[:10]} to "
                  f"{self.windows[-1].test_end[:10]}")

        # --- Step 3: Discover Experts ---
        print()
        print("[3/5] DISCOVERING EXPERTS...")
        all_entries = self.discovery.discover_all()

        # Filter to experts we can evaluate in Python
        # (PTH, JSON, and PYTHON_EXPERT -- MQ5 systems need Strategy Tester)
        evaluable = [e for e in all_entries
                     if e.expert_type != ExpertType.MQ5_SYSTEM]

        n_py = sum(1 for e in evaluable
                   if e.expert_type == ExpertType.PYTHON_EXPERT)
        n_pth = sum(1 for e in evaluable
                    if e.expert_type in (ExpertType.PTH_CONV1D,
                                         ExpertType.PTH_ARMY))
        n_json = sum(1 for e in evaluable
                     if e.expert_type == ExpertType.JSON_WEIGHTS)
        n_mq5 = len(all_entries) - len(evaluable)

        print(f"  Evaluable: {len(evaluable)} "
              f"({n_pth} PTH + {n_json} JSON + {n_py} Python)")
        print(f"  Deferred:  {n_mq5} MQ5 systems (need Strategy Tester)")
        logger.info(f"  {len(evaluable)} experts evaluable, "
                    f"{n_mq5} MQ5 deferred")

        if not evaluable:
            print("  FATAL: No evaluable experts found. Aborting.")
            return

        # --- Step 4: Try to Resume ---
        print()
        print("[4/5] CHECKING FOR PREVIOUS STATE...")
        resumed = self._try_resume()
        if not resumed:
            print("  No previous state found. Initializing fresh roster.")
            self.roster.initialize_roster(evaluable)
        else:
            print(f"  Resumed from window {self.last_completed_window + 1}")

        # --- Step 5: GPU Status ---
        print()
        print("[5/5] SYSTEM STATUS")
        if GPU_AVAILABLE:
            print(f"  GPU: ACTIVE (DirectML)")
        else:
            print(f"  GPU: NOT AVAILABLE (CPU only)")
        print(f"  Active Roster: {len(self.roster.get_active())}")
        print(f"  Minor League:  {len(self.roster.get_minor())}")
        print(f"  Total experts: {len(self.roster.get_all())}")

        # --- Main Loop ---
        start_window = self.last_completed_window + 1
        total_windows = len(self.windows)

        for window in self.windows[start_window:]:
            Dashboard.print_window_header(window, total_windows)
            self._process_window(window)
            self.last_completed_window = window.window_id
            self._save_state()

        # --- Final Report ---
        Dashboard.print_final_summary(self.roster, total_windows)
        self.db.close()

        # Save final state
        self._save_state()
        print(f"\n  State saved to: {self.config['state_file']}")
        print(f"  History DB:     {os.path.join(self.workspace, 'history.db')}")
        print()

    def _process_window(self, window: WalkForwardWindow):
        """Process a single walk-forward window."""
        all_experts = self.roster.get_all()
        symbols = self.config["symbols"]

        # --- Evaluate all experts ---
        backtester = BacktestEngine(self.data_layer, self.config)
        all_perfs: Dict[str, ExpertPerformance] = {}

        for ei, entry in enumerate(all_experts):
            combined = ExpertPerformance(
                expert_name=entry.name, window_id=window.window_id)

            for symbol in symbols:
                perf = backtester.evaluate_expert(
                    entry, symbol, window.test_start, window.test_end)

                combined.total_trades += perf.total_trades
                combined.wins += perf.wins
                combined.losses += perf.losses
                combined.gross_profit += perf.gross_profit
                combined.gross_loss += perf.gross_loss
                combined.net_pnl += perf.net_pnl
                combined.trade_pnls.extend(perf.trade_pnls)
                if perf.max_drawdown > combined.max_drawdown:
                    combined.max_drawdown = perf.max_drawdown

                # Log to DB
                self.db.log_evaluation(
                    window.window_id, entry, perf, symbol)

            all_perfs[entry.name] = combined

            # Progress
            pct = (ei + 1) / len(all_experts) * 100
            sys.stdout.write(
                f"\r  Evaluating: {ei+1}/{len(all_experts)} ({pct:.0f}%) "
                f"- {entry.name[:30]:<30} "
                f"[{combined.total_trades}t, ${combined.net_pnl:+.2f}]")
            sys.stdout.flush()

        print()

        # --- Score all experts ---
        scores = {}
        for name, perf in all_perfs.items():
            entry = self.roster.experts.get(name)
            if entry:
                history = entry.score_history
                scores[name] = self.scorer.compute_composite(perf, history)

        self.roster.update_scores(scores)

        # Rank active and minor separately
        active_names = {e.name for e in self.roster.get_active()}
        minor_names = {e.name for e in self.roster.get_minor()}

        active_scores = {n: s for n, s in scores.items() if n in active_names}
        minor_scores = {n: s for n, s in scores.items() if n in minor_names}

        active_ranked = self.scorer.rank_population(active_scores)
        minor_ranked = self.scorer.rank_population(minor_scores)

        Dashboard.print_evaluation_results(
            active_ranked, minor_ranked, all_perfs)

        # --- Promotions / Relegations ---
        swaps = self.roster.handle_promotions()
        for promoted, relegated in swaps:
            self.db.log_roster_change(
                window.window_id, "CALL_UP", promoted, relegated)

        # --- Breeding ---
        offspring = self._run_breeding(window.window_id, all_perfs)
        Dashboard.print_breeding_event(offspring)

        for child in offspring:
            self.roster.add_expert(child)

        # --- Prune minor league ---
        pruned = self.roster.prune_minor_league()

        Dashboard.print_roster_changes(swaps, pruned)

        # --- Update vote weights ---
        self.roster.update_vote_weights()

    def _run_breeding(self, window_id: int,
                      perfs: Dict[str, ExpertPerformance]
                      ) -> List[ExpertEntry]:
        """Execute breeding operators for this window."""
        offspring = []
        n_offspring = self.config["offspring_per_cycle"]

        # Get all experts sorted by score for parent selection
        all_entries = sorted(
            self.roster.get_all(),
            key=lambda e: e.composite_score,
            reverse=True
        )

        # Only breed from experts that have evaluable weights
        breedable = [e for e in all_entries
                     if e.weight_path and os.path.exists(e.weight_path)]

        if not breedable:
            return offspring

        for _ in range(n_offspring):
            op = self.breeder.select_operator()

            if op == BreedingOp.CROSSOVER_SAME:
                # Find two parents of the same architecture
                arch_groups = defaultdict(list)
                for e in breedable:
                    arch_groups[e.architecture].append(e)

                # Pick an architecture with 2+ members
                viable_archs = {a: es for a, es in arch_groups.items()
                                if len(es) >= 2}
                if viable_archs:
                    arch = random.choice(list(viable_archs.keys()))
                    parents = random.sample(viable_archs[arch], 2)
                else:
                    parents = [random.choice(breedable)]
                    op = BreedingOp.MUTATION

            elif op == BreedingOp.CROSSOVER_DIFF:
                # Cross-arch: skip (handled as vote weight adjustment)
                continue

            elif op == BreedingOp.BLEND:
                # Top 3-5 same-architecture experts
                arch_groups = defaultdict(list)
                for e in breedable[:10]:
                    arch_groups[e.architecture].append(e)
                viable = {a: es for a, es in arch_groups.items()
                          if len(es) >= 2}
                if viable:
                    arch = random.choice(list(viable.keys()))
                    parents = viable[arch][:5]
                else:
                    parents = [random.choice(breedable)]
                    op = BreedingOp.MUTATION

            else:
                # Single-parent operators
                # Tournament selection: pick best of 3 random
                candidates = random.sample(
                    breedable, min(3, len(breedable)))
                parents = [max(candidates,
                               key=lambda e: e.composite_score)]

            child = self.breeder.breed(op, parents, window_id)
            if child is not None:
                offspring.append(child)

                # Log breeding
                p1 = parents[0].name if parents else ""
                p2 = parents[1].name if len(parents) > 1 else ""
                rate = (self.config["radiation_rate"]
                        if op == BreedingOp.RADIATION
                        else self.config["mutation_rate"])
                self.db.log_breeding(
                    window_id, op.value, p1, p2, child.name, rate)

        return offspring

    def _save_state(self):
        """Persist current engine state to JSON."""
        state = {
            "engine": "J-TEAR",
            "version": "1.0",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "last_completed_window": self.last_completed_window,
            "config": {
                k: v for k, v in self.config.items()
                if isinstance(v, (str, int, float, bool, list, dict))
            },
            "roster": self.roster.to_dict(),
        }

        state_path = self.config["state_file"]
        with open(state_path, "w") as f:
            json.dump(state, f, indent=2, default=str)

    def _try_resume(self) -> bool:
        """Try to load previous state. Returns True if resumed."""
        state_path = self.config["state_file"]
        if not os.path.exists(state_path):
            return False

        try:
            with open(state_path, "r") as f:
                state = json.load(f)

            self.last_completed_window = state.get(
                "last_completed_window", -1)
            if "roster" in state:
                self.roster.from_dict(state["roster"])
            return True
        except Exception as e:
            logger.warning(f"Failed to resume: {e}")
            return False


# ===========================================================================
# ENTRY POINT
# ===========================================================================

def main():
    """Run J-TEAR."""
    engine = JTEAREngine(JTEAR_CONFIG)
    engine.run()


if __name__ == "__main__":
    main()
