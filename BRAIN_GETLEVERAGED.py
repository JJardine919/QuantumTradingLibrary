"""
QUANTUM BRAIN - GETLEVERAGED DEDICATED
=======================================
Handles GetLeveraged accounts ONLY.

Accounts:
  - 113326 (Account 1)
  - 113328 (Account 2)
  - 107245 (Account 3)

Run: python BRAIN_GETLEVERAGED.py
     python BRAIN_GETLEVERAGED.py --unlock-all

Author: DooDoo + Claude
Date: 2026-01-30
"""

import sys
import os
import json
import time
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from enum import Enum
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass

import torch
import torch.nn as nn
import MetaTrader5 as mt5

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][GL] %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler('brain_getleveraged.log'),
        logging.StreamHandler()
    ]
)

# ============================================================
# GETLEVERAGED ACCOUNTS ONLY
# ============================================================

ACCOUNTS = {
    'GL_ACCOUNT_1': {
        'account': 113326,
        'password': '',
        'server': 'GetLeveraged-Server',
        'terminal_path': r"C:\Program Files\GetLeveraged MT5 Terminal\terminal64.exe",
        'name': 'GetLeveraged Account 1',
        'initial_balance': 10000,
        'profit_target': 0.10,
        'daily_loss_limit': 0.05,
        'max_drawdown': 0.10,
        'locked': False,  # ACTIVE
        'symbols': ['BTCUSD'],
        'magic_number': 113001,
    },
    'GL_ACCOUNT_2': {
        'account': 113328,
        'password': '',
        'server': 'GetLeveraged-Server',
        'terminal_path': r"C:\Program Files\GetLeveraged MT5 Terminal\terminal64.exe",
        'name': 'GetLeveraged Account 2',
        'initial_balance': 10000,
        'profit_target': 0.10,
        'daily_loss_limit': 0.05,
        'max_drawdown': 0.10,
        'locked': False,  # ACTIVE
        'symbols': ['BTCUSD'],
        'magic_number': 113002,
    },
    'GL_ACCOUNT_3': {
        'account': 107245,
        'password': '',
        'server': 'GetLeveraged-Server',
        'terminal_path': r"C:\Program Files\GetLeveraged MT5 Terminal\terminal64.exe",
        'name': 'GetLeveraged Account 3',
        'initial_balance': 10000,
        'profit_target': 0.10,
        'daily_loss_limit': 0.05,
        'max_drawdown': 0.10,
        'locked': False,  # ACTIVE
        'symbols': ['BTCUSD'],
        'magic_number': 107001,
    },
}


# ============================================================
# CONFIG
# ============================================================

@dataclass
class BrainConfig:
    CLEAN_REGIME_THRESHOLD: float = 0.95
    VOLATILE_REGIME_THRESHOLD: float = 0.85
    CONFIDENCE_THRESHOLD: float = 0.55
    RISK_PER_TRADE_PCT: float = 0.005
    BARS_FOR_ANALYSIS: int = 256
    SEQUENCE_LENGTH: int = 30
    CHECK_INTERVAL_SECONDS: int = 60
    BASE_LOT: float = 0.01


class Action(Enum):
    HOLD = 0
    BUY = 1
    SELL = 2


class Regime(Enum):
    CLEAN = "CLEAN"
    VOLATILE = "VOLATILE"
    CHOPPY = "CHOPPY"


# ============================================================
# LSTM MODEL
# ============================================================

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, output_size=3, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out


# ============================================================
# REGIME DETECTOR (Compression-based)
# ============================================================

class RegimeDetector:
    def __init__(self, config: BrainConfig):
        self.config = config

    def analyze_regime(self, prices: np.ndarray) -> Tuple[Regime, float]:
        import zlib
        data_bytes = prices.astype(np.float32).tobytes()
        compressed = zlib.compress(data_bytes, level=9)
        ratio = len(data_bytes) / len(compressed)

        if ratio >= 3.5:
            fidelity = 0.96
            regime = Regime.CLEAN
        elif ratio >= 2.5:
            fidelity = 0.88
            regime = Regime.VOLATILE
        else:
            fidelity = 0.75
            regime = Regime.CHOPPY

        return regime, fidelity


# ============================================================
# EXPERT LOADER
# ============================================================

class ExpertLoader:
    def __init__(self, experts_dir: str = "top_50_experts"):
        self.experts_dir = Path(experts_dir)
        self.manifest = None
        self.loaded_experts: Dict[str, nn.Module] = {}
        self._load_manifest()

    def _load_manifest(self):
        manifest_path = self.experts_dir / "top_50_manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                self.manifest = json.load(f)
            logging.info(f"Loaded {len(self.manifest['experts'])} experts")

    def get_best_expert_for_symbol(self, symbol: str) -> Optional[nn.Module]:
        if not self.manifest:
            return None
        for expert in self.manifest['experts']:
            if expert['symbol'] == symbol:
                return self._load_expert(expert)
        return None

    def _load_expert(self, expert_info: dict) -> Optional[nn.Module]:
        filename = expert_info['filename']
        if filename in self.loaded_experts:
            return self.loaded_experts[filename]

        expert_path = self.experts_dir / filename
        if not expert_path.exists():
            return None

        try:
            model = LSTMModel(
                input_size=expert_info['input_size'],
                hidden_size=expert_info['hidden_size'],
                output_size=3,
                num_layers=2
            )
            state_dict = torch.load(str(expert_path), map_location='cpu', weights_only=False)
            model.load_state_dict(state_dict)
            model.eval()
            self.loaded_experts[filename] = model
            return model
        except Exception as e:
            logging.error(f"Failed to load expert: {e}")
            return None


# ============================================================
# FEATURE ENGINEER (MUST match training exactly)
# ============================================================

class FeatureEngineer:
    """Features: rsi, macd, macd_signal, bb_upper, bb_lower, momentum, roc, atr"""

    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        data = df.copy()
        for c in ['open', 'high', 'low', 'close', 'tick_volume']:
            if c in data.columns:
                data[c] = data[c].astype(float)

        # RSI
        delta = data['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))

        # MACD
        exp1 = data['close'].ewm(span=12, adjust=False).mean()
        exp2 = data['close'].ewm(span=26, adjust=False).mean()
        data['macd'] = exp1 - exp2
        data['macd_signal'] = data['macd'].ewm(span=9, adjust=False).mean()

        # Bollinger Bands
        data['bb_middle'] = data['close'].rolling(20).mean()
        data['bb_std'] = data['close'].rolling(20).std()
        data['bb_upper'] = data['bb_middle'] + 2 * data['bb_std']
        data['bb_lower'] = data['bb_middle'] - 2 * data['bb_std']

        # Momentum & ROC
        data['momentum'] = data['close'] / data['close'].shift(10)
        data['roc'] = data['close'].pct_change(10) * 100

        # ATR
        data['tr'] = np.maximum(
            data['high'] - data['low'],
            np.maximum(
                abs(data['high'] - data['close'].shift(1)),
                abs(data['low'] - data['close'].shift(1))
            )
        )
        data['atr'] = data['tr'].rolling(14).mean()

        feature_cols = ['rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower', 'momentum', 'roc', 'atr']
        data = data.dropna()

        if len(data) == 0:
            return np.zeros((1, len(feature_cols)))

        # Global Z-score normalization
        for col in feature_cols:
            col_mean = data[col].mean()
            col_std = data[col].std() + 1e-8
            data[col] = (data[col] - col_mean) / col_std

        data = data.fillna(0)
        return data[feature_cols].values


# ============================================================
# ACCOUNT TRADER
# ============================================================

class AccountTrader:
    def __init__(self, account_key: str, account_config: dict, config: BrainConfig):
        self.account_key = account_key
        self.account_config = account_config
        self.config = config
        self.regime_detector = RegimeDetector(config)
        self.expert_loader = ExpertLoader()
        self.feature_engineer = FeatureEngineer()
        self.starting_balance = 0.0

    def connect(self) -> bool:
        mt5.shutdown()
        if not mt5.initialize(path=self.account_config['terminal_path']):
            logging.error(f"[{self.account_key}] MT5 init failed: {mt5.last_error()}")
            return False

        if self.account_config.get('password'):
            if not mt5.login(self.account_config['account'],
                           password=self.account_config['password'],
                           server=self.account_config['server']):
                logging.error(f"[{self.account_key}] Login failed")
                return False

        account_info = mt5.account_info()
        if account_info is None:
            return False

        self.starting_balance = account_info.balance
        logging.info(f"[{self.account_key}] Connected - Balance: ${account_info.balance:,.2f}")
        return True

    def check_risk_limits(self) -> bool:
        account_info = mt5.account_info()
        if account_info is None:
            return False

        if self.starting_balance > 0:
            daily_loss = (self.starting_balance - account_info.balance) / self.starting_balance
            if daily_loss >= self.account_config['daily_loss_limit']:
                logging.warning(f"[{self.account_key}] DAILY LOSS LIMIT: {daily_loss*100:.2f}%")
                return False
        return True

    def get_lot_size(self, symbol: str) -> float:
        account_info = mt5.account_info()
        symbol_info = mt5.symbol_info(symbol)
        if account_info is None or symbol_info is None:
            return self.config.BASE_LOT

        vol_min = symbol_info.volume_min
        vol_max = symbol_info.volume_max
        vol_step = symbol_info.volume_step

        desired_lot = (account_info.balance / 5000) * 0.01
        lot = round(desired_lot / vol_step) * vol_step
        lot = max(vol_min, min(lot, vol_max))
        lot = min(lot, 1.0)
        return lot

    def analyze_symbol(self, symbol: str) -> Tuple[Regime, float, Optional[Action], float]:
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, self.config.BARS_FOR_ANALYSIS)
        if rates is None or len(rates) < self.config.BARS_FOR_ANALYSIS:
            return Regime.CHOPPY, 0.0, None, 0.0

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        prices = df['close'].values

        regime, fidelity = self.regime_detector.analyze_regime(prices)
        logging.info(f"[{self.account_key}][{symbol}] Regime: {regime.value} ({fidelity:.3f})")

        if regime != Regime.CLEAN:
            return regime, fidelity, Action.HOLD, 0.0

        expert = self.expert_loader.get_best_expert_for_symbol(symbol)
        if expert is None:
            return regime, fidelity, None, 0.0

        features = self.feature_engineer.prepare_features(df)
        seq_len = self.config.SEQUENCE_LENGTH
        if len(features) < seq_len:
            return regime, fidelity, Action.HOLD, 0.0

        sequence = features[-seq_len:]
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0)

        with torch.no_grad():
            output = expert(sequence_tensor)
            probs = torch.softmax(output, dim=1)
            action_idx = torch.argmax(probs).item()
            confidence = probs[0, action_idx].item()

        action = Action(action_idx)
        if confidence < self.config.CONFIDENCE_THRESHOLD:
            action = Action.HOLD

        logging.info(f"[{self.account_key}][{symbol}] Signal: {action.name} ({confidence:.2f})")
        return regime, fidelity, action, confidence

    def has_position(self, symbol: str) -> bool:
        positions = mt5.positions_get(symbol=symbol)
        if positions:
            for pos in positions:
                if pos.magic == self.account_config['magic_number']:
                    return True
        return False

    def execute_trade(self, symbol: str, action: Action, confidence: float) -> bool:
        if action not in [Action.BUY, Action.SELL]:
            return False

        if self.has_position(symbol):
            logging.info(f"[{self.account_key}] Already have position in {symbol}")
            return False

        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            return False

        if not symbol_info.visible:
            mt5.symbol_select(symbol, True)

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return False

        lot = self.get_lot_size(symbol)

        # ATR for SL/TP
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 15)
        if rates is not None:
            df = pd.DataFrame(rates)
            atr = (df['high'] - df['low']).rolling(14).mean().iloc[-1]
        else:
            atr = 100 * symbol_info.point

        if action == Action.BUY:
            order_type = mt5.ORDER_TYPE_BUY
            price = tick.ask
            sl = price - (atr * 1.5)
            tp = price + (atr * 3.0)
        else:
            order_type = mt5.ORDER_TYPE_SELL
            price = tick.bid
            sl = price + (atr * 1.5)
            tp = price - (atr * 3.0)

        filling_mode = mt5.ORDER_FILLING_IOC
        if symbol_info.filling_mode & mt5.ORDER_FILLING_FOK:
            filling_mode = mt5.ORDER_FILLING_FOK

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "magic": self.account_config['magic_number'],
            "comment": f"GL_{self.account_key}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling_mode,
        }

        result = mt5.order_send(request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            logging.info(f"[{self.account_key}] TRADE: {action.name} {symbol} @ {price:.2f} SL:{sl:.2f} TP:{tp:.2f}")
            return True
        else:
            logging.error(f"[{self.account_key}] FAILED: {result.comment} ({result.retcode})")
            return False

    def run_cycle(self) -> Dict[str, dict]:
        results = {}
        if not self.check_risk_limits():
            return results

        for symbol in self.account_config['symbols']:
            regime, fidelity, action, confidence = self.analyze_symbol(symbol)
            trade_executed = False
            if regime == Regime.CLEAN and action in [Action.BUY, Action.SELL]:
                trade_executed = self.execute_trade(symbol, action, confidence)

            results[symbol] = {
                'regime': regime.value,
                'fidelity': fidelity,
                'action': action.name if action else 'NONE',
                'confidence': confidence,
                'trade_executed': trade_executed,
            }
        return results


# ============================================================
# MAIN BRAIN
# ============================================================

class GetLeveragedBrain:
    def __init__(self, config: BrainConfig = None, unlock_all: bool = False):
        self.config = config or BrainConfig()
        self.unlock_all = unlock_all
        self.traders: Dict[str, AccountTrader] = {}

    def get_active_accounts(self) -> List[str]:
        active = []
        for key, account in ACCOUNTS.items():
            if self.unlock_all or not account.get('locked', True):
                active.append(key)
        return active

    def initialize(self) -> bool:
        active = self.get_active_accounts()
        if not active:
            logging.error("No active accounts!")
            return False

        logging.info(f"Initializing: {active}")
        for key in active:
            self.traders[key] = AccountTrader(key, ACCOUNTS[key], self.config)
        return True

    def run_loop(self):
        print("=" * 60)
        print("  GETLEVERAGED QUANTUM BRAIN")
        print("  Dedicated brain for GetLeveraged accounts")
        print("=" * 60)

        if not self.initialize():
            return

        account_keys = list(self.traders.keys())
        idx = 0

        try:
            while True:
                key = account_keys[idx]
                trader = self.traders[key]

                print(f"\n{'='*60}")
                print(f"  {ACCOUNTS[key]['name']} | {datetime.now().strftime('%H:%M:%S')}")
                print(f"{'='*60}")

                if trader.connect():
                    results = trader.run_cycle()
                    for symbol, data in results.items():
                        icon = "+" if data['regime'] == 'CLEAN' else "-"
                        status = "TRADED" if data['trade_executed'] else ""
                        print(f"  [{icon}] {symbol}: {data['regime']} | "
                              f"{data['action']} ({data['confidence']:.2f}) {status}")

                idx = (idx + 1) % len(account_keys)
                time.sleep(self.config.CHECK_INTERVAL_SECONDS)

        except KeyboardInterrupt:
            logging.info("Stopped")
        finally:
            mt5.shutdown()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GetLeveraged Quantum Brain')
    parser.add_argument('--unlock-all', action='store_true', help='Trade all GL accounts')
    args = parser.parse_args()

    brain = GetLeveragedBrain(unlock_all=args.unlock_all)
    brain.run_loop()
