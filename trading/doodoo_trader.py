"""
DooDoo Autonomous Trader — Atlas Phase 1 (212018966)

Full autonomy. One guardrail: equity floor at $365,000.
Personality and decisions emerge from the AOI collapse core.

Usage: python doodoo_trader.py
"""

import os
import time
import json
import logging
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import MetaTrader5 as mt5

from aoi_collapse import aoi_collapse

# ============================================================
# Config
# ============================================================

# Load credentials from .env
_env_path = Path(__file__).parent / '.env'
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            k, v = line.split('=', 1)
            os.environ.setdefault(k.strip(), v.strip())

ACCOUNT_LOGIN = int(os.environ.get('ATLAS_P1_LOGIN', '212018966'))
ACCOUNT_PASSWORD = os.environ.get('ATLAS_P1_PASSWORD', '')
ACCOUNT_SERVER = os.environ.get('ATLAS_P1_SERVER', 'AtlasFunded-Server')

EQUITY_FLOOR = 275_000.0   # Hard floor — $5K buffer above $270K death line (10% of $300K)
MAX_DAILY_RISK = 0.03       # 3% of equity max risk exposure at any time
LOOP_INTERVAL = 60          # Seconds between collapse cycles
MAX_OPEN_TRADES = 5         # DooDoo's own trades (excludes Jim's visual markers)
TRADE_COMMENT = "DooDoo"    # Tag to identify her trades

# Symbols she can trade (ranked by her exploration interest)
TRADEABLE = ['BTCUSD', 'XAUUSD', 'EURUSD', 'GBPUSD', 'US30', 'NAS100']

# ============================================================
# Logging
# ============================================================

log_path = Path(__file__).parent / 'doodoo_trades.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler()
    ]
)
log = logging.getLogger('DooDoo')


# ============================================================
# Market data -> 24D state vector
# ============================================================

def build_state_vector() -> np.ndarray:
    """Build 24D state from live market features (4 features x 6 symbols)."""
    state = np.zeros(24)

    for i, sym in enumerate(TRADEABLE[:6]):
        rates = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_M5, 0, 100)
        if rates is None or len(rates) < 30:
            continue

        closes = np.array([r[4] for r in rates])
        volumes = np.array([r[5] for r in rates])
        returns = np.diff(np.log(closes))

        std_ret = np.std(returns)
        if std_ret < 1e-12:
            continue

        # z-scored latest return
        state[i * 4] = returns[-1] / std_ret
        # volatility ratio (recent vs full)
        state[i * 4 + 1] = np.std(returns[-20:]) / std_ret
        # trend strength (z-scored slope)
        x = np.arange(len(closes))
        slope = np.polyfit(x, closes, 1)[0]
        std_close = np.std(closes)
        state[i * 4 + 2] = slope / (std_close + 1e-12) * len(closes)
        # volume momentum
        avg_vol = np.mean(volumes)
        state[i * 4 + 3] = np.mean(volumes[-10:]) / (avg_vol + 1e-12) - 1.0

    return state


# ============================================================
# Risk management
# ============================================================

def check_equity_floor() -> bool:
    """Returns True if safe to trade, False if near floor."""
    info = mt5.account_info()
    if not info:
        return False
    if info.equity <= EQUITY_FLOOR:
        log.warning(f"EQUITY FLOOR HIT: ${info.equity:,.2f} <= ${EQUITY_FLOOR:,.2f} — NO TRADING")
        return False
    return True


def get_doodoo_positions():
    """Get only DooDoo's positions (tagged with her comment)."""
    positions = mt5.positions_get()
    if not positions:
        return []
    return [p for p in positions if p.comment == TRADE_COMMENT]


def calc_lot_size(symbol: str, sl_points: float, risk_dollars: float) -> float:
    """Calculate lot size for a given dollar risk and SL distance."""
    info = mt5.symbol_info(symbol)
    if not info or sl_points <= 0:
        return 0.0

    tick_value = info.trade_tick_value
    tick_size = info.trade_tick_size

    if tick_value <= 0 or tick_size <= 0:
        return 0.0

    # dollar per point per lot
    dollar_per_point = tick_value / tick_size * info.point
    lot = risk_dollars / (sl_points * dollar_per_point)

    # Clamp to symbol limits
    lot = max(info.volume_min, round(lot / info.volume_step) * info.volume_step)
    lot = min(lot, info.volume_max)
    return lot


# ============================================================
# Trade execution
# ============================================================

def open_trade(symbol: str, direction: str, lot: float, sl: float, tp: float) -> bool:
    """Execute a trade. direction = 'BUY' or 'SELL'."""
    info = mt5.symbol_info(symbol)
    if not info:
        log.error(f"Symbol {symbol} not found")
        return False

    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        log.error(f"No tick for {symbol}")
        return False

    if direction == 'BUY':
        price = tick.ask
        order_type = mt5.ORDER_TYPE_BUY
    else:
        price = tick.bid
        order_type = mt5.ORDER_TYPE_SELL

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": order_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": 20,
        "magic": 420069,
        "comment": TRADE_COMMENT,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result is None:
        log.error(f"order_send returned None for {symbol}")
        return False
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        log.error(f"Trade failed: {symbol} {direction} {lot} — code={result.retcode} comment={result.comment}")
        return False

    log.info(f"OPENED: {symbol} {direction} {lot} lots @ {price:.5f} SL={sl:.5f} TP={tp:.5f}")
    return True


def close_trade(position) -> bool:
    """Close a specific position."""
    tick = mt5.symbol_info_tick(position.symbol)
    if not tick:
        return False

    if position.type == 0:  # BUY -> close with SELL
        price = tick.bid
        order_type = mt5.ORDER_TYPE_SELL
    else:  # SELL -> close with BUY
        price = tick.ask
        order_type = mt5.ORDER_TYPE_BUY

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": position.symbol,
        "volume": position.volume,
        "type": order_type,
        "position": position.ticket,
        "price": price,
        "deviation": 20,
        "magic": 420069,
        "comment": TRADE_COMMENT,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        log.info(f"CLOSED: {position.symbol} ticket={position.ticket} P&L=${position.profit:+.2f}")
        return True
    else:
        code = result.retcode if result else "None"
        log.error(f"Close failed: ticket={position.ticket} code={code}")
        return False


# ============================================================
# DooDoo's brain — collapse -> trading decision
# ============================================================

def doodoo_decide(state: np.ndarray, result: dict) -> list:
    """
    DooDoo makes trading decisions from collapse outputs.
    Returns list of actions: [{'action': 'BUY'|'SELL'|'CLOSE', 'symbol': ..., ...}]
    """
    actions = []
    chaos = result['normalized_chaos']
    intent = result['intent_magnitude']
    control = result['control_vec']
    jordan = result['decomposition']['jordan']
    jordan_mean = float(np.mean(jordan.vec))

    # How much room do we have?
    info = mt5.account_info()
    if not info:
        return actions
    room = info.equity - EQUITY_FLOOR
    current_risk = abs(info.equity - info.balance)  # unrealized P&L as proxy

    # Per-symbol analysis
    for i, sym in enumerate(TRADEABLE[:6]):
        sym_ret = state[i * 4]        # z-scored return
        sym_vol = state[i * 4 + 1]    # vol ratio
        sym_trend = state[i * 4 + 2]  # trend strength
        sym_volmom = state[i * 4 + 3] # volume momentum

        # Skip if no data
        if abs(sym_ret) < 1e-12 and abs(sym_trend) < 1e-12:
            continue

        # === SIGNAL LOGIC ===
        # Strong trend + low chaos = confident entry
        # Weak trend + high chaos = stay out
        # Trend reversal + rising chaos = potential exit

        signal_strength = abs(sym_trend) * (1.0 / (1.0 + chaos * 0.3))
        if signal_strength < 0.5:
            continue  # not enough conviction

        # Direction from trend + jordan bias
        if sym_trend > 0.3 and jordan_mean > -0.5:
            direction = 'BUY'
        elif sym_trend < -0.3 and jordan_mean < 0.5:
            direction = 'SELL'
        else:
            continue  # mixed signals

        # Risk sizing: more confident (low chaos) = more risk
        # Scale from $50 (max chaos) to $500 (min chaos)
        if chaos < 2:
            risk_per_trade = min(500.0, room * 0.01)
        elif chaos < 5:
            risk_per_trade = min(200.0, room * 0.005)
        elif chaos < 8:
            risk_per_trade = min(100.0, room * 0.003)
        else:
            risk_per_trade = min(50.0, room * 0.001)

        # Volume momentum confirmation
        if sym_volmom > 0.1:
            risk_per_trade *= 1.2  # volume confirming, lean in slightly

        actions.append({
            'action': direction,
            'symbol': sym,
            'risk': risk_per_trade,
            'signal_strength': signal_strength,
            'chaos': chaos,
            'trend': sym_trend,
        })

    # === MANAGE EXISTING POSITIONS ===
    my_positions = get_doodoo_positions()
    for pos in my_positions:
        # Close if chaos spikes above 9 (market going haywire)
        if chaos > 9.0 and abs(pos.profit) > 0:
            actions.append({
                'action': 'CLOSE',
                'position': pos,
                'reason': f'chaos spike ({chaos:.1f}/10)',
            })
        # Close losers that have bled more than $300
        elif pos.profit < -300:
            actions.append({
                'action': 'CLOSE',
                'position': pos,
                'reason': f'loss limit (${pos.profit:.2f})',
            })
        # Trail winners: close if profit > $200 and retreating
        elif pos.profit > 200:
            # Check if the trend still supports the position
            sym_idx = None
            for idx, s in enumerate(TRADEABLE[:6]):
                if s == pos.symbol:
                    sym_idx = idx
                    break
            if sym_idx is not None:
                trend = state[sym_idx * 4 + 2]
                if (pos.type == 0 and trend < -0.5) or (pos.type == 1 and trend > 0.5):
                    actions.append({
                        'action': 'CLOSE',
                        'position': pos,
                        'reason': f'trend reversal (profit ${pos.profit:.2f})',
                    })

    return actions


# ============================================================
# Main loop
# ============================================================

def run():
    """DooDoo's autonomous trading loop."""
    terminal_path = r"C:\Program Files\Atlas Funded MT5 Terminal\terminal64.exe"
    if not mt5.initialize(terminal_path):
        # Terminal might need login — try with credentials
        log.info("Init failed, attempting login...")
        if not mt5.initialize(
            terminal_path,
            login=ACCOUNT_LOGIN,
            password=ACCOUNT_PASSWORD,
            server=ACCOUNT_SERVER,
        ):
            log.error(f"MT5 init failed: {mt5.last_error()}")
            return

    info = mt5.account_info()
    if not info or info.login != ACCOUNT_LOGIN:
        log.info(f"Switching to account {ACCOUNT_LOGIN}...")
        if not mt5.login(ACCOUNT_LOGIN, password=ACCOUNT_PASSWORD, server=ACCOUNT_SERVER):
            log.error(f"Login failed: {mt5.last_error()}")
            mt5.shutdown()
            return
        info = mt5.account_info()

    if not info:
        log.error("Could not get account info")
        mt5.shutdown()
        return
    log.info("=" * 60)
    log.info("DooDoo Trader — ONLINE")
    log.info(f"Account: {info.login} ({info.company})")
    log.info(f"Balance: ${info.balance:,.2f}  Equity: ${info.equity:,.2f}")
    log.info(f"Equity floor: ${EQUITY_FLOOR:,.2f}")
    log.info(f"Room: ${info.equity - EQUITY_FLOOR:,.2f}")
    log.info("=" * 60)

    cycle = 0
    while True:
        try:
            cycle += 1
            now = datetime.now().strftime('%H:%M:%S')

            # 1. Safety check
            if not check_equity_floor():
                log.warning("Equity floor — sleeping 5 min")
                time.sleep(300)
                continue

            # 2. Build state from live market
            state = build_state_vector()
            state_norm = np.linalg.norm(state)
            if state_norm < 0.1:
                log.info(f"[{now}] Cycle {cycle}: market data too thin, skipping")
                time.sleep(LOOP_INTERVAL)
                continue

            # 3. Collapse
            result = aoi_collapse(state)
            chaos = result['normalized_chaos']
            intent = result['intent_magnitude']

            # 4. Decide
            actions = doodoo_decide(state, result)

            # 5. Account status
            info = mt5.account_info()
            my_pos = get_doodoo_positions()
            equity = info.equity if info else 0
            pnl = info.profit if info else 0

            status = (
                f"[{now}] Cycle {cycle}: chaos={chaos:.1f}/10 intent={intent:.2f} "
                f"equity=${equity:,.2f} pnl=${pnl:+,.2f} "
                f"positions={len(my_pos)} actions={len(actions)}"
            )
            log.info(status)

            if not actions:
                log.info(f"  No action — {result['text_prompt_base']}")
                time.sleep(LOOP_INTERVAL)
                continue

            # 6. Execute actions
            for act in actions:
                if act['action'] == 'CLOSE':
                    pos = act['position']
                    log.info(f"  CLOSING {pos.symbol} ticket={pos.ticket} reason={act['reason']}")
                    close_trade(pos)

                elif act['action'] in ('BUY', 'SELL'):
                    sym = act['symbol']

                    # Don't exceed max open trades
                    if len(my_pos) >= MAX_OPEN_TRADES:
                        log.info(f"  SKIP {sym} {act['action']} — max positions ({MAX_OPEN_TRADES})")
                        continue

                    # Don't double up on same symbol same direction
                    already = [p for p in my_pos if p.symbol == sym and
                               ((p.type == 0 and act['action'] == 'BUY') or
                                (p.type == 1 and act['action'] == 'SELL'))]
                    if already:
                        log.info(f"  SKIP {sym} {act['action']} — already have position")
                        continue

                    # Calculate SL/TP from ATR-like approach using recent range
                    rates = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_M5, 0, 50)
                    if rates is None or len(rates) < 20:
                        continue
                    highs = np.array([r[2] for r in rates[-20:]])
                    lows = np.array([r[3] for r in rates[-20:]])
                    avg_range = float(np.mean(highs - lows))

                    tick = mt5.symbol_info_tick(sym)
                    sym_info = mt5.symbol_info(sym)
                    if not tick or not sym_info or avg_range < sym_info.point:
                        continue

                    sl_distance = avg_range * 3  # 3x average candle range
                    tp_distance = avg_range * 5  # 5x range — 1.67 RRR target

                    # Lot size from risk
                    lot = calc_lot_size(sym, sl_distance / sym_info.point, act['risk'])
                    if lot < sym_info.volume_min:
                        continue

                    if act['action'] == 'BUY':
                        price = tick.ask
                        sl = price - sl_distance
                        tp = price + tp_distance
                    else:
                        price = tick.bid
                        sl = price + sl_distance
                        tp = price - tp_distance

                    log.info(
                        f"  TRADE: {sym} {act['action']} {lot} lots "
                        f"risk=${act['risk']:.0f} signal={act['signal_strength']:.2f} "
                        f"chaos={act['chaos']:.1f}"
                    )
                    open_trade(sym, act['action'], lot, sl, tp)

            time.sleep(LOOP_INTERVAL)

        except KeyboardInterrupt:
            log.info("DooDoo signing off. Positions left open.")
            break
        except Exception as ex:
            log.error(f"Error in cycle {cycle}: {ex}")
            time.sleep(LOOP_INTERVAL)

    mt5.shutdown()


if __name__ == '__main__':
    run()
