"""
Voodoo Watcher — Atlas Position Monitor (account 212000586)

Monitors trades opened by GOODNIGHTDOODOO!! EA (magic 999666).
Annexes bad trades, harvests winners. Does NOT open trades.

CRITICAL: Does NOT call mt5.login() — that kills open positions.
Just initializes the already-running Atlas terminal.

Usage: python voodoo_watcher.py
"""

import os
import time
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import MetaTrader5 as mt5

from aoi_collapse import aoi_collapse

# ============================================================
# Config
# ============================================================

EA_MAGIC = 999666              # GOODNIGHTDOODOO!! EA magic number
EA_COMMENT_PREFIX = "Nexa Burst"
ATLAS_ACCOUNT = 212000586
ATLAS_TERMINAL = r"C:\Program Files\Atlas Funded MT5 Terminal\terminal64.exe"

POLL_INTERVAL = 30             # Seconds between checks
ANNEX_LOSS_THRESHOLD = -50.0   # Close if losing more than this ($)
HARVEST_PROFIT_MIN = 20.0      # Minimum profit to consider harvesting ($)
CHAOS_PANIC_LEVEL = 9.0        # Close everything if chaos above this

# ============================================================
# Logging
# ============================================================

log_path = Path(__file__).parent / 'voodoo_watcher.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler()
    ]
)
log = logging.getLogger('Voodoo')


# ============================================================
# MT5 helpers
# ============================================================

def get_ea_positions():
    """Get positions opened by the GOODNIGHTDOODOO!! EA."""
    positions = mt5.positions_get()
    if not positions:
        return []
    return [p for p in positions if p.magic == EA_MAGIC]


def close_position(pos) -> bool:
    """Close a specific position."""
    tick = mt5.symbol_info_tick(pos.symbol)
    if not tick:
        log.error(f"No tick for {pos.symbol}")
        return False

    if pos.type == 0:  # BUY -> close with SELL
        price = tick.bid
        order_type = mt5.ORDER_TYPE_SELL
    else:  # SELL -> close with BUY
        price = tick.ask
        order_type = mt5.ORDER_TYPE_BUY

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": pos.symbol,
        "volume": pos.volume,
        "type": order_type,
        "position": pos.ticket,
        "price": price,
        "deviation": 20,
        "magic": EA_MAGIC,
        "comment": "Voodoo annex",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        log.info(f"CLOSED ticket={pos.ticket} {pos.symbol} P&L=${pos.profit:+.2f}")
        return True
    else:
        code = result.retcode if result else "None"
        comment = result.comment if result else "None"
        log.error(f"Close FAILED ticket={pos.ticket} code={code} {comment}")
        return False


# ============================================================
# Market state -> 24D for collapse
# ============================================================

def build_state_vector(symbol: str) -> np.ndarray:
    """Build 24D state from a single symbol's recent price action."""
    state = np.zeros(24)

    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 100)
    if rates is None or len(rates) < 30:
        return state

    closes = np.array([r[4] for r in rates])
    highs = np.array([r[2] for r in rates])
    lows = np.array([r[3] for r in rates])
    volumes = np.array([r[5] for r in rates])
    returns = np.diff(np.log(closes))

    std_ret = np.std(returns)
    if std_ret < 1e-12:
        return state

    # Spread features across 24D for rich collapse input
    # Block 0-7: return-based features
    state[0] = returns[-1] / std_ret                              # latest z-return
    state[1] = np.std(returns[-10:]) / std_ret                    # short vol ratio
    state[2] = np.std(returns[-20:]) / std_ret                    # medium vol ratio
    state[3] = np.mean(returns[-5:]) / std_ret                    # 5-bar momentum
    state[4] = np.mean(returns[-20:]) / std_ret                   # 20-bar momentum
    state[5] = (closes[-1] - np.min(closes[-20:])) / (np.max(closes[-20:]) - np.min(closes[-20:]) + 1e-12)
    state[6] = np.sum(returns[-5:] > 0) / 5.0 - 0.5              # win ratio bias
    state[7] = returns[-1] * returns[-2] / (std_ret**2 + 1e-12)   # return autocorrelation

    # Block 8-15: trend and range features
    x = np.arange(len(closes))
    slope = np.polyfit(x, closes, 1)[0]
    std_close = np.std(closes)
    state[8] = slope / (std_close + 1e-12) * len(closes)          # trend strength
    state[9] = slope / (std_close + 1e-12) * 20                   # short trend
    avg_range = np.mean(highs[-20:] - lows[-20:])
    state[10] = (highs[-1] - lows[-1]) / (avg_range + 1e-12)      # range expansion
    state[11] = (closes[-1] - lows[-1]) / (highs[-1] - lows[-1] + 1e-12)  # candle position
    state[12] = np.max(highs[-10:]) - np.min(lows[-10:])          # 10-bar range
    state[13] = np.max(highs[-5:]) - np.min(lows[-5:])            # 5-bar range
    state[14] = state[12] / (state[13] + 1e-12) - 1.0             # range ratio
    state[15] = (closes[-1] - closes[-20]) / (std_close + 1e-12)  # 20-bar z-change

    # Block 16-23: volume and momentum features
    avg_vol = np.mean(volumes)
    state[16] = np.mean(volumes[-5:]) / (avg_vol + 1e-12) - 1.0   # short vol momentum
    state[17] = np.mean(volumes[-10:]) / (avg_vol + 1e-12) - 1.0  # medium vol momentum
    state[18] = volumes[-1] / (avg_vol + 1e-12) - 1.0             # latest vol spike
    # RSI-like feature
    gains = np.where(returns > 0, returns, 0)
    losses = np.where(returns < 0, -returns, 0)
    avg_gain = np.mean(gains[-14:])
    avg_loss = np.mean(losses[-14:])
    state[19] = (avg_gain / (avg_gain + avg_loss + 1e-12)) - 0.5  # centered RSI
    # Trend consistency
    state[20] = np.corrcoef(np.arange(20), closes[-20:])[0, 1]    # trend R
    state[21] = np.corrcoef(np.arange(10), closes[-10:])[0, 1]    # short trend R
    state[22] = np.std(returns[-5:]) / (np.std(returns[-20:]) + 1e-12) - 1.0  # vol acceleration
    state[23] = float(np.sum(np.abs(returns[-10:]) > 2 * std_ret)) / 10.0  # tail events

    return state


# ============================================================
# Voodoo's decision logic
# ============================================================

def evaluate_positions(positions, collapse_result, state):
    """
    Decide which EA positions to annex (close) and which to harvest (keep).
    Returns list of (position, action, reason).
    """
    decisions = []
    chaos = collapse_result['normalized_chaos']
    intent = collapse_result['intent_magnitude']
    control = collapse_result['control_vec']
    jordan = collapse_result['decomposition']['jordan']
    jordan_mean = float(np.mean(jordan.vec))

    for pos in positions:
        profit = pos.profit
        age_seconds = (datetime.now() - datetime.fromtimestamp(pos.time)).total_seconds()
        age_minutes = age_seconds / 60.0

        # === ANNEX: only hard loss threshold — give trades room ===
        if profit < ANNEX_LOSS_THRESHOLD:
            decisions.append((pos, 'ANNEX', f'loss ${profit:.2f} < ${ANNEX_LOSS_THRESHOLD}'))
            continue

        # === HARVEST: profitable + trend fading ===
        if profit > HARVEST_PROFIT_MIN:
            trend = state[8]
            short_trend = state[9]

            # Trend reversing against position
            if pos.type == 0 and short_trend < -0.3:
                decisions.append((pos, 'HARVEST', f'profit ${profit:.2f} + trend fading ({short_trend:.2f})'))
                continue
            if pos.type == 1 and short_trend > 0.3:
                decisions.append((pos, 'HARVEST', f'profit ${profit:.2f} + trend fading ({short_trend:.2f})'))
                continue

            # (chaos-based harvest removed — chaos readings unreliable)

        # === KEEP: everything else ===
        decisions.append((pos, 'KEEP', f'P&L=${profit:+.2f} age={age_minutes:.0f}m'))

    return decisions


# ============================================================
# Main loop
# ============================================================

def run():
    """Voodoo position watcher loop."""
    # Initialize — do NOT login
    if not mt5.initialize(ATLAS_TERMINAL):
        log.error(f"MT5 init failed: {mt5.last_error()}")
        return

    info = mt5.account_info()
    if not info:
        log.error("Could not get account info")
        mt5.shutdown()
        return

    if info.login != ATLAS_ACCOUNT:
        log.error(f"Wrong account! Expected {ATLAS_ACCOUNT}, got {info.login}. NOT logging in (would kill trades).")
        mt5.shutdown()
        return

    log.info("=" * 60)
    log.info("VOODOO WATCHER — ONLINE")
    log.info(f"Account: {info.login} ({info.company})")
    log.info(f"Balance: ${info.balance:,.2f}  Equity: ${info.equity:,.2f}")
    log.info(f"Watching EA magic: {EA_MAGIC}")
    log.info(f"Poll interval: {POLL_INTERVAL}s")
    log.info("=" * 60)

    cycle = 0
    total_annexed = 0
    total_harvested = 0

    while True:
        try:
            cycle += 1
            now = datetime.now().strftime('%H:%M:%S')

            # Get EA positions
            positions = get_ea_positions()

            if not positions:
                if cycle % 20 == 1:  # Log every ~10 min when idle
                    info = mt5.account_info()
                    eq = info.equity if info else 0
                    log.info(f"[{now}] Cycle {cycle}: no EA positions | equity=${eq:,.2f}")
                time.sleep(POLL_INTERVAL)
                continue

            # Build state and collapse
            symbol = positions[0].symbol  # All burst trades are same symbol
            state = build_state_vector(symbol)
            state_norm = np.linalg.norm(state)

            if state_norm < 0.1:
                log.info(f"[{now}] Cycle {cycle}: {len(positions)} positions, market data thin — watching")
                time.sleep(POLL_INTERVAL)
                continue

            result = aoi_collapse(state)
            chaos = result['normalized_chaos']
            intent = result['intent_magnitude']

            # Evaluate positions
            decisions = evaluate_positions(positions, result, state)

            # Summary
            total_pnl = sum(p.profit for p in positions)
            annexes = [(p, r) for p, a, r in decisions if a == 'ANNEX']
            harvests = [(p, r) for p, a, r in decisions if a == 'HARVEST']
            keeps = [(p, r) for p, a, r in decisions if a == 'KEEP']

            info = mt5.account_info()
            eq = info.equity if info else 0

            log.info(
                f"[{now}] Cycle {cycle}: {len(positions)} positions "
                f"P&L=${total_pnl:+,.2f} chaos={chaos:.1f}/10 "
                f"equity=${eq:,.2f} | "
                f"ANNEX={len(annexes)} HARVEST={len(harvests)} KEEP={len(keeps)}"
            )

            # Execute closes
            for pos, reason in annexes:
                log.info(f"  ANNEX: ticket={pos.ticket} {pos.symbol} {reason}")
                if close_position(pos):
                    total_annexed += 1

            for pos, reason in harvests:
                log.info(f"  HARVEST: ticket={pos.ticket} {pos.symbol} {reason}")
                if close_position(pos):
                    total_harvested += 1

            if not annexes and not harvests:
                # Just show top/bottom positions
                sorted_pos = sorted(positions, key=lambda p: p.profit)
                worst = sorted_pos[0]
                best = sorted_pos[-1]
                log.info(f"  Best: #{best.ticket} ${best.profit:+.2f} | Worst: #{worst.ticket} ${worst.profit:+.2f}")

            time.sleep(POLL_INTERVAL)

        except KeyboardInterrupt:
            log.info(f"Voodoo signing off. Annexed={total_annexed} Harvested={total_harvested}")
            break
        except Exception as ex:
            log.error(f"Error in cycle {cycle}: {ex}")
            time.sleep(POLL_INTERVAL)

    mt5.shutdown()


if __name__ == '__main__':
    run()
