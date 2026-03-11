"""
MT5 Bridge — Layer 1: FastAPI REST server wrapping MetaTrader5 COM calls.
=========================================================================
Runs natively on Windows (cannot be containerized — MT5 needs COM access).
The Dockerized MCP server (Layer 2) connects to this bridge over HTTP.

Run:  python mt5_bridge.py
      Or: uvicorn mt5_bridge:app --host 0.0.0.0 --port 8787

Environment:
  MT5_BRIDGE_PORT=8787          (default)
  MT5_BRIDGE_API_KEY=<secret>   (optional, for auth between bridge and MCP server)
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# Add parent dir so we can import config_loader
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import MetaTrader5 as mt5

try:
    from config_loader import ACCOUNTS, MAX_LOSS_DOLLARS, AGENT_SL_MAX
except ImportError:
    ACCOUNTS = {}
    MAX_LOSS_DOLLARS = 1.00
    AGENT_SL_MAX = 1.00

# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
API_KEY = os.environ.get("MT5_BRIDGE_API_KEY", "")


def verify_api_key(x_api_key: Optional[str] = Header(None)):
    """If API_KEY is set, require it on every request."""
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="MT5 Bridge", version="1.0.0")

# Known magic numbers from the trading system
KNOWN_MAGIC = {
    212001, 366001, 365001, 113001, 113002, 107001,
    152001, 151201, 888888, 999999, 20251222, 20251227,
}

connected_account: Optional[int] = None


# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------
class ConnectRequest(BaseModel):
    account_key: Optional[str] = None


class CloseRequest(BaseModel):
    ticket: int


class ModifySLTPRequest(BaseModel):
    ticket: int
    sl: Optional[float] = None
    tp: Optional[float] = None


class ForceSLRequest(BaseModel):
    ticket: int
    max_loss_dollars: float = 2.0


class CloseLosersRequest(BaseModel):
    max_loss: Optional[float] = None


class HistoryRequest(BaseModel):
    days: int = 1


class OHLCVRequest(BaseModel):
    symbol: str
    timeframe: str = "H1"
    count: int = 100


class OrderSendRequest(BaseModel):
    symbol: str
    order_type: str  # "BUY" or "SELL"
    volume: float
    sl: Optional[float] = None
    tp: Optional[float] = None
    magic: int = 888888
    comment: str = "MCP_ORDER"
    confirmed: bool = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
TIMEFRAME_MAP = {
    "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1, "W1": mt5.TIMEFRAME_W1, "MN1": mt5.TIMEFRAME_MN1,
}


def _pos_to_dict(pos) -> dict:
    return {
        "ticket": pos.ticket,
        "symbol": pos.symbol,
        "type": "BUY" if pos.type == 0 else "SELL",
        "volume": pos.volume,
        "open_price": pos.price_open,
        "current_price": pos.price_current,
        "sl": pos.sl,
        "tp": pos.tp,
        "profit": pos.profit,
        "magic": pos.magic,
        "comment": pos.comment,
        "time": datetime.fromtimestamp(pos.time).isoformat(),
    }


def _get_filling_mode(symbol: str):
    info = mt5.symbol_info(symbol)
    if info and info.filling_mode & mt5.ORDER_FILLING_FOK:
        return mt5.ORDER_FILLING_FOK
    return mt5.ORDER_FILLING_IOC


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    """Health check — also reports whether MT5 is connected."""
    try:
        info = mt5.account_info()
        if info:
            return {"status": "ok", "mt5_connected": True, "account": info.login}
    except Exception:
        pass
    return {"status": "ok", "mt5_connected": False}


@app.post("/connect", dependencies=[Depends(verify_api_key)])
async def connect(req: ConnectRequest):
    global connected_account

    account_key = req.account_key

    # Check if already connected
    try:
        existing = mt5.account_info()
        if existing and account_key and account_key in ACCOUNTS:
            if existing.login == ACCOUNTS[account_key]["account"]:
                connected_account = existing.login
                return {
                    "account": existing.login,
                    "balance": existing.balance,
                    "equity": existing.equity,
                    "profit": existing.profit,
                    "margin_free": existing.margin_free,
                    "note": "Already connected",
                }
    except Exception:
        pass

    if account_key and account_key in ACCOUNTS:
        acc = ACCOUNTS[account_key]
        terminal_path = acc.get("terminal_path")

        if connected_account is not None:
            mt5.shutdown()

        if terminal_path:
            if not mt5.initialize(path=terminal_path):
                raise HTTPException(500, f"Init failed: {mt5.last_error()}")
        else:
            if not mt5.initialize():
                raise HTTPException(500, f"Init failed: {mt5.last_error()}")

        pre_check = mt5.account_info()
        if pre_check and pre_check.login == acc["account"]:
            pass  # Already correct
        elif acc.get("password"):
            if not mt5.login(acc["account"], password=acc["password"], server=acc["server"]):
                raise HTTPException(500, f"Login failed: {mt5.last_error()}")
    elif account_key and account_key not in ACCOUNTS:
        valid = ", ".join(ACCOUNTS.keys()) if ACCOUNTS else "(none)"
        raise HTTPException(400, f"Unknown account_key '{account_key}'. Valid: {valid}")
    else:
        if not mt5.initialize():
            raise HTTPException(500, f"Init failed: {mt5.last_error()}")

    info = mt5.account_info()
    if info:
        connected_account = info.login
        return {
            "account": info.login,
            "balance": info.balance,
            "equity": info.equity,
            "profit": info.profit,
            "margin_free": info.margin_free,
        }
    raise HTTPException(500, "Could not get account info")


@app.get("/account", dependencies=[Depends(verify_api_key)])
async def account_info():
    info = mt5.account_info()
    if not info:
        raise HTTPException(503, "Not connected to MT5")
    return {
        "account": info.login,
        "balance": info.balance,
        "equity": info.equity,
        "profit": info.profit,
        "margin_free": info.margin_free,
        "margin": info.margin,
        "leverage": info.leverage,
        "currency": info.currency,
    }


@app.get("/positions", dependencies=[Depends(verify_api_key)])
async def positions():
    pos = mt5.positions_get()
    if pos is None:
        return []
    return [_pos_to_dict(p) for p in pos]


@app.get("/summary", dependencies=[Depends(verify_api_key)])
async def summary():
    info = mt5.account_info()
    if not info:
        raise HTTPException(503, "Not connected to MT5")

    pos = mt5.positions_get()
    pos_list = [_pos_to_dict(p) for p in pos] if pos else []
    total_profit = sum(p["profit"] for p in pos_list)
    winning = [p for p in pos_list if p["profit"] > 0]
    losing = [p for p in pos_list if p["profit"] < 0]

    return {
        "account": info.login,
        "balance": info.balance,
        "equity": info.equity,
        "floating_pl": total_profit,
        "margin_used": info.margin,
        "margin_free": info.margin_free,
        "positions_count": len(pos_list),
        "winning_count": len(winning),
        "losing_count": len(losing),
        "worst_position": min(pos_list, key=lambda x: x["profit"]) if pos_list else None,
        "best_position": max(pos_list, key=lambda x: x["profit"]) if pos_list else None,
        "positions": pos_list,
    }


@app.post("/close", dependencies=[Depends(verify_api_key)])
async def close_position(req: CloseRequest):
    positions = mt5.positions_get(ticket=req.ticket)
    if not positions:
        raise HTTPException(404, f"Position {req.ticket} not found")

    pos = positions[0]
    tick = mt5.symbol_info_tick(pos.symbol)
    if tick is None:
        raise HTTPException(500, f"Cannot get tick for {pos.symbol}")

    if pos.type == mt5.POSITION_TYPE_BUY:
        order_type = mt5.ORDER_TYPE_SELL
        price = tick.bid
    else:
        order_type = mt5.ORDER_TYPE_BUY
        price = tick.ask

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": pos.symbol,
        "volume": pos.volume,
        "type": order_type,
        "position": req.ticket,
        "price": price,
        "deviation": 50,
        "magic": 888888,
        "comment": "MCP_CLOSE",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": _get_filling_mode(pos.symbol),
    }

    result = mt5.order_send(request)
    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        return {"success": True, "ticket": req.ticket, "closed_profit": pos.profit}
    raise HTTPException(500, f"Close failed: {result.comment if result else 'None'} ({result.retcode if result else 'N/A'})")


@app.post("/modify_sltp", dependencies=[Depends(verify_api_key)])
async def modify_sltp(req: ModifySLTPRequest):
    positions = mt5.positions_get(ticket=req.ticket)
    if not positions:
        raise HTTPException(404, f"Position {req.ticket} not found")

    pos = positions[0]

    if req.sl == 0 or req.sl == 0.0:
        raise HTTPException(400, "Refusing to set SL=0 (removes stop loss)")

    request = {
        "action": mt5.TRADE_ACTION_SLTP,
        "symbol": pos.symbol,
        "position": req.ticket,
        "sl": req.sl if req.sl is not None else pos.sl,
        "tp": req.tp if req.tp is not None else pos.tp,
    }

    result = mt5.order_send(request)
    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        return {"success": True, "ticket": req.ticket, "new_sl": request["sl"], "new_tp": request["tp"]}
    raise HTTPException(500, f"Modify failed: {result.comment if result else 'None'}")


@app.post("/close_losers", dependencies=[Depends(verify_api_key)])
async def close_losers(req: CloseLosersRequest):
    max_loss = req.max_loss if req.max_loss is not None else AGENT_SL_MAX
    if max_loss > 10.0:
        raise HTTPException(400, f"max_loss={max_loss} exceeds safety cap of $10.00")

    positions = mt5.positions_get()
    if not positions:
        return {"closed": [], "count": 0}

    closed = []
    for pos in positions:
        if pos.profit < -max_loss:
            try:
                tick = mt5.symbol_info_tick(pos.symbol)
                if tick is None:
                    continue
                if pos.type == mt5.POSITION_TYPE_BUY:
                    ot, price = mt5.ORDER_TYPE_SELL, tick.bid
                else:
                    ot, price = mt5.ORDER_TYPE_BUY, tick.ask

                req_close = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": pos.symbol,
                    "volume": pos.volume,
                    "type": ot,
                    "position": pos.ticket,
                    "price": price,
                    "deviation": 50,
                    "magic": 888888,
                    "comment": "MCP_CLOSE_LOSER",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": _get_filling_mode(pos.symbol),
                }
                result = mt5.order_send(req_close)
                closed.append({
                    "ticket": pos.ticket,
                    "symbol": pos.symbol,
                    "profit": pos.profit,
                    "success": result and result.retcode == mt5.TRADE_RETCODE_DONE,
                })
            except Exception as e:
                closed.append({"ticket": pos.ticket, "error": str(e)})

    return {"closed": closed, "count": len(closed)}


@app.get("/scan_no_sl", dependencies=[Depends(verify_api_key)])
async def scan_no_sl():
    positions = mt5.positions_get()
    if positions is None:
        raise HTTPException(503, "Could not get positions")

    no_sl, no_tp, rogue = [], [], []
    for pos in positions:
        entry = _pos_to_dict(pos)
        if pos.sl == 0.0:
            no_sl.append(entry)
        if pos.tp == 0.0:
            no_tp.append(entry)
        if pos.magic not in KNOWN_MAGIC:
            rogue.append(entry)

    return {
        "total_positions": len(positions),
        "missing_sl": len(no_sl),
        "missing_tp": len(no_tp),
        "rogue_magic": len(rogue),
        "positions_without_sl": no_sl,
        "positions_without_tp": no_tp,
        "rogue_trades": rogue,
    }


@app.post("/force_sl", dependencies=[Depends(verify_api_key)])
async def force_sl(req: ForceSLRequest):
    if req.max_loss_dollars > 5.0:
        raise HTTPException(400, f"max_loss_dollars={req.max_loss_dollars} exceeds safety cap of $5.00")

    positions = mt5.positions_get(ticket=req.ticket)
    if not positions:
        raise HTTPException(404, f"Position {req.ticket} not found")

    pos = positions[0]
    if pos.sl != 0.0:
        return {"info": f"Position already has SL={pos.sl}", "current_sl": pos.sl}

    symbol_info = mt5.symbol_info(pos.symbol)
    tick = mt5.symbol_info_tick(pos.symbol)
    if symbol_info is None or tick is None:
        raise HTTPException(500, f"Cannot get symbol info for {pos.symbol}")

    tick_value = symbol_info.trade_tick_value
    tick_size = symbol_info.trade_tick_size
    point = symbol_info.point
    digits = symbol_info.digits
    stops_level = symbol_info.trade_stops_level
    spread = symbol_info.spread

    if tick_value > 0 and pos.volume > 0:
        sl_ticks = req.max_loss_dollars / (tick_value * pos.volume)
        sl_distance = sl_ticks * tick_size
    else:
        sl_distance = 200 * point

    min_sl_distance = (stops_level + spread + 20) * point
    if sl_distance < min_sl_distance:
        sl_distance = min_sl_distance

    if pos.type == mt5.POSITION_TYPE_BUY:
        current = tick.bid
        sl_price = round(current - sl_distance, digits)
    else:
        current = tick.ask
        sl_price = round(current + sl_distance, digits)

    request = {
        "action": mt5.TRADE_ACTION_SLTP,
        "symbol": pos.symbol,
        "position": req.ticket,
        "sl": sl_price,
        "tp": pos.tp,
    }

    result = mt5.order_send(request)
    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        return {
            "success": True,
            "ticket": req.ticket,
            "new_sl": sl_price,
            "current_price": current,
            "sl_distance": round(sl_distance, digits),
            "max_further_loss": req.max_loss_dollars,
        }
    raise HTTPException(500, f"SL modify failed: {result.comment if result else 'None'} ({result.retcode if result else 'N/A'})")


@app.post("/history", dependencies=[Depends(verify_api_key)])
async def history(req: HistoryRequest):
    days = max(1, min(req.days, 90))
    now = datetime.now()
    from_date = now - timedelta(days=days)

    deals = mt5.history_deals_get(from_date, now)
    if deals is None:
        return []

    result = []
    for deal in deals:
        if deal.entry == mt5.DEAL_ENTRY_OUT:
            result.append({
                "ticket": deal.ticket,
                "order": deal.order,
                "symbol": deal.symbol,
                "type": "BUY" if deal.type == 0 else "SELL",
                "volume": deal.volume,
                "price": deal.price,
                "profit": deal.profit,
                "commission": deal.commission,
                "swap": deal.swap,
                "magic": deal.magic,
                "comment": deal.comment,
                "time": datetime.fromtimestamp(deal.time).isoformat(),
            })

    return sorted(result, key=lambda x: x["time"], reverse=True)


@app.get("/symbol/{symbol}", dependencies=[Depends(verify_api_key)])
async def symbol_info(symbol: str):
    info = mt5.symbol_info(symbol)
    if info is None:
        raise HTTPException(404, f"Symbol {symbol} not found")

    tick = mt5.symbol_info_tick(symbol)
    return {
        "symbol": info.name,
        "bid": tick.bid if tick else None,
        "ask": tick.ask if tick else None,
        "spread": info.spread,
        "digits": info.digits,
        "point": info.point,
        "trade_tick_value": info.trade_tick_value,
        "trade_tick_size": info.trade_tick_size,
        "volume_min": info.volume_min,
        "volume_max": info.volume_max,
        "volume_step": info.volume_step,
        "trade_stops_level": info.trade_stops_level,
    }


@app.post("/ohlcv", dependencies=[Depends(verify_api_key)])
async def ohlcv(req: OHLCVRequest):
    tf = TIMEFRAME_MAP.get(req.timeframe.upper())
    if tf is None:
        raise HTTPException(400, f"Unknown timeframe '{req.timeframe}'. Valid: {list(TIMEFRAME_MAP.keys())}")

    count = max(1, min(req.count, 1000))
    rates = mt5.copy_rates_from_pos(req.symbol, tf, 0, count)
    if rates is None or len(rates) == 0:
        raise HTTPException(404, f"No data for {req.symbol} {req.timeframe}")

    return [
        {
            "time": datetime.fromtimestamp(r[0]).isoformat(),
            "open": float(r[1]),
            "high": float(r[2]),
            "low": float(r[3]),
            "close": float(r[4]),
            "tick_volume": int(r[5]),
            "spread": int(r[6]),
            "real_volume": int(r[7]),
        }
        for r in rates
    ]


@app.post("/order_send", dependencies=[Depends(verify_api_key)])
async def order_send(req: OrderSendRequest):
    if not req.confirmed:
        raise HTTPException(
            400,
            "Safety: set confirmed=true to place this order. "
            "This prevents accidental trades.",
        )

    tick = mt5.symbol_info_tick(req.symbol)
    if tick is None:
        raise HTTPException(500, f"Cannot get tick for {req.symbol}")

    symbol_info = mt5.symbol_info(req.symbol)
    if symbol_info is None:
        raise HTTPException(500, f"Cannot get symbol info for {req.symbol}")

    # Validate SL is within safety limits
    if req.sl is not None and req.sl != 0:
        tick_value = symbol_info.trade_tick_value
        tick_size = symbol_info.trade_tick_size
        if tick_value > 0 and tick_size > 0:
            if req.order_type.upper() == "BUY":
                sl_distance = tick.ask - req.sl
            else:
                sl_distance = req.sl - tick.bid
            sl_ticks = sl_distance / tick_size
            potential_loss = sl_ticks * tick_value * req.volume
            if potential_loss > MAX_LOSS_DOLLARS * 2:
                raise HTTPException(
                    400,
                    f"SL would risk ${potential_loss:.2f}, exceeding 2x MAX_LOSS (${MAX_LOSS_DOLLARS * 2:.2f}). "
                    "Tighten SL or reduce volume.",
                )

    if req.order_type.upper() == "BUY":
        ot = mt5.ORDER_TYPE_BUY
        price = tick.ask
    elif req.order_type.upper() == "SELL":
        ot = mt5.ORDER_TYPE_SELL
        price = tick.bid
    else:
        raise HTTPException(400, f"order_type must be BUY or SELL, got '{req.order_type}'")

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": req.symbol,
        "volume": req.volume,
        "type": ot,
        "price": price,
        "sl": req.sl or 0.0,
        "tp": req.tp or 0.0,
        "deviation": 50,
        "magic": req.magic,
        "comment": req.comment,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": _get_filling_mode(req.symbol),
    }

    result = mt5.order_send(request)
    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        return {
            "success": True,
            "order": result.order,
            "deal": result.deal,
            "volume": result.volume,
            "price": result.price,
        }
    raise HTTPException(
        500,
        f"Order failed: {result.comment if result else 'None'} ({result.retcode if result else 'N/A'})",
    )


@app.get("/accounts", dependencies=[Depends(verify_api_key)])
async def list_accounts():
    """List available account keys (no passwords exposed)."""
    return {
        k: {
            "account": v["account"],
            "server": v["server"],
            "enabled": v.get("enabled", False),
            "symbols": v.get("symbols", []),
        }
        for k, v in ACCOUNTS.items()
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("MT5_BRIDGE_PORT", "8787"))
    print(f"MT5 Bridge starting on port {port}...")
    print(f"Accounts loaded: {list(ACCOUNTS.keys())}")
    uvicorn.run(app, host="0.0.0.0", port=port)
