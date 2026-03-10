"""
MT5 MCP Server — Layer 2: Dockerized MCP server for Claude.
============================================================
Connects to the Layer 1 FastAPI bridge over HTTP.
This container is portable — anyone with a running bridge can use it.

Vendor: com.lattice24/mt5-trading
Capabilities: account, positions, orders, market_data, history, safety

Environment:
  MT5_BRIDGE_URL=http://host.docker.internal:8787  (default)
  MT5_BRIDGE_API_KEY=<secret>                       (optional)
"""

import os
import json
import sys
import asyncio
from typing import Any

import httpx

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BRIDGE_URL = os.environ.get("MT5_BRIDGE_URL", "http://host.docker.internal:8787")
API_KEY = os.environ.get("MT5_BRIDGE_API_KEY", "")

EXTENSION_ID = "com.lattice24/mt5-trading"
CAPABILITIES = ["account", "positions", "orders", "market_data", "history", "safety"]


# ---------------------------------------------------------------------------
# HTTP client to bridge
# ---------------------------------------------------------------------------
def _headers() -> dict:
    h = {"Content-Type": "application/json"}
    if API_KEY:
        h["X-Api-Key"] = API_KEY
    return h


async def bridge_get(path: str) -> dict:
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.get(f"{BRIDGE_URL}{path}", headers=_headers())
        if r.status_code >= 400:
            return {"error": f"Bridge returned {r.status_code}: {r.text}"}
        return r.json()


async def bridge_post(path: str, body: dict = None) -> dict:
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(f"{BRIDGE_URL}{path}", headers=_headers(), json=body or {})
        if r.status_code >= 400:
            return {"error": f"Bridge returned {r.status_code}: {r.text}"}
        return r.json()


async def bridge_health() -> bool:
    try:
        result = await bridge_get("/health")
        return result.get("status") == "ok"
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "name": "mt5_connect",
        "description": "Connect to an MT5 trading account. Available keys: ATLAS, BG_INSTANT, BG_CHALLENGE, GL_1, GL_2, GL_3, FTMO, JIMMY_FTMO",
        "inputSchema": {
            "type": "object",
            "properties": {
                "account_key": {"type": "string", "description": "Account key from config"}
            },
        },
    },
    {
        "name": "mt5_account",
        "description": "Get account info: balance, equity, margin, leverage",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "mt5_positions",
        "description": "Get all open positions with P/L, SL, TP",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "mt5_summary",
        "description": "Full account summary: balance, equity, floating P/L, all positions",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "mt5_close",
        "description": "Close a position by ticket number",
        "inputSchema": {
            "type": "object",
            "properties": {"ticket": {"type": "integer", "description": "Position ticket"}},
            "required": ["ticket"],
        },
    },
    {
        "name": "mt5_modify_sltp",
        "description": "Modify stop loss and/or take profit of a position",
        "inputSchema": {
            "type": "object",
            "properties": {
                "ticket": {"type": "integer", "description": "Position ticket"},
                "sl": {"type": "number", "description": "New stop loss price"},
                "tp": {"type": "number", "description": "New take profit price"},
            },
            "required": ["ticket"],
        },
    },
    {
        "name": "mt5_close_losers",
        "description": "Close all positions exceeding loss limit",
        "inputSchema": {
            "type": "object",
            "properties": {
                "max_loss": {"type": "number", "description": "Max loss in dollars (default: $1.00)"}
            },
        },
    },
    {
        "name": "mt5_history",
        "description": "Get closed trade history for a date range",
        "inputSchema": {
            "type": "object",
            "properties": {
                "days": {"type": "integer", "description": "Number of days (default 1, max 90)"}
            },
        },
    },
    {
        "name": "mt5_scan_no_sl",
        "description": "Scan all positions for missing stop losses and rogue trades (unknown magic numbers)",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "mt5_force_sl",
        "description": "Apply emergency stop loss to a position missing SL. Sets SL based on max dollar loss from current price.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "ticket": {"type": "integer", "description": "Position ticket"},
                "max_loss_dollars": {"type": "number", "description": "Max further loss from current price (default $2.00, cap $5.00)"},
            },
            "required": ["ticket"],
        },
    },
    {
        "name": "mt5_symbol",
        "description": "Get symbol details: spread, tick size, volume limits, etc.",
        "inputSchema": {
            "type": "object",
            "properties": {"symbol": {"type": "string", "description": "Symbol name (e.g. BTCUSD)"}},
            "required": ["symbol"],
        },
    },
    {
        "name": "mt5_ohlcv",
        "description": "Get OHLCV price bars for a symbol",
        "inputSchema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Symbol name"},
                "timeframe": {"type": "string", "description": "Timeframe: M1, M5, M15, M30, H1, H4, D1, W1, MN1 (default H1)"},
                "count": {"type": "integer", "description": "Number of bars (default 100, max 1000)"},
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "mt5_order_send",
        "description": "Place a new order. REQUIRES confirmed=true to prevent accidental trades. SL validated against MAX_LOSS_DOLLARS.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Symbol to trade"},
                "order_type": {"type": "string", "description": "BUY or SELL"},
                "volume": {"type": "number", "description": "Lot size"},
                "sl": {"type": "number", "description": "Stop loss price"},
                "tp": {"type": "number", "description": "Take profit price"},
                "magic": {"type": "integer", "description": "Magic number (default 888888)"},
                "comment": {"type": "string", "description": "Order comment"},
                "confirmed": {"type": "boolean", "description": "MUST be true to execute. Safety gate."},
            },
            "required": ["symbol", "order_type", "volume", "confirmed"],
        },
    },
    {
        "name": "mt5_accounts",
        "description": "List available account keys (no passwords exposed)",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------
async def call_tool(name: str, args: dict) -> Any:
    """Route tool call to the bridge."""
    try:
        if name == "mt5_connect":
            return await bridge_post("/connect", {"account_key": args.get("account_key")})
        elif name == "mt5_account":
            return await bridge_get("/account")
        elif name == "mt5_positions":
            return await bridge_get("/positions")
        elif name == "mt5_summary":
            return await bridge_get("/summary")
        elif name == "mt5_close":
            return await bridge_post("/close", {"ticket": args["ticket"]})
        elif name == "mt5_modify_sltp":
            return await bridge_post("/modify_sltp", {
                "ticket": args["ticket"],
                "sl": args.get("sl"),
                "tp": args.get("tp"),
            })
        elif name == "mt5_close_losers":
            return await bridge_post("/close_losers", {"max_loss": args.get("max_loss")})
        elif name == "mt5_history":
            return await bridge_post("/history", {"days": args.get("days", 1)})
        elif name == "mt5_scan_no_sl":
            return await bridge_get("/scan_no_sl")
        elif name == "mt5_force_sl":
            return await bridge_post("/force_sl", {
                "ticket": args["ticket"],
                "max_loss_dollars": args.get("max_loss_dollars", 2.0),
            })
        elif name == "mt5_symbol":
            return await bridge_get(f"/symbol/{args['symbol']}")
        elif name == "mt5_ohlcv":
            return await bridge_post("/ohlcv", {
                "symbol": args["symbol"],
                "timeframe": args.get("timeframe", "H1"),
                "count": args.get("count", 100),
            })
        elif name == "mt5_order_send":
            return await bridge_post("/order_send", {
                "symbol": args["symbol"],
                "order_type": args["order_type"],
                "volume": args["volume"],
                "sl": args.get("sl"),
                "tp": args.get("tp"),
                "magic": args.get("magic", 888888),
                "comment": args.get("comment", "MCP_ORDER"),
                "confirmed": args.get("confirmed", False),
            })
        elif name == "mt5_accounts":
            return await bridge_get("/accounts")
        else:
            return {"error": f"Unknown tool: {name}"}
    except httpx.ConnectError:
        return {
            "error": f"Cannot reach MT5 bridge at {BRIDGE_URL}. "
            "Is mt5_bridge.py running on the Windows host?"
        }
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# MCP Protocol (JSON-RPC over stdio)
# ---------------------------------------------------------------------------
def make_response(req_id, result):
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def make_error(req_id, code, message):
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


async def handle_request(request: dict) -> dict:
    method = request.get("method", "")
    params = request.get("params", {})
    req_id = request.get("id")

    if method == "initialize":
        return make_response(req_id, {
            "protocolVersion": "2025-06-18",
            "capabilities": {
                "tools": {},
                "extensions": {
                    EXTENSION_ID: {
                        "capabilities": CAPABILITIES,
                        "bridge_url": BRIDGE_URL,
                    }
                },
            },
            "serverInfo": {"name": "mt5-trading", "version": "1.0.0"},
        })

    elif method == "notifications/initialized":
        return None  # No response needed for notifications

    elif method == "tools/list":
        return make_response(req_id, {"tools": TOOLS})

    elif method == "tools/call":
        tool_name = params.get("name", "")
        args = params.get("arguments", {})

        result = await call_tool(tool_name, args)
        return make_response(req_id, {
            "content": [{"type": "text", "text": json.dumps(result, indent=2)}]
        })

    return make_error(req_id, -32601, f"Unknown method: {method}")


async def main():
    print("MT5 MCP Server (Docker Layer 2) starting...", file=sys.stderr)
    print(f"Bridge URL: {BRIDGE_URL}", file=sys.stderr)

    # Check bridge connectivity at startup
    healthy = await bridge_health()
    if healthy:
        print("Bridge: CONNECTED", file=sys.stderr)
    else:
        print(f"Bridge: UNREACHABLE at {BRIDGE_URL} (will retry on tool calls)", file=sys.stderr)

    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await asyncio.get_event_loop().connect_read_pipe(lambda: protocol, sys.stdin)

    while True:
        try:
            line = await reader.readline()
            if not line:
                break
            request = json.loads(line.strip())
            response = await handle_request(request)
            if response is not None:
                sys.stdout.write(json.dumps(response) + "\n")
                sys.stdout.flush()
        except json.JSONDecodeError:
            continue
        except Exception as e:
            error_resp = make_error(None, -32700, str(e))
            sys.stdout.write(json.dumps(error_resp) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    asyncio.run(main())
