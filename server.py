#!/usr/bin/env python3
"""
MT5 VPS Trade API — single-file server for Windows VPS with MetaTrader 5.

  pip install -r requirements.txt
  python server.py

Endpoints:
  GET  /health
  GET  /version
  GET  /getAccountHealth
  GET  /getPrice?symbol=XAUUSD
  GET  /getCandles?symbol=XAUUSD&timeframe=M5&count=100
  GET  /getAnalysis?symbol=XAUUSD&timeframe=M5
  GET  /getTradeSuggestion?symbol=XAUUSD&timeframe=M5
  POST /placeOrder
  POST /placeTrades
  GET  /getPositions
  POST /closePositions
  POST /modifyPosition
  POST /trailPosition_MODE1
  POST /trailPosition_MODE2
  POST /trail/stop
  GET  /trail/status
  POST /placeGrid
  POST /gridGuard/start   — monitor floating profit, auto-close basket at target
  POST /gridGuard/stop
  GET  /gridGuard/status
  POST /basketTp/start    — auto TP+SL on all basket positions (+$ / -$ targets)
  POST /basketTp/stop
  GET  /basketTp/status
  POST /basketTp/apply      — one-shot TP recalc
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Callable

import MetaTrader5 as mt5
from flask import Flask, jsonify, request

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_KEY = "alphafx"
API_VERSION = "1.5.0"
MT5_PATH = os.environ.get("MT5_TERMINAL_PATH", "")
HOST = "0.0.0.0"
PORT = 8080
DEFAULT_MAGIC = int(os.environ.get("MT5_DEFAULT_MAGIC", "202611"))
TRAIL_POLL_MS = int(os.environ.get("TRAIL_POLL_MS", "200"))

TIMEFRAMES = {
    "M1": mt5.TIMEFRAME_M1,
    "M2": mt5.TIMEFRAME_M2,
    "M3": mt5.TIMEFRAME_M3,
    "M4": mt5.TIMEFRAME_M4,
    "M5": mt5.TIMEFRAME_M5,
    "M6": mt5.TIMEFRAME_M6,
    "M10": mt5.TIMEFRAME_M10,
    "M12": mt5.TIMEFRAME_M12,
    "M15": mt5.TIMEFRAME_M15,
    "M20": mt5.TIMEFRAME_M20,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H2": mt5.TIMEFRAME_H2,
    "H3": mt5.TIMEFRAME_H3,
    "H4": mt5.TIMEFRAME_H4,
    "H6": mt5.TIMEFRAME_H6,
    "H8": mt5.TIMEFRAME_H8,
    "H12": mt5.TIMEFRAME_H12,
    "D1": mt5.TIMEFRAME_D1,
    "W1": mt5.TIMEFRAME_W1,
    "MN1": mt5.TIMEFRAME_MN1,
}

app = Flask(__name__)


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, X-API-Key"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


@app.route("/", methods=["OPTIONS"])
@app.route("/<path:path>", methods=["OPTIONS"])
def cors_preflight(path=""):
    return "", 204


# ---------------------------------------------------------------------------
# MT5 helpers
# ---------------------------------------------------------------------------
def last_error() -> dict[str, Any]:
    code, msg = mt5.last_error()
    return {"error_code": code, "error_message": msg}


def ensure_mt5(path: str | None = None) -> tuple[bool, str]:
    if mt5.terminal_info() is not None:
        return True, "already connected"
    kwargs = {}
    if path:
        kwargs["path"] = path
    if not mt5.initialize(**kwargs):
        return False, str(mt5.last_error())
    return True, "connected"


def resolve_symbol(symbol: str) -> tuple[Any | None, str]:
    sym = symbol.strip()
    if not sym:
        return None, symbol

    info = mt5.symbol_info(sym)
    if info is not None:
        if not info.visible:
            mt5.symbol_select(sym, True)
            info = mt5.symbol_info(sym)
        return info, sym

    root = sym.split(".")[0]
    for suffix in (".pr", ".c", ".m", ".i", ".raw", "-c", ".pro"):
        candidate = root + suffix
        if candidate == sym:
            continue
        info = mt5.symbol_info(candidate)
        if info is not None:
            if not info.visible:
                mt5.symbol_select(candidate, True)
                info = mt5.symbol_info(candidate)
            return info, candidate

    root_u = root.upper()
    best_name = None
    best_score = -1
    for s in mt5.symbols_get() or []:
        name = s.name
        name_u = name.upper()
        if name_u == root_u:
            score = 100
        elif name_u.startswith(root_u + "."):
            score = 90 - len(name)
        elif name_u.startswith(root_u):
            score = 70 - len(name)
        else:
            continue
        if score > best_score:
            best_score = score
            best_name = name

    if best_name:
        info = mt5.symbol_info(best_name)
        if info and not info.visible:
            mt5.symbol_select(best_name, True)
            info = mt5.symbol_info(best_name)
        if info:
            return info, best_name

    return None, symbol


def resolve_basket_symbol(symbol: str, magic: int | None = None) -> tuple[Any | None, str]:
    """Resolve broker symbol; fall back to open positions for this magic."""
    info, sym = resolve_symbol(symbol)
    if info is not None:
        return info, sym
    if magic is not None:
        positions = get_basket_positions(None, magic)
        if positions:
            sym = positions[0].symbol
            info = mt5.symbol_info(sym)
            if info:
                if not info.visible:
                    mt5.symbol_select(sym, True)
                    info = mt5.symbol_info(sym)
                return info, sym
    return None, symbol


def supported_filling(symbol: str) -> int:
    info = mt5.symbol_info(symbol)
    if not info:
        return mt5.ORDER_FILLING_IOC
    mode = info.filling_mode
    if mode & 1:
        return mt5.ORDER_FILLING_FOK
    if mode & 2:
        return mt5.ORDER_FILLING_IOC
    return mt5.ORDER_FILLING_RETURN


def round_price(symbol: str, price: float) -> float:
    info = mt5.symbol_info(symbol)
    digits = info.digits if info else 5
    return round(price, digits)


def normalize_volume(symbol: str, volume: float) -> float:
    info = mt5.symbol_info(symbol)
    if not info:
        return volume
    step = info.volume_step or 0.01
    vol = max(info.volume_min, min(volume, info.volume_max))
    steps = round(vol / step)
    return round(steps * step, 8)


def profit_points(pos, tick=None) -> float:
    info = mt5.symbol_info(pos.symbol)
    if not info or info.point <= 0:
        return 0.0
    if tick is None:
        tick = mt5.symbol_info_tick(pos.symbol)
    if not tick:
        return 0.0
    if pos.type == mt5.ORDER_TYPE_BUY:
        return (tick.bid - pos.price_open) / info.point
    return (pos.price_open - tick.ask) / info.point


def sl_price_from_entry_pts(pos, sl_pts_from_entry: float) -> float:
    info = mt5.symbol_info(pos.symbol)
    pt = info.point
    if pos.type == mt5.ORDER_TYPE_BUY:
        return round_price(pos.symbol, pos.price_open + sl_pts_from_entry * pt)
    return round_price(pos.symbol, pos.price_open - sl_pts_from_entry * pt)


def pos_to_dict(p) -> dict[str, Any]:
    return {
        "ticket": p.ticket,
        "symbol": p.symbol,
        "type": "buy" if p.type == mt5.ORDER_TYPE_BUY else "sell",
        "volume": p.volume,
        "price_open": p.price_open,
        "price_current": p.price_current,
        "sl": p.sl,
        "tp": p.tp,
        "profit": p.profit,
        "swap": p.swap,
        "magic": p.magic,
        "comment": p.comment,
    }


def result_to_dict(r) -> dict[str, Any] | None:
    if r is None:
        return None
    return {
        "retcode": r.retcode,
        "deal": r.deal,
        "order": r.order,
        "volume": r.volume,
        "price": r.price,
        "comment": r.comment,
    }


def send_order(req: dict) -> tuple[bool, Any]:
    result = mt5.order_send(req)
    ok = result is not None and result.retcode == mt5.TRADE_RETCODE_DONE
    return ok, result


def build_market_request(
    symbol: str,
    order_type: str,
    volume: float,
    sl: float = 0.0,
    tp: float = 0.0,
    magic: int = 0,
    comment: str = "API",
    deviation: int = 50,
) -> tuple[dict | None, str | None]:
    info, sym = resolve_symbol(symbol)
    if info is None:
        return None, f"symbol not found: {symbol}"

    tick = mt5.symbol_info_tick(sym)
    if not tick:
        return None, "no quote"

    volume = normalize_volume(sym, volume)
    order_type = order_type.lower()

    if order_type == "buy":
        trade_type, price = mt5.ORDER_TYPE_BUY, tick.ask
        action = mt5.TRADE_ACTION_DEAL
    elif order_type == "sell":
        trade_type, price = mt5.ORDER_TYPE_SELL, tick.bid
        action = mt5.TRADE_ACTION_DEAL
    else:
        return None, f"invalid market type: {order_type}"

    req = {
        "action": action,
        "symbol": sym,
        "volume": volume,
        "type": trade_type,
        "price": round_price(sym, price),
        "sl": round_price(sym, sl) if sl else 0.0,
        "tp": round_price(sym, tp) if tp else 0.0,
        "deviation": deviation,
        "magic": magic,
        "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": supported_filling(sym),
    }
    return req, None


def build_pending_request(
    symbol: str,
    order_type: str,
    volume: float,
    price: float,
    sl: float = 0.0,
    tp: float = 0.0,
    magic: int = 0,
    comment: str = "API",
) -> tuple[dict | None, str | None]:
    info, sym = resolve_symbol(symbol)
    if info is None:
        return None, f"symbol not found: {symbol}"

    volume = normalize_volume(sym, volume)
    order_type = order_type.lower()
    type_map = {
        "buy_limit": mt5.ORDER_TYPE_BUY_LIMIT,
        "sell_limit": mt5.ORDER_TYPE_SELL_LIMIT,
        "buy_stop": mt5.ORDER_TYPE_BUY_STOP,
        "sell_stop": mt5.ORDER_TYPE_SELL_STOP,
    }
    if order_type not in type_map:
        return None, f"invalid pending type: {order_type}"

    req = {
        "action": mt5.TRADE_ACTION_PENDING,
        "symbol": sym,
        "volume": volume,
        "type": type_map[order_type],
        "price": round_price(sym, price),
        "sl": round_price(sym, sl) if sl else 0.0,
        "tp": round_price(sym, tp) if tp else 0.0,
        "magic": magic,
        "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": supported_filling(sym),
    }
    return req, None


def apply_points_to_prices(
    symbol: str,
    order_type: str,
    entry_price: float,
    sl_points: int | None,
    tp_points: int | None,
) -> tuple[float, float]:
    info = mt5.symbol_info(symbol)
    if not info:
        return 0.0, 0.0
    pt = info.point
    sl, tp = 0.0, 0.0
    is_buy = order_type.lower() in ("buy", "buy_limit", "buy_stop")
    if sl_points and sl_points > 0:
        sl = entry_price - sl_points * pt if is_buy else entry_price + sl_points * pt
    if tp_points and tp_points > 0:
        tp = entry_price + tp_points * pt if is_buy else entry_price - tp_points * pt
    return round_price(symbol, sl), round_price(symbol, tp)


def place_grid_fast(
    symbol: str,
    lot: float,
    distance: int,
    initial_distance: int,
    orders_quantity: int,
    incremental: bool = False,
    magic: int = 78001,
    anchor: float | None = None,
) -> dict[str, Any]:
    """Fast grid — tight OrderSend loop (AlphaFX_XAU_GridTrigger_EA style)."""
    info, sym = resolve_symbol(symbol)
    if info is None:
        return {"ok": False, "error": f"symbol not found: {symbol}"}

    tick = mt5.symbol_info_tick(sym)
    if not tick:
        return {"ok": False, "error": "no quote", **last_error()}

    pt = info.point
    min_dist = max(1, info.trade_stops_level) * pt
    ask, bid = tick.ask, tick.bid
    if anchor is None:
        anchor = round_price(sym, (bid + ask) / 2.0)

    filling = supported_filling(sym)
    t0 = time.perf_counter()
    placed = failed = skipped = 0
    results: list[dict] = []

    for i in range(1, orders_quantity + 1):
        vol = lot * (2 ** (i - 1)) if incremental else lot
        vol = normalize_volume(sym, vol)
        offset_pts = initial_distance + (i - 1) * distance

        buy_price = round_price(sym, anchor + offset_pts * pt)
        if buy_price > ask + min_dist:
            req = {
                "action": mt5.TRADE_ACTION_PENDING,
                "symbol": sym,
                "volume": vol,
                "type": mt5.ORDER_TYPE_BUY_STOP,
                "price": buy_price,
                "sl": 0.0,
                "tp": 0.0,
                "magic": magic,
                "comment": f"GRID_B{i}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": filling,
            }
            ok, res = send_order(req)
            placed += int(ok)
            failed += int(not ok)
            results.append({"side": "buy_stop", "level": i, "price": buy_price, "ok": ok, "order": getattr(res, "order", None)})
        else:
            skipped += 1

        sell_price = round_price(sym, anchor - offset_pts * pt)
        if sell_price < bid - min_dist:
            req = {
                "action": mt5.TRADE_ACTION_PENDING,
                "symbol": sym,
                "volume": vol,
                "type": mt5.ORDER_TYPE_SELL_STOP,
                "price": sell_price,
                "sl": 0.0,
                "tp": 0.0,
                "magic": magic,
                "comment": f"GRID_S{i}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": filling,
            }
            ok, res = send_order(req)
            placed += int(ok)
            failed += int(not ok)
            results.append({"side": "sell_stop", "level": i, "price": sell_price, "ok": ok, "order": getattr(res, "order", None)})
        else:
            skipped += 1

    return {
        "ok": placed > 0,
        "symbol": sym,
        "anchor": anchor,
        "placed": placed,
        "failed": failed,
        "skipped": skipped,
        "elapsed_ms": int((time.perf_counter() - t0) * 1000),
        "orders_quantity": orders_quantity,
        "magic": magic,
        "results": results,
    }


def get_basket_floating(symbol: str | None = None, magic: int | None = None) -> dict[str, Any]:
    """Sum floating P/L (+ swap) for positions matching symbol and/or magic."""
    positions = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
    positions = positions or []
    if magic is not None:
        positions = [p for p in positions if p.magic == int(magic)]

    profit = sum(p.profit + p.swap for p in positions)
    return {
        "floating": round(profit, 2),
        "positions_count": len(positions),
        "symbol": symbol,
        "magic": magic,
    }


def cancel_pending_orders(symbol: str | None = None, magic: int | None = None) -> dict[str, Any]:
    orders = mt5.orders_get(symbol=symbol) if symbol else mt5.orders_get()
    orders = orders or []
    if magic is not None:
        orders = [o for o in orders if o.magic == int(magic)]

    cancelled = failed = 0
    results = []
    for o in orders:
        req = {"action": mt5.TRADE_ACTION_REMOVE, "order": o.ticket}
        ok, res = send_order(req)
        if ok:
            cancelled += 1
        else:
            failed += 1
        results.append({"ticket": o.ticket, "ok": ok})

    return {"cancelled": cancelled, "failed": failed, "total": len(orders), "results": results}


def close_basket(symbol: str | None = None, magic: int | None = None, comment: str = "Grid guard close") -> dict[str, Any]:
    """Close all matching positions and cancel pending orders (grid basket exit)."""
    positions = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
    positions = positions or []
    if magic is not None:
        positions = [p for p in positions if p.magic == int(magic)]

    closed = failed = 0
    pos_results = []
    for pos in positions:
        tick = mt5.symbol_info_tick(pos.symbol)
        if not tick:
            pos_results.append({"ticket": pos.ticket, "ok": False, "error": "no tick"})
            failed += 1
            continue
        close_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        close_price = tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask
        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": pos.symbol,
            "volume": pos.volume,
            "type": close_type,
            "position": pos.ticket,
            "price": close_price,
            "deviation": 50,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": supported_filling(pos.symbol),
        }
        ok, res = send_order(req)
        if ok:
            closed += 1
        else:
            failed += 1
        pos_results.append({"ticket": pos.ticket, "ok": ok, "result": result_to_dict(res)})

    order_result = cancel_pending_orders(symbol, magic)
    return {
        "ok": True,
        "closed_positions": closed,
        "failed_positions": failed,
        "cancelled_orders": order_result["cancelled"],
        "positions": pos_results,
        "orders": order_result,
    }


def get_basket_positions(symbol: str | None = None, magic: int | None = None) -> list:
    if magic is not None:
        positions = [p for p in (mt5.positions_get() or []) if p.magic == int(magic)]
        if not symbol:
            return positions
        _, sym = resolve_symbol(symbol)
        if sym:
            exact = [p for p in positions if p.symbol == sym]
            if exact:
                return exact
        exact = [p for p in positions if p.symbol == symbol]
        if exact:
            return exact
        root = symbol.split(".")[0].upper()
        prefixed = [p for p in positions if p.symbol.upper().startswith(root)]
        return prefixed or positions

    positions = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
    positions = list(positions or [])
    return positions


def position_profit_at_price(pos, close_price: float) -> float:
    action = mt5.ORDER_TYPE_BUY if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_SELL
    profit = mt5.order_calc_profit(action, pos.symbol, pos.volume, pos.price_open, close_price)
    if profit is None:
        return 0.0
    return float(profit) + pos.swap


def basket_profit_at_price(positions: list, close_price: float) -> float:
    return sum(position_profit_at_price(p, close_price) for p in positions)


def find_price_for_basket_profit(
    positions: list,
    target_profit: float,
    price_min: float,
    price_max: float,
    tolerance: float = 0.05,
) -> float | None:
    """Binary-search price where basket P/L equals target (linear in price for one symbol)."""
    if not positions:
        return None
    sym = positions[0].symbol
    lo, hi = price_min, price_max
    p_lo = basket_profit_at_price(positions, lo)
    p_hi = basket_profit_at_price(positions, hi)
    if target_profit < min(p_lo, p_hi) - tolerance or target_profit > max(p_lo, p_hi) + tolerance:
        return None

    for _ in range(80):
        mid = round_price(sym, (lo + hi) / 2.0)
        p_mid = basket_profit_at_price(positions, mid)
        if abs(p_mid - target_profit) <= tolerance:
            return mid
        if p_lo <= p_hi:
            if p_mid < target_profit:
                lo, p_lo = mid, p_mid
            else:
                hi, p_hi = mid, p_mid
        else:
            if p_mid > target_profit:
                lo, p_lo = mid, p_mid
            else:
                hi, p_hi = mid, p_mid
    return round_price(sym, (lo + hi) / 2.0)


def calc_basket_sltp_levels(
    positions: list,
    target_profit: float,
    target_loss: float | None = None,
    search_points: int = 200000,
) -> dict[str, Any]:
    """
    Buy TP / sell SL = price above market where basket P/L = +target_profit.
    Sell TP / buy SL = price below market where basket P/L = +target_profit.
    Buy SL / sell SL (loss side) = prices where basket P/L = -target_loss.
    """
    if not positions:
        return {"ok": False, "error": "no positions"}

    loss_target = target_loss if target_loss and target_loss > 0 else target_profit

    sym = positions[0].symbol
    info = mt5.symbol_info(sym)
    tick = mt5.symbol_info_tick(sym)
    if not info or not tick:
        return {"ok": False, "error": "no symbol info"}

    pt = info.point
    mid = (tick.bid + tick.ask) / 2.0
    min_dist = max(1, info.trade_stops_level) * pt
    range_px = search_points * pt

    buy_tp = find_price_for_basket_profit(positions, target_profit, mid, mid + range_px)
    sell_tp = find_price_for_basket_profit(positions, target_profit, mid - range_px, mid)
    buy_sl = find_price_for_basket_profit(positions, -loss_target, mid - range_px, mid)
    sell_sl = find_price_for_basket_profit(positions, -loss_target, mid, mid + range_px)

    buys = [p for p in positions if p.type == mt5.POSITION_TYPE_BUY]
    sells = [p for p in positions if p.type == mt5.POSITION_TYPE_SELL]

    if buy_tp is not None and buy_tp < tick.ask + min_dist:
        buy_tp = None
    if sell_tp is not None and sell_tp > tick.bid - min_dist:
        sell_tp = None
    if buy_sl is not None and buy_sl > tick.bid - min_dist:
        buy_sl = None
    if sell_sl is not None and sell_sl < tick.ask + min_dist:
        sell_sl = None

    return {
        "ok": True,
        "symbol": sym,
        "target_profit": target_profit,
        "target_loss": loss_target,
        "current_floating": round(basket_profit_at_price(positions, mid), 2),
        "bid": tick.bid,
        "ask": tick.ask,
        "buy_tp_price": buy_tp,
        "sell_tp_price": sell_tp,
        "buy_sl_price": buy_sl,
        "sell_sl_price": sell_sl,
        "buy_positions": len(buys),
        "sell_positions": len(sells),
    }


def calc_basket_tp_levels(positions: list, target_profit: float, search_points: int = 200000) -> dict[str, Any]:
    return calc_basket_sltp_levels(positions, target_profit, target_profit, search_points)


def modify_position_sltp(
    pos,
    sl: float | None = None,
    tp: float | None = None,
    *,
    update_sl: bool = True,
    update_tp: bool = True,
) -> tuple[bool, Any]:
    new_sl = round_price(pos.symbol, sl) if (update_sl and sl is not None) else (pos.sl or 0.0)
    new_tp = round_price(pos.symbol, tp) if (update_tp and tp is not None) else (pos.tp or 0.0)
    ok, result = send_order({
        "action": mt5.TRADE_ACTION_SLTP,
        "symbol": pos.symbol,
        "position": pos.ticket,
        "sl": new_sl,
        "tp": new_tp,
    })
    return ok, result


def apply_basket_sltp(
    symbol: str,
    magic: int,
    target_profit: float,
    target_loss: float | None = None,
    price_tolerance: float = 0.0,
) -> dict[str, Any]:
    """Set buy/sell TP+SL so basket hits +target_profit or -target_loss at those prices."""
    positions = get_basket_positions(symbol, magic)
    if not positions:
        return {"ok": False, "error": "no positions", "modified": 0}

    levels = calc_basket_sltp_levels(positions, target_profit, target_loss)
    if not levels.get("ok"):
        return levels

    buy_tp = levels.get("buy_tp_price")
    sell_tp = levels.get("sell_tp_price")
    buy_sl = levels.get("buy_sl_price")
    sell_sl = levels.get("sell_sl_price")
    modified = failed = 0
    results: list[dict[str, Any]] = []

    for pos in positions:
        is_buy = pos.type == mt5.POSITION_TYPE_BUY
        desired_tp = buy_tp if is_buy else sell_tp
        desired_sl = buy_sl if is_buy else sell_sl

        if desired_tp is None and desired_sl is None:
            results.append({"ticket": pos.ticket, "ok": False, "skipped": True, "reason": "no sl/tp level"})
            continue

        cur_tp = pos.tp or 0.0
        cur_sl = pos.sl or 0.0
        tp_ok = desired_tp is None or (
            price_tolerance > 0 and cur_tp > 0 and abs(cur_tp - desired_tp) <= price_tolerance
        )
        sl_ok = desired_sl is None or (
            price_tolerance > 0 and cur_sl > 0 and abs(cur_sl - desired_sl) <= price_tolerance
        )
        if tp_ok and sl_ok:
            results.append({
                "ticket": pos.ticket,
                "ok": True,
                "skipped": True,
                "tp": cur_tp,
                "sl": cur_sl,
            })
            continue

        ok, res = modify_position_sltp(
            pos,
            sl=desired_sl,
            tp=desired_tp,
            update_sl=desired_sl is not None,
            update_tp=desired_tp is not None,
        )
        modified += int(ok)
        failed += int(not ok)
        results.append({
            "ticket": pos.ticket,
            "type": "buy" if is_buy else "sell",
            "ok": ok,
            "tp": desired_tp,
            "sl": desired_sl,
            "result": result_to_dict(res),
        })

    return {
        "ok": modified > 0 or any(
            levels.get(k) is not None
            for k in ("buy_tp_price", "sell_tp_price", "buy_sl_price", "sell_sl_price")
        ),
        "modified": modified,
        "failed": failed,
        "buy_tp_price": buy_tp,
        "sell_tp_price": sell_tp,
        "buy_sl_price": buy_sl,
        "sell_sl_price": sell_sl,
        "target_profit": target_profit,
        "target_loss": levels.get("target_loss"),
        "current_floating": levels.get("current_floating"),
        "positions": results,
    }


def apply_basket_tps(
    symbol: str,
    magic: int,
    target_profit: float,
    tp_tolerance: float = 0.0,
) -> dict[str, Any]:
    return apply_basket_sltp(symbol, magic, target_profit, target_profit, tp_tolerance)
def build_account_health() -> dict[str, Any]:
    """Full account + terminal health snapshot."""
    ok, msg = ensure_mt5(MT5_PATH or None)
    if not ok:
        return {"ok": False, "connected": False, "error": msg}

    account = mt5.account_info()
    terminal = mt5.terminal_info()
    if account is None:
        return {"ok": False, "connected": False, "error": "no account info"}

    positions = mt5.positions_get() or []
    orders = mt5.orders_get() or []

    floating_profit = sum(p.profit + p.swap for p in positions)
    floating_swap = sum(p.swap for p in positions)

    # Today's closed P/L from deals
    today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    deals = mt5.history_deals_get(today_start, datetime.now() + timedelta(days=1)) or []
    closed_today = sum(d.profit + d.commission + d.swap for d in deals if d.entry == mt5.DEAL_ENTRY_OUT)

    balance = account.balance
    equity = account.equity
    drawdown_usd = balance - equity if balance > equity else 0.0
    drawdown_pct = (drawdown_usd / balance * 100.0) if balance > 0 else 0.0
    margin_level = account.margin_level if account.margin > 0 else None

    health_status = "healthy"
    warnings: list[str] = []
    if not terminal or not terminal.connected:
        health_status = "disconnected"
        warnings.append("terminal not connected to broker")
    if not account.trade_allowed:
        health_status = "restricted"
        warnings.append("trading not allowed on this account")
    if margin_level is not None and margin_level < 150:
        health_status = "warning"
        warnings.append(f"low margin level: {margin_level:.1f}%")
    if drawdown_pct > 10:
        health_status = "warning"
        warnings.append(f"drawdown {drawdown_pct:.1f}%")

    return {
        "ok": True,
        "connected": bool(terminal and terminal.connected),
        "health_status": health_status,
        "warnings": warnings,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "account": {
            "login": account.login,
            "server": account.server,
            "name": account.name,
            "currency": account.currency,
            "leverage": account.leverage,
            "trade_mode": account.trade_mode,
            "trade_allowed": account.trade_allowed,
            "trade_expert": account.trade_expert,
        },
        "balance": {
            "balance": round(balance, 2),
            "equity": round(equity, 2),
            "margin": round(account.margin, 2),
            "free_margin": round(account.margin_free, 2),
            "margin_level": round(margin_level, 2) if margin_level else None,
            "profit": round(account.profit, 2),
            "credit": round(account.credit, 2),
        },
        "drawdown": {
            "usd": round(drawdown_usd, 2),
            "percent": round(drawdown_pct, 2),
        },
        "floating": {
            "positions_count": len(positions),
            "orders_count": len(orders),
            "profit": round(floating_profit, 2),
            "swap": round(floating_swap, 2),
        },
        "today": {
            "closed_pl": round(closed_today, 2),
            "deals_count": len(deals),
        },
        "terminal": {
            "company": terminal.company if terminal else None,
            "name": terminal.name if terminal else None,
            "build": terminal.build if terminal else None,
            "connected": terminal.connected if terminal else False,
            "trade_allowed": terminal.trade_allowed if terminal else False,
        },
        "trail_jobs": len(trail_mgr.status()),
    }


# ---------------------------------------------------------------------------
# Trailing manager
# ---------------------------------------------------------------------------
@dataclass
class TrailJob:
    ticket: int
    mode: int
    step_points: int
    active: bool = True
    last_sl_pts: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)


class TrailingManager:
    def __init__(self, poll_ms: int = 200):
        self._jobs: dict[int, TrailJob] = {}
        self._lock = threading.Lock()
        self._poll_ms = poll_ms
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="trail-manager")
        self._thread.start()

    def add_mode1(self, ticket: int, trail_points: int) -> dict[str, Any]:
        return self._add(ticket, mode=1, step_points=trail_points)

    def add_mode2(self, ticket: int, step_points: int) -> dict[str, Any]:
        return self._add(ticket, mode=2, step_points=step_points)

    def remove(self, ticket: int) -> bool:
        with self._lock:
            job = self._jobs.pop(int(ticket), None)
        return job is not None

    def status(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {
                    "ticket": j.ticket,
                    "mode": j.mode,
                    "step_points": j.step_points,
                    "active": j.active,
                    "last_sl_pts": j.last_sl_pts,
                }
                for j in self._jobs.values()
            ]

    def _add(self, ticket: int, mode: int, step_points: int) -> dict[str, Any]:
        positions = mt5.positions_get(ticket=int(ticket))
        if not positions:
            return {"ok": False, "error": "position not found", "ticket": ticket}

        pos = positions[0]
        step_points = max(1, int(step_points))
        job = TrailJob(ticket=int(ticket), mode=mode, step_points=step_points)

        if mode == 2:
            init_sl_pts = -step_points
            new_sl = sl_price_from_entry_pts(pos, init_sl_pts)
            ok = self._modify_sl(pos, new_sl)
            job.last_sl_pts = init_sl_pts
            with self._lock:
                self._jobs[job.ticket] = job
            self.start()
            return {
                "ok": ok,
                "ticket": ticket,
                "mode": 2,
                "step_points": step_points,
                "initial_sl_pts_from_entry": init_sl_pts,
                "sl": new_sl,
            }

        with self._lock:
            self._jobs[job.ticket] = job
        self.start()
        self._tick_job(job)
        return {"ok": True, "ticket": ticket, "mode": 1, "trail_points": step_points, "sl": pos.sl}

    def _loop(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                tickets = list(self._jobs.keys())
            for ticket in tickets:
                with self._lock:
                    job = self._jobs.get(ticket)
                if job and job.active:
                    self._tick_job(job)
            time.sleep(self._poll_ms / 1000.0)

    def _tick_job(self, job: TrailJob) -> None:
        positions = mt5.positions_get(ticket=job.ticket)
        if not positions:
            with self._lock:
                self._jobs.pop(job.ticket, None)
            return

        pos = positions[0]
        pts = profit_points(pos)
        step = job.step_points

        if job.mode == 1:
            locked_pts = pts - step
            if locked_pts < -step:
                return
        else:
            buckets = int(pts // step)
            locked_pts = step * (buckets - 1)

        if job.last_sl_pts is not None and locked_pts <= job.last_sl_pts:
            return

        new_sl = sl_price_from_entry_pts(pos, locked_pts)
        if not self._is_better_sl(pos, new_sl):
            return

        if self._modify_sl(pos, new_sl):
            job.last_sl_pts = locked_pts

    def _is_better_sl(self, pos, new_sl: float) -> bool:
        if pos.sl <= 0:
            return True
        if pos.type == mt5.ORDER_TYPE_BUY:
            return new_sl > pos.sl
        return new_sl < pos.sl

    def _modify_sl(self, pos, new_sl: float) -> bool:
        new_sl = round_price(pos.symbol, new_sl)
        req = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": pos.symbol,
            "position": pos.ticket,
            "sl": new_sl,
            "tp": pos.tp,
        }
        result = mt5.order_send(req)
        return result is not None and result.retcode == mt5.TRADE_RETCODE_DONE


trail_mgr = TrailingManager(poll_ms=TRAIL_POLL_MS)


# ---------------------------------------------------------------------------
# Grid basket guard — auto-close on floating profit target
# ---------------------------------------------------------------------------
@dataclass
class GridGuardJob:
    key: str
    symbol: str
    magic: int
    max_floating_profit: float  # close when floating >= this (e.g. 2 = +$2)
    active: bool = True
    triggered: bool = False
    last_floating: float = 0.0


class GridGuardManager:
    def __init__(self, poll_ms: int = 500):
        self._jobs: dict[str, GridGuardJob] = {}
        self._lock = threading.Lock()
        self._poll_ms = poll_ms
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="grid-guard")
        self._thread.start()

    def _job_key(self, symbol: str, magic: int) -> str:
        return f"{symbol}:{magic}"

    def add(
        self,
        symbol: str,
        magic: int,
        max_floating_profit: float,
    ) -> dict[str, Any]:
        if max_floating_profit <= 0:
            return {"ok": False, "error": "max_floating_profit must be > 0"}

        info, sym = resolve_symbol(symbol)
        if info is None:
            return {"ok": False, "error": f"symbol not found: {symbol}"}

        key = self._job_key(sym, int(magic))
        job = GridGuardJob(
            key=key,
            symbol=sym,
            magic=int(magic),
            max_floating_profit=float(max_floating_profit),
        )
        with self._lock:
            self._jobs[key] = job
        self.start()
        snap = get_basket_floating(sym, job.magic)
        job.last_floating = snap["floating"]
        return {
            "ok": True,
            "symbol": sym,
            "magic": job.magic,
            "max_floating_profit": job.max_floating_profit,
            "current_floating": snap["floating"],
            "message": f"Guard active: close basket if floating >= +{job.max_floating_profit}",
        }

    def remove(self, symbol: str | None = None, magic: int | None = None) -> int:
        removed = 0
        with self._lock:
            if symbol is not None and magic is not None:
                _, sym = resolve_symbol(symbol)
                key = self._job_key(sym or symbol, int(magic))
                if self._jobs.pop(key, None):
                    removed = 1
            else:
                removed = len(self._jobs)
                self._jobs.clear()
        return removed

    def status(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {
                    "symbol": j.symbol,
                    "magic": j.magic,
                    "active": j.active,
                    "triggered": j.triggered,
                    "max_floating_profit": j.max_floating_profit,
                    "last_floating": j.last_floating,
                }
                for j in self._jobs.values()
            ]

    def _loop(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                keys = list(self._jobs.keys())
            for key in keys:
                with self._lock:
                    job = self._jobs.get(key)
                if job and job.active and not job.triggered:
                    self._tick(job)
            time.sleep(self._poll_ms / 1000.0)

    def _tick(self, job: GridGuardJob) -> None:
        snap = get_basket_floating(job.symbol, job.magic)
        floating = snap["floating"]
        job.last_floating = floating

        if floating < job.max_floating_profit:
            return

        print(f"[GridGuard] floating profit hit: {job.symbol} magic={job.magic} floating=${floating}")
        close_basket(job.symbol, job.magic, comment="GridGuard floating_profit")
        job.triggered = True
        job.active = False
        with self._lock:
            self._jobs.pop(job.key, None)


grid_guard = GridGuardManager(poll_ms=500)


# ---------------------------------------------------------------------------
# Basket SL/TP manager — dynamic per-position TP+SL for basket profit/loss targets
# ---------------------------------------------------------------------------
@dataclass
class BasketTpJob:
    key: str
    symbol: str
    magic: int
    target_profit: float
    target_loss: float
    active: bool = True
    last_position_count: int = 0
    buy_tp_price: float | None = None
    sell_tp_price: float | None = None
    buy_sl_price: float | None = None
    sell_sl_price: float | None = None
    last_floating: float = 0.0
    last_modified: int = 0


class BasketTpManager:
    def __init__(self, poll_ms: int = 1000):
        self._jobs: dict[str, BasketTpJob] = {}
        self._lock = threading.Lock()
        self._poll_ms = poll_ms
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="basket-tp")
        self._thread.start()

    def _job_key(self, symbol: str, magic: int) -> str:
        return f"{symbol}:{magic}"

    def add(
        self,
        symbol: str,
        magic: int,
        target_profit: float,
        target_loss: float | None = None,
    ) -> dict[str, Any]:
        if target_profit <= 0:
            return {"ok": False, "error": "target_profit must be > 0"}

        loss = float(target_loss) if target_loss and target_loss > 0 else float(target_profit)

        info, sym = resolve_basket_symbol(symbol, magic)
        if info is None:
            return {"ok": False, "error": f"symbol not found: {symbol}"}

        key = self._job_key(sym, int(magic))
        job = BasketTpJob(
            key=key,
            symbol=sym,
            magic=int(magic),
            target_profit=float(target_profit),
            target_loss=loss,
        )
        positions = get_basket_positions(sym, job.magic)
        job.last_position_count = len(positions)

        with self._lock:
            self._jobs[key] = job
        self.start()

        apply_result = (
            apply_basket_sltp(sym, job.magic, job.target_profit, job.target_loss)
            if positions
            else {"ok": True, "modified": 0, "message": "waiting for positions"}
        )
        levels = calc_basket_sltp_levels(
            get_basket_positions(sym, job.magic), job.target_profit, job.target_loss
        )
        job.buy_tp_price = levels.get("buy_tp_price")
        job.sell_tp_price = levels.get("sell_tp_price")
        job.buy_sl_price = levels.get("buy_sl_price")
        job.sell_sl_price = levels.get("sell_sl_price")
        job.last_floating = levels.get("current_floating", 0.0)

        return {
            "ok": True,
            "symbol": sym,
            "magic": job.magic,
            "target_profit": job.target_profit,
            "target_loss": job.target_loss,
            "buy_tp_price": job.buy_tp_price,
            "sell_tp_price": job.sell_tp_price,
            "buy_sl_price": job.buy_sl_price,
            "sell_sl_price": job.sell_sl_price,
            "current_floating": job.last_floating,
            "apply": apply_result,
            "message": (
                f"Basket SL/TP active: +${job.target_profit} / -${job.target_loss} | "
                f"buy TP={job.buy_tp_price} SL={job.buy_sl_price} | "
                f"sell TP={job.sell_tp_price} SL={job.sell_sl_price}"
            ),
        }

    def remove(self, symbol: str | None = None, magic: int | None = None) -> int:
        removed = 0
        with self._lock:
            if symbol is not None and magic is not None:
                _, sym = resolve_symbol(symbol)
                key = self._job_key(sym or symbol, int(magic))
                if self._jobs.pop(key, None):
                    removed = 1
            else:
                removed = len(self._jobs)
                self._jobs.clear()
        return removed

    def status(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {
                    "symbol": j.symbol,
                    "magic": j.magic,
                    "active": j.active,
                    "target_profit": j.target_profit,
                    "target_loss": j.target_loss,
                    "buy_tp_price": j.buy_tp_price,
                    "sell_tp_price": j.sell_tp_price,
                    "buy_sl_price": j.buy_sl_price,
                    "sell_sl_price": j.sell_sl_price,
                    "last_floating": j.last_floating,
                    "last_position_count": j.last_position_count,
                    "last_modified": j.last_modified,
                }
                for j in self._jobs.values()
            ]

    def _loop(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                keys = list(self._jobs.keys())
            for key in keys:
                with self._lock:
                    job = self._jobs.get(key)
                if job and job.active:
                    self._tick(job)
            time.sleep(self._poll_ms / 1000.0)

    def _tick(self, job: BasketTpJob) -> None:
        positions = get_basket_positions(job.symbol, job.magic)
        count = len(positions)

        if count == 0:
            with self._lock:
                self._jobs.pop(job.key, None)
            return

        snap = get_basket_floating(job.symbol, job.magic)
        floating = snap["floating"]
        job.last_floating = floating

        if floating >= job.target_profit:
            print(f"[BasketSLTP] profit target hit: {job.symbol} magic={job.magic} floating=${floating}")
            close_basket(job.symbol, job.magic, comment="BasketSLTP profit")
            with self._lock:
                self._jobs.pop(job.key, None)
            return

        if floating <= -job.target_loss:
            print(f"[BasketSLTP] loss target hit: {job.symbol} magic={job.magic} floating=${floating}")
            close_basket(job.symbol, job.magic, comment="BasketSLTP loss")
            with self._lock:
                self._jobs.pop(job.key, None)
            return

        if job.last_position_count > 0 and count < job.last_position_count:
            print(f"[BasketSLTP] partial SL/TP hit, closing remainder: {job.symbol} magic={job.magic}")
            close_basket(job.symbol, job.magic, comment="BasketSLTP partial")
            with self._lock:
                self._jobs.pop(job.key, None)
            return

        job.last_position_count = count

        info = mt5.symbol_info(job.symbol)
        pt = info.point if info else 0.01
        apply_result = apply_basket_sltp(
            job.symbol, job.magic, job.target_profit, job.target_loss, price_tolerance=pt * 2
        )
        job.last_modified = apply_result.get("modified", 0)

        levels = calc_basket_sltp_levels(positions, job.target_profit, job.target_loss)
        if levels.get("ok"):
            job.buy_tp_price = levels.get("buy_tp_price")
            job.sell_tp_price = levels.get("sell_tp_price")
            job.buy_sl_price = levels.get("buy_sl_price")
            job.sell_sl_price = levels.get("sell_sl_price")


basket_tp_mgr = BasketTpManager(poll_ms=1000)


# ---------------------------------------------------------------------------
# Auth / helpers
# ---------------------------------------------------------------------------
def require_api_key(fn: Callable) -> Callable:
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if API_KEY:
            key = request.headers.get("X-API-Key") or request.args.get("api_key")
            if key != API_KEY:
                return jsonify({"ok": False, "error": "unauthorized"}), 401
        return fn(*args, **kwargs)
    return wrapper


def require_mt5(fn: Callable) -> Callable:
    @wraps(fn)
    def wrapper(*args, **kwargs):
        ok, msg = ensure_mt5(MT5_PATH or None)
        if not ok:
            return jsonify({"ok": False, "error": "mt5 not connected", "detail": msg}), 503
        return fn(*args, **kwargs)
    return wrapper


def json_body() -> dict[str, Any]:
    return request.get_json(silent=True) or {}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    ok, msg = ensure_mt5(MT5_PATH or None)
    account = mt5.account_info() if ok else None
    return jsonify({
        "ok": ok,
        "service": "mt5-vps-trade-api",
        "version": API_VERSION,
        "mt5": msg,
        "account": account.login if account else None,
        "server": account.server if account else None,
        "trail_jobs": len(trail_mgr.status()),
    })


@app.route("/version", methods=["GET"])
def get_version():
    """Lightweight version check — no API key, no MT5 required."""
    return jsonify({
        "ok": True,
        "service": "mt5-vps-trade-api",
        "version": API_VERSION,
        "file": "server.py",
    })


@app.route("/getAccountHealth", methods=["GET"])
@require_api_key
def get_account_health():
    """Full account health: balance, equity, margin, drawdown, floating P/L, today's closed P/L."""
    payload = build_account_health()
    status = 200 if payload.get("ok") else 503
    return jsonify(payload), status


@app.route("/getPrice", methods=["GET"])
@require_api_key
@require_mt5
def get_price():
    """Live bid/ask for a symbol. ?symbol=XAUUSD"""
    symbol = request.args.get("symbol", "").strip()
    if not symbol:
        return jsonify({"ok": False, "error": "symbol query param required"}), 400

    info, sym = resolve_symbol(symbol)
    if info is None:
        return jsonify({"ok": False, "error": f"symbol not found: {symbol}"}), 404

    tick = mt5.symbol_info_tick(sym)
    if tick is None:
        return jsonify({"ok": False, "error": "no quote", **last_error()}), 400

    pt = info.point or 0.00001
    spread_pts = round((tick.ask - tick.bid) / pt, 1) if pt > 0 else 0

    return jsonify({
        "ok": True,
        "symbol": sym,
        "bid": tick.bid,
        "ask": tick.ask,
        "last": tick.last,
        "spread_points": spread_pts,
        "spread": round(tick.ask - tick.bid, info.digits),
        "time": datetime.utcfromtimestamp(tick.time).isoformat() + "Z",
        "digits": info.digits,
        "point": info.point,
    })


@app.route("/getCandles", methods=["GET"])
@require_api_key
@require_mt5
def get_candles():
    """
    OHLCV candles from MT5.
    ?symbol=XAUUSD&timeframe=M5&count=100
    count max 5000
    """
    symbol = request.args.get("symbol", "").strip()
    if not symbol:
        return jsonify({"ok": False, "error": "symbol query param required"}), 400

    tf_str = request.args.get("timeframe", "M5").upper()
    timeframe = TIMEFRAMES.get(tf_str)
    if timeframe is None:
        return jsonify({"ok": False, "error": f"invalid timeframe: {tf_str}", "allowed": list(TIMEFRAMES.keys())}), 400

    count = int(request.args.get("count", 100))
    count = max(1, min(count, 5000))

    info, sym = resolve_symbol(symbol)
    if info is None:
        return jsonify({"ok": False, "error": f"symbol not found: {symbol}"}), 404

    rates = mt5.copy_rates_from_pos(sym, timeframe, 0, count)
    if rates is None or len(rates) == 0:
        return jsonify({"ok": False, "error": "no candle data", **last_error()}), 400

    candles = []
    for r in rates:
        candles.append({
            "time": datetime.utcfromtimestamp(int(r["time"])).isoformat() + "Z",
            "open": float(r["open"]),
            "high": float(r["high"]),
            "low": float(r["low"]),
            "close": float(r["close"]),
            "tick_volume": int(r["tick_volume"]),
            "spread": int(r["spread"]) if "spread" in r.dtype.names else 0,
            "real_volume": int(r["real_volume"]) if "real_volume" in r.dtype.names else 0,
        })

    return jsonify({
        "ok": True,
        "symbol": sym,
        "timeframe": tf_str,
        "count": len(candles),
        "first": candles[0]["time"] if candles else None,
        "last": candles[-1]["time"] if candles else None,
        "last_close": candles[-1]["close"] if candles else None,
        "candles": candles,
    })


# ---------------------------------------------------------------------------
# Chart analysis — trends, RSI, order blocks
# ---------------------------------------------------------------------------
def calc_ema(values: list[float], period: int) -> list[float]:
    if len(values) < period:
        return []
    k = 2 / (period + 1)
    ema = [sum(values[:period]) / period]
    for v in values[period:]:
        ema.append(v * k + ema[-1] * (1 - k))
    return ema


def calc_rsi(closes: list[float], period: int = 14) -> list[float]:
    if len(closes) < period + 1:
        return []
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = [max(d, 0) for d in deltas]
    losses = [max(-d, 0) for d in deltas]
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    rsi: list[float] = []
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        rs = avg_gain / avg_loss if avg_loss else 100.0
        rsi.append(100 - (100 / (1 + rs)))
    return rsi


def rates_to_bars(rates) -> list[dict[str, Any]]:
    bars = []
    for r in rates:
        bars.append({
            "time": int(r["time"]),
            "open": float(r["open"]),
            "high": float(r["high"]),
            "low": float(r["low"]),
            "close": float(r["close"]),
            "volume": int(r["tick_volume"]),
        })
    return bars


def calculate_trend(bars: list[dict[str, Any]]) -> str:
    if len(bars) < 5:
        return "NEUTRAL"
    closes = [b["close"] for b in bars]
    ema9 = calc_ema(closes, 9)
    ema21 = calc_ema(closes, 21)
    if not ema9 or not ema21:
        return "NEUTRAL"
    short = ema9[-1]
    long = ema21[-1]
    if short > long * 1.0003:
        return "UP"
    if short < long * 0.9997:
        return "DOWN"
    change = closes[-1] - closes[-4]
    if change > 0:
        return "UP"
    if change < 0:
        return "DOWN"
    return "NEUTRAL"


def find_swing_points(bars: list[dict[str, Any]], lookback: int = 3) -> tuple[list, list]:
    swing_highs, swing_lows = [], []
    if len(bars) < lookback * 2 + 1:
        return swing_highs, swing_lows
    for i in range(lookback, len(bars) - lookback):
        h, l = bars[i]["high"], bars[i]["low"]
        if all(bars[i - j]["high"] < h and bars[i + j]["high"] < h for j in range(1, lookback + 1)):
            swing_highs.append({"index": i, "time": bars[i]["time"], "price": h})
        if all(bars[i - j]["low"] > l and bars[i + j]["low"] > l for j in range(1, lookback + 1)):
            swing_lows.append({"index": i, "time": bars[i]["time"], "price": l})
    return swing_highs, swing_lows


def find_order_blocks(bars: list[dict[str, Any]], swing_highs: list, swing_lows: list) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for swing in swing_lows[-6:]:
        idx = swing["index"]
        for i in range(idx - 1, max(0, idx - 6), -1):
            if bars[i]["close"] < bars[i]["open"]:
                move = bars[min(idx + 2, len(bars) - 1)]["close"] - bars[i]["low"]
                body = abs(bars[i]["close"] - bars[i]["open"])
                if move > body * 1.5:
                    blocks.append({
                        "type": "BULLISH_OB",
                        "time": bars[i]["time"],
                        "high": bars[i]["high"],
                        "low": bars[i]["low"],
                        "rsi_ob": False,
                    })
                    break
    for swing in swing_highs[-6:]:
        idx = swing["index"]
        for i in range(idx - 1, max(0, idx - 6), -1):
            if bars[i]["close"] > bars[i]["open"]:
                move = bars[i]["high"] - bars[min(idx + 2, len(bars) - 1)]["close"]
                body = abs(bars[i]["close"] - bars[i]["open"])
                if move > body * 1.5:
                    blocks.append({
                        "type": "BEARISH_OB",
                        "time": bars[i]["time"],
                        "high": bars[i]["high"],
                        "low": bars[i]["low"],
                        "rsi_ob": False,
                    })
                    break
    return blocks


def format_volume_k(volume: int | float) -> str:
    """Format tick volume like stream overlays: 5200 → '5.2k'."""
    v = float(volume)
    if v >= 1_000_000:
        return f"{v / 1_000_000:.1f}M".replace(".0M", "M")
    if v >= 1000:
        s = f"{v / 1000:.1f}k"
        return s.replace(".0k", "k")
    return str(int(v))


def enrich_order_block_intensity(bars: list[dict[str, Any]], blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    RSI OB intensity — volume at OB candle (5k, 2.5k) + relative volume (2.5x avg).
    Streams usually label OB strength by tick volume and how extreme RSI was.
    """
    time_to_idx = {b["time"]: i for i, b in enumerate(bars)}
    all_vols = [b["volume"] for b in bars]

    for ob in blocks:
        idx = time_to_idx.get(ob["time"])
        if idx is None:
            continue

        bar = bars[idx]
        vol = int(bar.get("volume") or 0)
        start = max(0, idx - 20)
        window = all_vols[start:idx] or all_vols[max(0, idx - 5): idx + 1]
        avg_vol = sum(window) / len(window) if window else max(vol, 1)
        rel = vol / avg_vol if avg_vol else 1.0

        rsi = float(ob.get("rsi") or 50)
        if ob["type"] == "BULLISH_OB":
            rsi_depth = max(0.0, 35.0 - rsi)
        else:
            rsi_depth = max(0.0, rsi - 65.0)

        vol_score = min(100.0, rel * 25.0)
        rsi_score = min(100.0, rsi_depth * 5.0)
        intensity = round(min(100.0, vol_score * 0.65 + rsi_score * 0.35), 1)

        if intensity >= 70:
            tier = "HIGH"
        elif intensity >= 40:
            tier = "MED"
        else:
            tier = "LOW"

        ob["volume"] = vol
        ob["volume_k"] = format_volume_k(vol)
        ob["volume_avg"] = round(avg_vol)
        ob["volume_rel"] = round(rel, 2)
        ob["volume_rel_label"] = f"{rel:.1f}x".replace(".0x", "x")
        ob["rsi_depth"] = round(rsi_depth, 1)
        ob["intensity"] = intensity
        ob["intensity_tier"] = tier
        ob["intensity_label"] = f"{format_volume_k(vol)} · {ob['volume_rel_label']}"

    blocks.sort(key=lambda x: x.get("intensity", 0), reverse=True)
    return blocks


def enrich_order_block_validity(
    bars: list[dict[str, Any]],
    blocks: list[dict[str, Any]],
    current_price: float,
    atr: float,
    max_age_bars: int = 80,
    max_distance_atr: float = 4.0,
) -> list[dict[str, Any]]:
    """
    Classify each OB as tradeable or expired (SMC-style).

    FRESH    — never retested; still valid
    AT_ZONE  — price inside OB now
    TESTED   — one retest, not broken
    MITIGATED — 2+ retests (zone used up)
    BROKEN   — close through OB (bull: close < low, bear: close > high)
    STALE    — too many bars old, or too far without any retest
    """
    time_to_idx = {b["time"]: i for i, b in enumerate(bars)}
    n = len(bars)
    safe_atr = atr if atr > 0 else 1.0

    for ob in blocks:
        idx = time_to_idx.get(ob["time"])
        if idx is None:
            ob.update({
                "status": "UNKNOWN",
                "status_label": "Unknown",
                "is_valid": False,
                "is_tradeable": False,
            })
            continue

        zone_low = float(ob["low"])
        zone_high = float(ob["high"])
        bars_since = n - 1 - idx
        ob["bars_since"] = bars_since

        broken = False
        touches = 0
        for j in range(idx + 1, n):
            bar = bars[j]
            close_p, high_p, low_p = bar["close"], bar["high"], bar["low"]
            if ob["type"] == "BULLISH_OB" and close_p < zone_low:
                broken = True
                break
            if ob["type"] == "BEARISH_OB" and close_p > zone_high:
                broken = True
                break
            if high_p >= zone_low and low_p <= zone_high:
                touches += 1

        if current_price > zone_high:
            dist = current_price - zone_high
        elif current_price < zone_low:
            dist = zone_low - current_price
        else:
            dist = 0.0

        dist_atr = round(dist / safe_atr, 2)
        ob["distance"] = round(dist, 5)
        ob["distance_atr"] = dist_atr
        ob["touch_count"] = touches
        ob["broken"] = broken

        if broken:
            status, label = "BROKEN", "Expired — close through OB"
            is_valid = is_tradeable = False
        elif touches >= 2:
            status, label = "MITIGATED", "Mitigated — tested 2+ times"
            is_valid = is_tradeable = False
        elif dist == 0:
            status, label = "AT_ZONE", "Price at OB now — valid"
            is_valid = is_tradeable = True
        elif bars_since > max_age_bars:
            status, label = "STALE", f"Expired — older than {max_age_bars} bars"
            is_valid = is_tradeable = False
        elif touches == 1:
            status, label = "TESTED", "Tested once — still valid"
            is_valid = is_tradeable = True
        elif dist_atr > max_distance_atr:
            status, label = "STALE", f"Stale — {dist_atr}x ATR away, never tested"
            is_valid = is_tradeable = False
        else:
            status, label = "FRESH", "Fresh — untested OB"
            is_valid = is_tradeable = True

        ob["status"] = status
        ob["status_label"] = label
        ob["is_valid"] = is_valid
        ob["is_tradeable"] = is_tradeable

    return blocks


def mark_rsi_order_blocks(bars: list[dict[str, Any]], blocks: list[dict[str, Any]], period: int = 14) -> list[dict[str, Any]]:
    """Flag OBs where RSI was oversold (bull) or overbought (bear) at formation."""
    closes = [b["close"] for b in bars]
    rsi = calc_rsi(closes, period)
    if not rsi:
        return blocks
    rsi_offset = len(closes) - len(rsi)
    time_to_idx = {b["time"]: i for i, b in enumerate(bars)}
    for ob in blocks:
        idx = time_to_idx.get(ob["time"])
        if idx is None:
            continue
        ri = idx - rsi_offset
        if ri < 0 or ri >= len(rsi):
            continue
        val = rsi[ri]
        ob["rsi"] = round(val, 1)
        if ob["type"] == "BULLISH_OB" and val <= 35:
            ob["rsi_ob"] = True
        if ob["type"] == "BEARISH_OB" and val >= 65:
            ob["rsi_ob"] = True
    return blocks


def analyze_timeframe(sym: str, tf_key: str, count: int = 120) -> dict[str, Any]:
    tf = TIMEFRAMES.get(tf_key)
    if tf is None:
        return {"timeframe": tf_key, "trend": "NEUTRAL", "rsi": None}
    rates = mt5.copy_rates_from_pos(sym, tf, 0, count)
    if rates is None or len(rates) == 0:
        return {"timeframe": tf_key, "trend": "NEUTRAL", "rsi": None}
    bars = rates_to_bars(rates)
    closes = [b["close"] for b in bars]
    rsi_vals = calc_rsi(closes)
    ema20 = calc_ema(closes, 20)
    ema50 = calc_ema(closes, 50)
    return {
        "timeframe": tf_key,
        "trend": calculate_trend(bars),
        "rsi": round(rsi_vals[-1], 1) if rsi_vals else None,
        "rsi_signal": (
            "oversold" if rsi_vals and rsi_vals[-1] < 30 else
            "overbought" if rsi_vals and rsi_vals[-1] > 70 else
            "neutral"
        ),
        "ema20": round(ema20[-1], 5) if ema20 else None,
        "ema50": round(ema50[-1], 5) if ema50 else None,
        "close": closes[-1],
    }


def build_chart_analysis(sym: str, chart_tf: str, count: int = 200) -> dict[str, Any]:
    chart_tf = chart_tf.upper()
    tf = TIMEFRAMES.get(chart_tf, mt5.TIMEFRAME_M5)
    rates = mt5.copy_rates_from_pos(sym, tf, 0, count)
    if rates is None or len(rates) == 0:
        return {"ok": False, "error": "no candle data"}

    bars = rates_to_bars(rates)
    closes = [b["close"] for b in bars]
    highs = [b["high"] for b in bars]
    lows = [b["low"] for b in bars]
    rsi_series = calc_rsi(closes)
    rsi_offset = len(closes) - len(rsi_series)
    rsi_points = [
        {"time": datetime.utcfromtimestamp(bars[rsi_offset + i]["time"]).isoformat() + "Z", "rsi": round(rsi_series[i], 2)}
        for i in range(len(rsi_series))
    ]

    ema20 = calc_ema(closes, 20)
    ema50 = calc_ema(closes, 50)
    ema20_pts, ema50_pts = [], []
    for i, val in enumerate(ema20):
        t = bars[i + (len(closes) - len(ema20))]["time"]
        ema20_pts.append({"time": datetime.utcfromtimestamp(t).isoformat() + "Z", "value": round(val, 5)})
    for i, val in enumerate(ema50):
        t = bars[i + (len(closes) - len(ema50))]["time"]
        ema50_pts.append({"time": datetime.utcfromtimestamp(t).isoformat() + "Z", "value": round(val, 5)})

    swing_h, swing_l = find_swing_points(bars)
    order_blocks = find_order_blocks(bars, swing_h, swing_l)
    order_blocks = mark_rsi_order_blocks(bars, order_blocks)
    order_blocks = enrich_order_block_intensity(bars, order_blocks)

    tick = mt5.symbol_info_tick(sym)
    current_price = tick.bid if tick else closes[-1]
    atr_val = (
        sum(
            max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
            for i in range(1, min(15, len(closes)))
        ) / 14
        if len(closes) > 14 else 5.0
    )
    order_blocks = enrich_order_block_validity(bars, order_blocks, current_price, atr_val)

    for ob in order_blocks:
        ob["time"] = datetime.utcfromtimestamp(ob["time"]).isoformat() + "Z"

    mtf = {tf_key: analyze_timeframe(sym, tf_key) for tf_key in ("M1", "M5", "M15", "M30")}
    trends = {k: v["trend"] for k, v in mtf.items()}
    up = sum(1 for t in trends.values() if t == "UP")
    down = sum(1 for t in trends.values() if t == "DOWN")
    if up >= 3:
        overall = "STRONG_UP"
    elif down >= 3:
        overall = "STRONG_DOWN"
    elif up > down:
        overall = "UP"
    elif down > up:
        overall = "DOWN"
    else:
        overall = "NEUTRAL"

    tick = mt5.symbol_info_tick(sym)
    return {
        "ok": True,
        "symbol": sym,
        "chart_timeframe": chart_tf,
        "price": tick.bid if tick else closes[-1],
        "overall_trend": overall,
        "mtf": mtf,
        "chart": {
            "trend": calculate_trend(bars),
            "rsi": round(rsi_series[-1], 1) if rsi_series else None,
            "rsi_signal": mtf.get(chart_tf, {}).get("rsi_signal"),
            "atr": round(
                sum(max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
                    for i in range(1, min(15, len(closes)))) / 14,
                2,
            ) if len(closes) > 14 else None,
        },
        "rsi_series": rsi_points[-80:],
        "ema20": ema20_pts[-80:],
        "ema50": ema50_pts[-80:],
        "order_blocks": order_blocks,
        "rsi_order_blocks": [ob for ob in order_blocks if ob.get("rsi_ob")],
        "valid_order_blocks": [ob for ob in order_blocks if ob.get("is_tradeable")],
        "valid_rsi_order_blocks": [ob for ob in order_blocks if ob.get("rsi_ob") and ob.get("is_tradeable")],
        "strongest_rsi_ob": next(
            (ob for ob in order_blocks if ob.get("rsi_ob") and ob.get("is_tradeable")),
            next((ob for ob in order_blocks if ob.get("rsi_ob")), None),
        ),
        "swing_highs": [
            {"time": datetime.utcfromtimestamp(s["time"]).isoformat() + "Z", "price": s["price"]}
            for s in swing_h[-8:]
        ],
        "swing_lows": [
            {"time": datetime.utcfromtimestamp(s["time"]).isoformat() + "Z", "price": s["price"]}
            for s in swing_l[-8:]
        ],
    }


def build_trade_suggestion(sym: str, chart_tf: str, count: int = 200) -> dict[str, Any]:
    """Suggest pending order (limit/stop) from trend + order-block analysis."""
    analysis = build_chart_analysis(sym, chart_tf, count)
    if not analysis.get("ok"):
        return analysis

    info = mt5.symbol_info(sym)
    tick = mt5.symbol_info_tick(sym)
    if not info or not tick:
        return {"ok": False, "error": "no quote"}

    bid, ask = tick.bid, tick.ask
    atr = float(analysis["chart"].get("atr") or 5.0)
    pt = info.point
    min_dist = max(1, info.trade_stops_level) * pt
    buffer = max(atr * 0.35, min_dist * 2)

    overall = analysis["overall_trend"]
    rsi_obs = list(analysis.get("rsi_order_blocks") or [])
    all_obs = list(analysis.get("order_blocks") or [])
    tradeable = lambda ob: ob.get("is_tradeable", True)
    pool = [ob for ob in all_obs if tradeable(ob)]
    blocks = sorted(pool, key=lambda o: o.get("intensity", 0), reverse=True)
    chart_trend = (analysis.get("chart") or {}).get("trend", "NEUTRAL")

    def rr_ratio(entry: float, sl: float, tp: float) -> float:
        risk = abs(entry - sl)
        reward = abs(tp - entry)
        return round(reward / risk, 2) if risk > 0 else 0.0

    def nearest_swing_target(levels: list[dict], entry: float, side: str) -> float | None:
        prices = [float(x["price"]) for x in levels]
        if side == "sell":
            below = [p for p in prices if p < entry - buffer]
            return max(below) if below else None
        above = [p for p in prices if p > entry + buffer]
        return min(above) if above else None

    def make_setup(
        order_type: str,
        side: str,
        entry: float,
        sl: float,
        tp: float,
        reason: str,
        confidence: str,
        ob: dict | None,
    ) -> dict[str, Any]:
        entry = round_price(sym, entry)
        sl = round_price(sym, sl)
        tp = round_price(sym, tp)
        return {
            "has_setup": True,
            "side": side,
            "order_type": order_type,
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "rr": rr_ratio(entry, sl, tp),
            "confidence": confidence,
            "reason": reason,
            "bid": bid,
            "ask": ask,
            "atr": atr,
            "overall_trend": overall,
            "ob": ob,
        }

    setup: dict[str, Any] | None = None

    # --- Trend-following: SELL at bearish OB ---
    if overall in ("STRONG_DOWN", "DOWN"):
        bears = [ob for ob in blocks if ob["type"] == "BEARISH_OB"]
        bears_above = sorted([ob for ob in bears if ob["high"] > bid], key=lambda o: o["high"])
        ob = bears_above[0] if bears_above else (bears[0] if bears else None)
        if ob:
            entry = float(ob["high"])
            sl = entry + buffer
            tp = entry - atr * 2.0
            swing_tp = nearest_swing_target(analysis.get("swing_lows") or [], entry, "sell")
            if swing_tp and swing_tp < entry - buffer:
                tp = swing_tp
            order_type = "sell_limit" if entry > ask + min_dist else "sell_stop"
            if order_type == "sell_stop" and entry <= ask:
                entry = ask + min_dist
            conf = ob.get("intensity_tier", "MED") if ob.get("rsi_ob") else "MED"
            setup = make_setup(
                order_type, "SELL", entry, sl, tp,
                f"{overall} — sell retest bearish OB ({ob.get('volume_k', '?')}) · {ob.get('status', 'FRESH')}",
                conf, ob,
            )

    # --- Trend-following: BUY at bullish OB ---
    elif overall in ("STRONG_UP", "UP"):
        bulls = [ob for ob in blocks if ob["type"] == "BULLISH_OB"]
        bulls_below = sorted([ob for ob in bulls if ob["low"] < ask], key=lambda o: -o["low"])
        ob = bulls_below[0] if bulls_below else (bulls[0] if bulls else None)
        if ob:
            entry = float(ob["low"])
            sl = entry - buffer
            tp = entry + atr * 2.0
            swing_tp = nearest_swing_target(analysis.get("swing_highs") or [], entry, "buy")
            if swing_tp and swing_tp > entry + buffer:
                tp = swing_tp
            order_type = "buy_limit" if entry < bid - min_dist else "buy_stop"
            if order_type == "buy_stop" and entry <= ask:
                entry = ask + min_dist
            conf = ob.get("intensity_tier", "MED") if ob.get("rsi_ob") else "MED"
            setup = make_setup(
                order_type, "BUY", entry, sl, tp,
                f"{overall} — buy retest bullish OB ({ob.get('volume_k', '?')}) · {ob.get('status', 'FRESH')}",
                conf, ob,
            )

    # --- NEUTRAL overall: use chart TF trend + any valid OB ---
    elif overall == "NEUTRAL" and blocks:
        prefer_sell = chart_trend in ("DOWN", "STRONG_DOWN")
        prefer_buy = chart_trend in ("UP", "STRONG_UP")
        bears = [ob for ob in blocks if ob["type"] == "BEARISH_OB"]
        bulls = [ob for ob in blocks if ob["type"] == "BULLISH_OB"]
        ob = None
        if prefer_sell and bears:
            ob = bears[0]
        elif prefer_buy and bulls:
            ob = bulls[0]
        elif bears and not bulls:
            ob = bears[0]
        elif bulls and not bears:
            ob = bulls[0]
        elif blocks:
            ob = blocks[0]
        if ob:
            if ob["type"] == "BEARISH_OB":
                entry = float(ob["high"])
                sl = entry + buffer
                tp = entry - atr * 2.0
                swing_tp = nearest_swing_target(analysis.get("swing_lows") or [], entry, "sell")
                if swing_tp and swing_tp < entry - buffer:
                    tp = swing_tp
                if ob.get("status") == "AT_ZONE":
                    order_type = "sell"
                    entry = bid
                    sl = float(ob["high"]) + buffer
                else:
                    order_type = "sell_limit" if entry > ask + min_dist else "sell_stop"
                    if order_type == "sell_stop" and entry <= ask:
                        entry = ask + min_dist
                setup = make_setup(
                    order_type, "SELL", entry, sl, tp,
                    f"NEUTRAL / M5 {chart_trend} — bearish OB ({ob.get('volume_k', '?')}) · {ob.get('status', 'FRESH')}",
                    ob.get("intensity_tier", "MED"), ob,
                )
            else:
                entry = float(ob["low"])
                sl = entry - buffer
                tp = entry + atr * 2.0
                swing_tp = nearest_swing_target(analysis.get("swing_highs") or [], entry, "buy")
                if swing_tp and swing_tp > entry + buffer:
                    tp = swing_tp
                if ob.get("status") == "AT_ZONE":
                    order_type = "buy"
                    entry = ask
                    sl = float(ob["low"]) - buffer
                else:
                    order_type = "buy_limit" if entry < bid - min_dist else "buy_stop"
                    if order_type == "buy_stop" and entry <= ask:
                        entry = ask + min_dist
                setup = make_setup(
                    order_type, "BUY", entry, sl, tp,
                    f"NEUTRAL / M5 {chart_trend} — bullish OB ({ob.get('volume_k', '?')}) · {ob.get('status', 'FRESH')}",
                    ob.get("intensity_tier", "MED"), ob,
                )

    # --- RSI OB fallback when trend branches did not fire ---
    if setup is None and rsi_obs:
        ob = next((o for o in rsi_obs if tradeable(o)), None)
        if ob and ob["type"] == "BULLISH_OB":
            entry = float(ob["low"])
            sl = entry - buffer
            tp = entry + atr * 2.0
            order_type = "buy_limit" if entry < bid - min_dist else "buy_stop"
            setup = make_setup(
                order_type, "BUY", entry, sl, tp,
                f"RSI bullish OB ({ob.get('volume_k', '?')} · RSI {ob.get('rsi')}) · {ob.get('status', 'FRESH')}",
                ob.get("intensity_tier", "LOW"), ob,
            )
        elif ob and ob["type"] == "BEARISH_OB":
            entry = float(ob["high"])
            sl = entry + buffer
            tp = entry - atr * 2.0
            order_type = "sell_limit" if entry > ask + min_dist else "sell_stop"
            setup = make_setup(
                order_type, "SELL", entry, sl, tp,
                f"RSI bearish OB ({ob.get('volume_k', '?')} · RSI {ob.get('rsi')}) · {ob.get('status', 'FRESH')}",
                ob.get("intensity_tier", "LOW"), ob,
            )

    if setup is None:
        return {
            "ok": True,
            "has_setup": False,
            "symbol": sym,
            "chart_timeframe": chart_tf,
            "overall_trend": overall,
            "price": bid,
            "message": "No valid OB setup — all zones broken, mitigated, or stale",
        }

    return {
        "ok": True,
        "symbol": sym,
        "chart_timeframe": chart_tf,
        **setup,
    }


@app.route("/getAnalysis", methods=["GET"])
@require_api_key
@require_mt5
def get_analysis():
    """
    Multi-TF trend LEDs, RSI, order blocks for chart overlays.
    ?symbol=XAUUSD&timeframe=M5&count=200
    """
    symbol = request.args.get("symbol", "").strip()
    if not symbol:
        return jsonify({"ok": False, "error": "symbol required"}), 400
    chart_tf = request.args.get("timeframe", "M5").upper()
    count = max(50, min(int(request.args.get("count", 200)), 5000))
    info, sym = resolve_symbol(symbol)
    if info is None:
        return jsonify({"ok": False, "error": f"symbol not found: {symbol}"}), 404
    result = build_chart_analysis(sym, chart_tf, count)
    return jsonify(result), (200 if result.get("ok") else 400)


@app.route("/getTradeSuggestion", methods=["GET"])
@require_api_key
@require_mt5
def get_trade_suggestion():
    """
    Suggested pending order from analysis: type, entry, SL, TP, R:R.
    ?symbol=XAUUSD&timeframe=M5&count=200
    """
    symbol = request.args.get("symbol", "").strip()
    if not symbol:
        return jsonify({"ok": False, "error": "symbol required"}), 400
    chart_tf = request.args.get("timeframe", "M5").upper()
    count = max(50, min(int(request.args.get("count", 200)), 5000))
    info, sym = resolve_symbol(symbol)
    if info is None:
        return jsonify({"ok": False, "error": f"symbol not found: {symbol}"}), 404
    result = build_trade_suggestion(sym, chart_tf, count)
    return jsonify(result), (200 if result.get("ok") else 400)


def _execute_place_order(data: dict[str, Any]) -> tuple[dict[str, Any], int]:
    symbol = data.get("symbol")
    if not symbol:
        return {"ok": False, "error": "symbol required"}, 400

    order_type = str(data.get("type", "buy")).lower()
    volume = float(data.get("volume", 0.01))
    magic = int(data.get("magic", DEFAULT_MAGIC))
    comment = str(data.get("comment", "API"))
    deviation = int(data.get("deviation", 50))
    sl = float(data.get("sl", 0) or 0)
    tp = float(data.get("tp", 0) or 0)
    sl_points = data.get("sl_points")
    tp_points = data.get("tp_points")
    price = data.get("price")

    if order_type in ("buy", "sell"):
        req, err = build_market_request(symbol, order_type, volume, magic=magic, comment=comment, deviation=deviation)
        if err:
            return {"ok": False, "error": err}, 400
        entry = req["price"]
        if sl_points is not None or tp_points is not None:
            slp = int(sl_points) if sl_points is not None else None
            tpp = int(tp_points) if tp_points is not None else None
            sl, tp = apply_points_to_prices(req["symbol"], order_type, entry, slp, tpp)
        if sl:
            req["sl"] = sl
        if tp:
            req["tp"] = tp
    else:
        if price is None:
            return {"ok": False, "error": "price required for pending orders"}, 400
        req, err = build_pending_request(symbol, order_type, volume, float(price), sl=sl, tp=tp, magic=magic, comment=comment)
        if err:
            return {"ok": False, "error": err}, 400
        if sl_points is not None or tp_points is not None:
            slp = int(sl_points) if sl_points is not None else None
            tpp = int(tp_points) if tp_points is not None else None
            sl, tp = apply_points_to_prices(req["symbol"], order_type, float(price), slp, tpp)
            if sl:
                req["sl"] = sl
            if tp:
                req["tp"] = tp

    ok, result = send_order(req)
    if ok:
        return {"ok": True, "request": {k: req[k] for k in req if k != "type_filling"}, "result": result_to_dict(result)}, 200
    return {"ok": False, "result": result_to_dict(result), **last_error()}, 400


@app.route("/placeOrder", methods=["POST"])
@require_api_key
@require_mt5
def place_order():
    payload, status = _execute_place_order(json_body())
    return jsonify(payload), status


@app.route("/placeTrades", methods=["POST"])
@require_api_key
@require_mt5
def place_trades():
    trades = json_body().get("trades") or []
    if not trades:
        return jsonify({"ok": False, "error": "trades array required"}), 400
    results = []
    for i, t in enumerate(trades):
        payload, status = _execute_place_order(t)
        results.append({"index": i, "status": status, **payload})
    ok_count = sum(1 for r in results if r.get("ok"))
    return jsonify({"ok": ok_count > 0, "placed": ok_count, "total": len(trades), "results": results})


@app.route("/getPositions", methods=["GET"])
@require_api_key
@require_mt5
def get_positions():
    symbol = request.args.get("symbol")
    ticket = request.args.get("ticket")
    magic = request.args.get("magic")

    if ticket:
        positions = mt5.positions_get(ticket=int(ticket))
    elif symbol:
        _, sym = resolve_symbol(symbol)
        positions = mt5.positions_get(symbol=sym)
    else:
        positions = mt5.positions_get()

    if positions is None:
        return jsonify({"ok": False, **last_error()}), 400

    rows = [pos_to_dict(p) for p in positions]
    if magic is not None:
        rows = [p for p in rows if p["magic"] == int(magic)]

    return jsonify({"ok": True, "count": len(rows), "total_profit": sum(p["profit"] for p in rows), "positions": rows})


@app.route("/closePositions", methods=["POST"])
@require_api_key
@require_mt5
def close_positions():
    data = json_body()
    ticket, symbol, magic, volume = data.get("ticket"), data.get("symbol"), data.get("magic"), data.get("volume")

    if ticket:
        positions = mt5.positions_get(ticket=int(ticket))
    elif symbol:
        _, sym = resolve_symbol(symbol)
        positions = mt5.positions_get(symbol=sym)
    else:
        positions = mt5.positions_get()

    if not positions:
        return jsonify({"ok": True, "closed": 0, "message": "no positions"})

    if magic is not None:
        positions = [p for p in positions if p.magic == int(magic)]

    results = []
    for pos in positions:
        tick = mt5.symbol_info_tick(pos.symbol)
        if not tick:
            results.append({"ticket": pos.ticket, "ok": False, "error": "no tick"})
            continue
        close_vol = normalize_volume(pos.symbol, float(volume) if volume else pos.volume)
        close_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        close_price = tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask
        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": pos.symbol,
            "volume": close_vol,
            "type": close_type,
            "position": pos.ticket,
            "price": close_price,
            "deviation": int(data.get("deviation", 50)),
            "comment": str(data.get("comment", "API close")),
            "type_time": mt5.ORDER_TIME_GTC,
        }
        ok, res = send_order(req)
        results.append({"ticket": pos.ticket, "ok": ok, "result": result_to_dict(res)})

    closed = sum(1 for r in results if r["ok"])
    return jsonify({"ok": closed > 0, "closed": closed, "results": results})


@app.route("/modifyPosition", methods=["POST"])
@require_api_key
@require_mt5
def modify_position():
    data = json_body()
    ticket = data.get("ticket")
    if not ticket:
        return jsonify({"ok": False, "error": "ticket required"}), 400

    positions = mt5.positions_get(ticket=int(ticket))
    if not positions:
        return jsonify({"ok": False, "error": "position not found"}), 404

    pos = positions[0]
    new_sl, new_tp = pos.sl, pos.tp
    side = "buy" if pos.type == mt5.ORDER_TYPE_BUY else "sell"

    if "sl_points" in data:
        sp = int(data["sl_points"])
        if sp < 0:
            new_sl = 0.0
        elif sp == 0:
            new_sl = pos.sl
        else:
            new_sl, _ = apply_points_to_prices(pos.symbol, side, pos.price_open, sp, None)
    elif "sl" in data:
        new_sl = float(data["sl"])

    if "tp_points" in data:
        tp = int(data["tp_points"])
        if tp < 0:
            new_tp = 0.0
        elif tp == 0:
            new_tp = pos.tp
        else:
            _, new_tp = apply_points_to_prices(pos.symbol, side, pos.price_open, None, tp)
    elif "tp" in data:
        new_tp = float(data["tp"])

    new_sl = round_price(pos.symbol, new_sl) if new_sl else 0.0
    new_tp = round_price(pos.symbol, new_tp) if new_tp else 0.0

    ok, result = send_order({
        "action": mt5.TRADE_ACTION_SLTP,
        "symbol": pos.symbol,
        "position": pos.ticket,
        "sl": new_sl,
        "tp": new_tp,
    })
    if ok:
        return jsonify({"ok": True, "ticket": pos.ticket, "sl": new_sl, "tp": new_tp, "result": result_to_dict(result)})
    return jsonify({"ok": False, "result": result_to_dict(result), **last_error()}), 400


@app.route("/trailPosition_MODE1", methods=["POST"])
@require_api_key
@require_mt5
def trail_mode1():
    data = json_body()
    ticket = data.get("ticket")
    trail_pts = data.get("trail_points") or data.get("step_points")
    if not ticket or trail_pts is None:
        return jsonify({"ok": False, "error": "ticket and trail_points required"}), 400
    result = trail_mgr.add_mode1(int(ticket), int(trail_pts))
    return jsonify(result), (200 if result.get("ok") else 400)


@app.route("/trailPosition_MODE2", methods=["POST"])
@require_api_key
@require_mt5
def trail_mode2():
    data = json_body()
    ticket = data.get("ticket")
    step_pts = data.get("step_points") or data.get("trail_points")
    if not ticket or step_pts is None:
        return jsonify({"ok": False, "error": "ticket and step_points required"}), 400
    result = trail_mgr.add_mode2(int(ticket), int(step_pts))
    return jsonify(result), (200 if result.get("ok") else 400)


@app.route("/trail/stop", methods=["POST"])
@require_api_key
def trail_stop():
    data = json_body()
    ticket = data.get("ticket")
    if ticket:
        return jsonify({"ok": trail_mgr.remove(int(ticket)), "ticket": int(ticket)})
    for job in trail_mgr.status():
        trail_mgr.remove(job["ticket"])
    return jsonify({"ok": True, "message": "all trail jobs stopped"})


@app.route("/trail/status", methods=["GET"])
@require_api_key
def trail_status():
    return jsonify({"ok": True, "jobs": trail_mgr.status()})


@app.route("/placeGrid", methods=["POST"])
@require_api_key
@require_mt5
def place_grid():
    data = json_body()
    symbol = data.get("symbol")
    if not symbol:
        return jsonify({"ok": False, "error": "symbol required"}), 400

    magic = int(data.get("magic", 78001))
    max_profit = float(data.get("max_floating_profit") or data.get("floating_profit") or 0)
    basket_tp = float(
        data.get("basket_tp_profit") or data.get("basket_tp") or max_profit or 0
    )
    basket_sl = float(
        data.get("basket_sl_loss") or data.get("basket_sl") or data.get("target_loss") or basket_tp or 0
    )

    result = place_grid_fast(
        symbol=symbol,
        lot=float(data.get("lot", 0.01)),
        distance=int(data.get("distance", 100)),
        initial_distance=int(data.get("initial_distance", 200)),
        orders_quantity=int(data.get("orders_quantity", 50)),
        incremental=bool(data.get("incremental", False)),
        magic=magic,
        anchor=float(data["anchor"]) if data.get("anchor") else None,
    )

    if result.get("ok") and max_profit > 0:
        guard = grid_guard.add(
            symbol=result.get("symbol", symbol),
            magic=magic,
            max_floating_profit=max_profit,
        )
        result["grid_guard"] = guard

    if result.get("ok") and basket_tp > 0:
        result["basket_tp"] = basket_tp_mgr.add(
            symbol=result.get("symbol", symbol),
            magic=magic,
            target_profit=basket_tp,
            target_loss=basket_sl if basket_sl > 0 else None,
        )

    return jsonify(result), (200 if result.get("ok") else 400)


@app.route("/gridGuard/start", methods=["POST"])
@require_api_key
@require_mt5
def grid_guard_start():
    """
    Monitor basket floating profit and auto-close all positions + pending orders.
    Body: symbol, magic, max_floating_profit (e.g. 2 = close at +$2)
    """
    data = json_body()
    symbol = data.get("symbol")
    if not symbol:
        return jsonify({"ok": False, "error": "symbol required"}), 400
    magic = int(data.get("magic", 78001))
    max_profit = float(data.get("max_floating_profit") or data.get("floating_profit") or 0)
    result = grid_guard.add(symbol, magic, max_profit)
    return jsonify(result), (200 if result.get("ok") else 400)


@app.route("/gridGuard/stop", methods=["POST"])
@require_api_key
def grid_guard_stop():
    data = json_body()
    symbol = data.get("symbol")
    magic = data.get("magic")
    if symbol and magic is not None:
        removed = grid_guard.remove(symbol, int(magic))
    else:
        removed = grid_guard.remove()
    return jsonify({"ok": True, "removed": removed})


@app.route("/gridGuard/status", methods=["GET"])
@require_api_key
def grid_guard_status():
    symbol = request.args.get("symbol")
    magic = request.args.get("magic")
    floating = None
    if symbol:
        _, sym = resolve_symbol(symbol)
        floating = get_basket_floating(sym, int(magic) if magic else None)
    return jsonify({
        "ok": True,
        "guards": grid_guard.status(),
        "floating": floating,
    })


@app.route("/closeGridBasket", methods=["POST"])
@require_api_key
@require_mt5
def close_grid_basket():
    """Manually close all positions + cancel pending orders for symbol/magic."""
    data = json_body()
    symbol = data.get("symbol")
    magic = data.get("magic")
    sym = None
    if symbol:
        _, sym = resolve_symbol(symbol)
    result = close_basket(sym, int(magic) if magic is not None else None)
    if symbol and magic is not None:
        grid_guard.remove(symbol, int(magic))
        basket_tp_mgr.remove(symbol, int(magic))
    return jsonify(result)


@app.route("/basketTp/start", methods=["POST"])
@require_api_key
@require_mt5
def basket_tp_start():
    """
    Auto-set buy/sell TP+SL so basket hits +target_profit / -target_loss.
    Keeps recalculating as new grid positions open.
    Body: symbol, magic, target_profit, target_loss (optional, defaults to target_profit)
    """
    data = json_body()
    symbol = data.get("symbol")
    if not symbol:
        return jsonify({"ok": False, "error": "symbol required"}), 400
    magic = int(data.get("magic", 78001))
    target = float(data.get("target_profit") or data.get("basket_tp_profit") or data.get("basket_tp") or 0)
    target_loss = float(
        data.get("target_loss") or data.get("basket_sl_loss") or data.get("basket_sl") or target
    )
    result = basket_tp_mgr.add(symbol, magic, target, target_loss)
    return jsonify(result), (200 if result.get("ok") else 400)


@app.route("/basketTp/stop", methods=["POST"])
@require_api_key
def basket_tp_stop():
    data = json_body()
    symbol = data.get("symbol")
    magic = data.get("magic")
    if symbol and magic is not None:
        removed = basket_tp_mgr.remove(symbol, int(magic))
    else:
        removed = basket_tp_mgr.remove()
    return jsonify({"ok": True, "removed": removed})


@app.route("/basketTp/status", methods=["GET"])
@require_api_key
def basket_tp_status():
    symbol = request.args.get("symbol")
    magic = request.args.get("magic")
    levels = None
    if symbol:
        _, sym = resolve_symbol(symbol)
        positions = get_basket_positions(sym, int(magic) if magic else None)
        jobs = basket_tp_mgr.status()
        target = None
        target_loss = None
        if magic and jobs:
            for j in jobs:
                if j["symbol"] == sym and j["magic"] == int(magic):
                    target = j["target_profit"]
                    target_loss = j.get("target_loss")
                    break
        if positions and target:
            levels = calc_basket_sltp_levels(positions, target, target_loss)
        elif positions:
            levels = {"positions_count": len(positions), "current_floating": get_basket_floating(sym, int(magic) if magic else None)}
    return jsonify({
        "ok": True,
        "jobs": basket_tp_mgr.status(),
        "levels": levels,
    })


@app.route("/basketTp/apply", methods=["POST"])
@require_api_key
@require_mt5
def basket_tp_apply():
    """One-shot: recalculate and apply basket TP+SL without starting background job."""
    data = json_body()
    symbol = data.get("symbol")
    if not symbol:
        return jsonify({"ok": False, "error": "symbol required"}), 400
    magic = int(data.get("magic", 78001))
    target = float(data.get("target_profit") or data.get("basket_tp_profit") or data.get("basket_tp") or 0)
    target_loss = float(
        data.get("target_loss") or data.get("basket_sl_loss") or data.get("basket_sl") or target
    )
    if target <= 0:
        return jsonify({"ok": False, "error": "target_profit must be > 0"}), 400
    _, sym = resolve_basket_symbol(symbol, magic)
    if not sym or mt5.symbol_info(sym) is None:
        return jsonify({"ok": False, "error": f"symbol not found: {symbol}"}), 400
    result = apply_basket_sltp(sym, magic, target, target_loss)
    levels = calc_basket_sltp_levels(get_basket_positions(sym, magic), target, target_loss)
    result["levels"] = levels
    return jsonify(result), (200 if result.get("ok") else 400)


if __name__ == "__main__":
    ok, msg = ensure_mt5(MT5_PATH or None)
    print(f"[MT5] {msg}")
    print(f"[MT5] API key: {API_KEY} | listening on port {PORT}")
    trail_mgr.start()
    grid_guard.start()
    basket_tp_mgr.start()
    app.run(host=HOST, port=PORT, threaded=True)
