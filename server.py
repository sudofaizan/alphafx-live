"""
MT5 REST API Server - Full Featured (No numpy)
"""
from flask import Flask, jsonify, request
import MetaTrader5 as mt5
from datetime import datetime, timedelta

# Import SMC Analysis Module (official documented methodology)
try:
    from smc_analysis import SMCAnalyzer, analyze_smc, calculate_fibonacci_ote
    SMC_MODULE_LOADED = True
    print("[INFO] SMC Analysis module loaded successfully")
except ImportError:
    SMC_MODULE_LOADED = False
    print("[WARNING] smc_analysis module not found - using legacy SMC functions")

app = Flask(__name__)

# ============== Constants ==============
TIMEFRAMES = {
    'M1': mt5.TIMEFRAME_M1, 'M2': mt5.TIMEFRAME_M2, 'M3': mt5.TIMEFRAME_M3,
    'M4': mt5.TIMEFRAME_M4, 'M5': mt5.TIMEFRAME_M5, 'M6': mt5.TIMEFRAME_M6,
    'M10': mt5.TIMEFRAME_M10, 'M12': mt5.TIMEFRAME_M12, 'M15': mt5.TIMEFRAME_M15,
    'M20': mt5.TIMEFRAME_M20, 'M30': mt5.TIMEFRAME_M30, 'H1': mt5.TIMEFRAME_H1,
    'H2': mt5.TIMEFRAME_H2, 'H3': mt5.TIMEFRAME_H3, 'H4': mt5.TIMEFRAME_H4,
    'H6': mt5.TIMEFRAME_H6, 'H8': mt5.TIMEFRAME_H8, 'H12': mt5.TIMEFRAME_H12,
    'D1': mt5.TIMEFRAME_D1, 'W1': mt5.TIMEFRAME_W1, 'MN1': mt5.TIMEFRAME_MN1
}

# ============== Helpers ==============
def err(): 
    e = mt5.last_error()
    return {"error_code": e[0], "error_message": e[1]}

def pos_to_dict(p):
    return {"ticket": p.ticket, "time": datetime.fromtimestamp(p.time).isoformat(),
            "type": "buy" if p.type == 0 else "sell", "magic": p.magic, "volume": p.volume,
            "price_open": p.price_open, "sl": p.sl, "tp": p.tp, "price_current": p.price_current,
            "swap": p.swap, "profit": p.profit, "symbol": p.symbol, "comment": p.comment}

def deal_to_dict(d):
    return {"ticket": d.ticket, "order": d.order, "time": datetime.fromtimestamp(d.time).isoformat(),
            "type": d.type, "entry": d.entry, "volume": d.volume, "price": d.price,
            "commission": d.commission, "swap": d.swap, "profit": d.profit, "symbol": d.symbol,
            "comment": d.comment, "magic": d.magic, "reason": d.reason}

def order_to_dict(o):
    return {"ticket": o.ticket, "time_setup": datetime.fromtimestamp(o.time_setup).isoformat(),
            "type": o.type, "state": o.state, "volume_initial": o.volume_initial,
            "volume_current": o.volume_current, "price_open": o.price_open, "sl": o.sl,
            "tp": o.tp, "symbol": o.symbol, "comment": o.comment, "magic": o.magic}

def result_to_dict(r):
    if not r: return None
    return {"retcode": r.retcode, "deal": r.deal, "order": r.order, "volume": r.volume,
            "price": r.price, "bid": r.bid, "ask": r.ask, "comment": r.comment}

# ============== Technical Indicators (Pure Python) ==============
def calc_sma(data, period):
    if len(data) < period: return []
    return [sum(data[i-period:i])/period for i in range(period, len(data)+1)]

def calc_ema(data, period):
    if len(data) < period: return []
    ema = [sum(data[:period])/period]
    k = 2 / (period + 1)
    for i in range(period, len(data)):
        ema.append(ema[-1] + k * (data[i] - ema[-1]))
    return ema

def calc_smma(data, period):
    if len(data) < period: return []
    smma = [sum(data[:period])/period]
    for i in range(period, len(data)):
        smma.append((smma[-1] * (period - 1) + data[i]) / period)
    return smma

def calc_lwma(data, period):
    if len(data) < period: return []
    result = []
    weight_sum = sum(range(1, period + 1))
    for i in range(period - 1, len(data)):
        weighted = sum(data[i-period+1+j] * (j+1) for j in range(period))
        result.append(weighted / weight_sum)
    return result

def calc_rsi(data, period=14):
    if len(data) < period + 1: return []
    deltas = [data[i] - data[i-1] for i in range(1, len(data))]
    gains = [d if d > 0 else 0 for d in deltas]
    losses = [-d if d < 0 else 0 for d in deltas]
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    rsi = []
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        rs = avg_gain / avg_loss if avg_loss != 0 else 100
        rsi.append(100 - (100 / (1 + rs)))
    return rsi

def calc_macd(data, fast=12, slow=26, signal=9):
    if len(data) < slow + signal: return {"macd": [], "signal": [], "histogram": []}
    ema_fast = calc_ema(data, fast)
    ema_slow = calc_ema(data, slow)
    diff = slow - fast
    macd_line = [ema_fast[i] - ema_slow[i-diff] for i in range(diff, len(ema_fast))]
    signal_line = calc_ema(macd_line, signal)
    diff2 = len(macd_line) - len(signal_line)
    histogram = [macd_line[i+diff2] - signal_line[i] for i in range(len(signal_line))]
    return {"macd": macd_line[-len(signal_line):], "signal": signal_line, "histogram": histogram}

def calc_bollinger(data, period=20, std_dev=2):
    if len(data) < period: return {"upper": [], "middle": [], "lower": []}
    middle = calc_sma(data, period)
    upper, lower = [], []
    for i in range(period - 1, len(data)):
        subset = data[i-period+1:i+1]
        mean = sum(subset) / len(subset)
        variance = sum((x - mean) ** 2 for x in subset) / len(subset)
        std = variance ** 0.5
        upper.append(middle[i-period+1] + std_dev * std)
        lower.append(middle[i-period+1] - std_dev * std)
    return {"upper": upper, "middle": middle, "lower": lower}

def calc_stochastic(high, low, close, k_period=14, d_period=3):
    if len(close) < k_period: return {"k": [], "d": []}
    k_values = []
    for i in range(k_period - 1, len(close)):
        highest = max(high[i-k_period+1:i+1])
        lowest = min(low[i-k_period+1:i+1])
        if highest - lowest != 0:
            k_values.append(100 * (close[i] - lowest) / (highest - lowest))
        else:
            k_values.append(50.0)
    d_values = calc_sma(k_values, d_period)
    return {"k": k_values, "d": d_values}

def calc_atr(high, low, close, period=14):
    if len(close) < period + 1: return []
    tr = [high[0] - low[0]]
    for i in range(1, len(close)):
        tr.append(max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1])))
    return calc_smma(tr, period)

def calc_cci(high, low, close, period=20):
    if len(close) < period: return []
    tp = [(high[i] + low[i] + close[i]) / 3 for i in range(len(close))]
    result = []
    for i in range(period - 1, len(tp)):
        subset = tp[i-period+1:i+1]
        sma = sum(subset) / len(subset)
        mad = sum(abs(x - sma) for x in subset) / len(subset)
        if mad != 0:
            result.append((tp[i] - sma) / (0.015 * mad))
        else:
            result.append(0.0)
    return result

def calc_williams_r(high, low, close, period=14):
    if len(close) < period: return []
    result = []
    for i in range(period - 1, len(close)):
        highest = max(high[i-period+1:i+1])
        lowest = min(low[i-period+1:i+1])
        if highest - lowest != 0:
            result.append(-100 * (highest - close[i]) / (highest - lowest))
        else:
            result.append(-50.0)
    return result

def calc_momentum(data, period=10):
    if len(data) < period + 1: return []
    return [data[i] - data[i-period] for i in range(period, len(data))]

def calc_roc(data, period=10):
    if len(data) < period + 1: return []
    return [(data[i] - data[i-period]) / data[i-period] * 100 if data[i-period] != 0 else 0 for i in range(period, len(data))]

# ============== Connection ==============
# Default MT5 terminal path (adjust as needed)
MT5_TERMINAL_PATH = r"C:\Program Files\MetaTrader 5\terminal64.exe"

_INVOKE_KEYS = frozenset({
    "ephemeral", "kill_terminal", "launch_terminal", "login", "password",
    "server", "path", "portable", "timeout",
})


def _is_mt5_running():
    import subprocess
    try:
        result = subprocess.run(
            ['tasklist', '/FI', 'IMAGENAME eq terminal64.exe'],
            capture_output=True, text=True, timeout=5,
        )
        return 'terminal64.exe' in result.stdout
    except Exception:
        return False


def _kill_mt5_terminal():
    import os
    import subprocess
    killed = False
    try:
        ret = os.system('taskkill /F /IM terminal64.exe >nul 2>&1')
        if ret == 0:
            killed = True
        else:
            ret2 = os.system('taskkill /F /IM terminal.exe >nul 2>&1')
            if ret2 == 0:
                killed = True
    except Exception as e:
        print(f"[MT5] os.system kill failed: {e}")
    if not killed:
        try:
            proc = subprocess.run(
                'taskkill /F /IM terminal64.exe',
                shell=True, capture_output=True, text=True, timeout=10,
            )
            if proc.returncode == 0 or 'SUCCESS' in (proc.stdout or '').upper():
                killed = True
        except Exception as e:
            print(f"[MT5] subprocess kill failed: {e}")
    return killed


def _ensure_mt5_connected(data=None):
    """Launch terminal if needed, initialize Python API. Credentials from body or env."""
    import os
    import time

    data = data or {}
    launch_terminal = data.get('launch_terminal', True)
    terminal_path = data.get('path') or os.environ.get('MT5_TERMINAL_PATH') or MT5_TERMINAL_PATH

    existing = mt5.terminal_info()
    if existing is not None:
        acct = mt5.account_info()
        return {
            "success": True,
            "terminal_launched": False,
            "already_connected": True,
            "account": acct.login if acct else None,
            "server": acct.server if acct else None,
        }

    terminal_launched = False
    if launch_terminal and not _is_mt5_running():
        try:
            print(f"[MT5] Launching terminal: {terminal_path}")
            import subprocess
            subprocess.Popen([terminal_path], shell=False)
            terminal_launched = True
            time.sleep(5)
        except Exception as e:
            return {"success": False, "terminal_launched": False, "error": f"launch failed: {e}"}

    kwargs = {"timeout": int(data.get('timeout', 60000)), "portable": bool(data.get('portable', False))}
    if data.get('path'):
        kwargs['path'] = data['path']
    login = data.get('login') or os.environ.get('MT5_LOGIN')
    password = data.get('password') or os.environ.get('MT5_PASSWORD')
    server = data.get('server') or os.environ.get('MT5_SERVER')
    if login:
        kwargs['login'] = int(login)
    if password:
        kwargs['password'] = str(password)
    if server:
        kwargs['server'] = str(server)

    max_retries = 3 if terminal_launched else 1
    for attempt in range(max_retries):
        if mt5.initialize(**kwargs):
            acct = mt5.account_info()
            return {
                "success": True,
                "terminal_launched": terminal_launched,
                "account": acct.login if acct else None,
                "server": acct.server if acct else None,
            }
        if terminal_launched and attempt < max_retries - 1:
            time.sleep(3)

    e = err()
    return {"success": False, "terminal_launched": terminal_launched, **e}


def _disconnect_mt5(kill_terminal=True):
    mt5.shutdown()
    result = {"success": True, "api_disconnected": True, "terminal_killed": False}
    if kill_terminal:
        result["terminal_killed"] = _kill_mt5_terminal()
        result["message"] = (
            "MT5 terminal closed" if result["terminal_killed"]
            else "API disconnected (terminal may still be running)"
        )
    else:
        result["message"] = "API disconnected (terminal still running)"
    return result


def _invoke_strip(data):
    return {k: v for k, v in data.items() if k not in _INVOKE_KEYS}


@app.route('/api/init', methods=['POST'])
def initialize():
    """Initialize MT5 connection. Optionally launches MT5 terminal if not running."""
    data = request.json or {}
    result = _ensure_mt5_connected(data)
    if result.get("success"):
        return jsonify({"success": True, "message": "MT5 initialized", **result})
    return jsonify(result), 400


@app.route('/api/shutdown', methods=['POST'])
def shutdown():
    """Shutdown MT5 API connection and optionally kill the terminal"""
    data = request.json or {}
    kill_terminal = data.get('kill_terminal', True)
    return jsonify(_disconnect_mt5(kill_terminal))


@app.route('/api/invoke/trade', methods=['POST'])
def invoke_trade():
    """
    One-shot trade: connect (launch terminal if needed) -> open order -> disconnect.
    Set ephemeral=false to keep MT5 running after the call.
    Credentials: JSON login/password/server or env MT5_LOGIN, MT5_PASSWORD, MT5_SERVER.
    """
    data = request.json or {}
    ephemeral = data.get('ephemeral', True)
    kill_terminal = data.get('kill_terminal', ephemeral)

    conn = _ensure_mt5_connected(data)
    if not conn.get("success"):
        return jsonify({"phase": "connect", **conn}), 400

    trade_body = _invoke_strip(data)
    with app.test_request_context('/api/trade/open', method='POST', json=trade_body):
        resp = open_trade()

    payload = resp.get_json() if hasattr(resp, 'get_json') else {}
    status_code = resp.status_code if hasattr(resp, 'status_code') else 200
    payload["connect"] = conn

    if ephemeral:
        payload["shutdown"] = _disconnect_mt5(kill_terminal)

    return jsonify(payload), status_code


@app.route('/api/invoke/close', methods=['POST'])
def invoke_close():
    """One-shot close position by ticket."""
    data = request.json or {}
    ephemeral = data.get('ephemeral', True)
    kill_terminal = data.get('kill_terminal', ephemeral)

    conn = _ensure_mt5_connected(data)
    if not conn.get("success"):
        return jsonify({"phase": "connect", **conn}), 400

    trade_body = _invoke_strip(data)
    with app.test_request_context('/api/trade/close', method='POST', json=trade_body):
        resp = close_trade()

    payload = resp.get_json() if hasattr(resp, 'get_json') else {}
    status_code = resp.status_code if hasattr(resp, 'status_code') else 200
    payload["connect"] = conn

    if ephemeral:
        payload["shutdown"] = _disconnect_mt5(kill_terminal)

    return jsonify(payload), status_code


@app.route('/api/invoke/account', methods=['POST'])
def invoke_account():
    """One-shot account info (connect -> read -> disconnect)."""
    data = request.json or {}
    ephemeral = data.get('ephemeral', True)
    kill_terminal = data.get('kill_terminal', ephemeral)

    conn = _ensure_mt5_connected(data)
    if not conn.get("success"):
        return jsonify({"phase": "connect", **conn}), 400

    acct = mt5.account_info()
    payload = {
        "success": bool(acct),
        "connect": conn,
        "account": acct._asdict() if acct else None,
    }
    if not acct:
        payload.update(err())

    if ephemeral:
        payload["shutdown"] = _disconnect_mt5(kill_terminal)

    return jsonify(payload), 200 if acct else 400

@app.route('/api/status', methods=['GET'])
def status():
    t = mt5.terminal_info()
    if t:
        return jsonify({"connected": True, "build": t.build, "name": t.name, 
                       "path": t.path, "trade_allowed": t.trade_allowed})
    return jsonify({"connected": False, **err()})

@app.route('/api/version', methods=['GET'])
def version():
    v = mt5.version()
    return jsonify({"version": v[0], "build": v[1], "date": v[2]}) if v else (jsonify(err()), 400)

# ============== Account ==============
@app.route('/api/account', methods=['GET'])
def account_info():
    a = mt5.account_info()
    if a:
        return jsonify({"login": a.login, "leverage": a.leverage, "balance": a.balance,
                       "equity": a.equity, "margin": a.margin, "margin_free": a.margin_free,
                       "margin_level": a.margin_level, "profit": a.profit, "currency": a.currency,
                       "server": a.server, "name": a.name, "company": a.company})
    return jsonify(err()), 400

# ============== Symbols ==============
@app.route('/api/symbols', methods=['GET'])
def get_symbols():
    group = request.args.get('group')
    symbols = mt5.symbols_get(group=group) if group else mt5.symbols_get()
    if symbols:
        return jsonify({"count": len(symbols), "symbols": [s.name for s in symbols]})
    return jsonify(err()), 400

@app.route('/api/symbol/<symbol>', methods=['GET'])
def symbol_info(symbol):
    s = mt5.symbol_info(symbol)
    if s:
        return jsonify({"name": s.name, "bid": s.bid, "ask": s.ask, "spread": s.spread,
                       "digits": s.digits, "point": s.point, "volume_min": s.volume_min,
                       "volume_max": s.volume_max, "volume_step": s.volume_step,
                       "trade_mode": s.trade_mode, "description": s.description,
                       "trade_stops_level": s.trade_stops_level, "trade_freeze_level": s.trade_freeze_level})
    return jsonify({"error": f"Symbol {symbol} not found"}), 404

@app.route('/api/symbol/<symbol>/select', methods=['POST'])
def symbol_select(symbol):
    data = request.json or {}
    if mt5.symbol_select(symbol, data.get('enable', True)):
        return jsonify({"success": True})
    return jsonify(err()), 400

# ============== Price ==============
@app.route('/api/price/<symbol>', methods=['GET'])
def get_price(symbol):
    tick = mt5.symbol_info_tick(symbol)
    if tick:
        return jsonify({"symbol": symbol, "bid": tick.bid, "ask": tick.ask, "last": tick.last,
                       "volume": tick.volume, "time": datetime.fromtimestamp(tick.time).isoformat()})
    return jsonify(err()), 400

@app.route('/api/tick/<symbol>', methods=['GET'])
def get_tick(symbol):
    return get_price(symbol)

# ============== Candles ==============
@app.route('/api/candles/<symbol>', methods=['GET'])
def get_candles(symbol):
    tf = request.args.get('timeframe', 'H1').upper()
    count = int(request.args.get('count', 100))
    if tf not in TIMEFRAMES:
        return jsonify({"error": f"Invalid timeframe. Valid: {list(TIMEFRAMES.keys())}"}), 400
    rates = mt5.copy_rates_from_pos(symbol, TIMEFRAMES[tf], 0, count)
    if rates is not None and len(rates) > 0:
        candles = [{"time": datetime.fromtimestamp(r['time']).isoformat(), "open": float(r['open']),
                   "high": float(r['high']), "low": float(r['low']), "close": float(r['close']),
                   "volume": int(r['tick_volume']), "spread": int(r['spread'])} for r in rates]
        return jsonify({"symbol": symbol, "timeframe": tf, "count": len(candles), "candles": candles})
    return jsonify(err()), 400

@app.route('/api/rates/<symbol>', methods=['GET'])
def get_rates(symbol):
    return get_candles(symbol)

# ============== Indicators ==============
def get_rates_data(symbol, tf, count):
    if tf not in TIMEFRAMES:
        return None, "Invalid timeframe"
    rates = mt5.copy_rates_from_pos(symbol, TIMEFRAMES[tf], 0, count)
    if rates is None or len(rates) == 0:
        return None, err()
    return rates, None

@app.route('/api/indicator/rsi/<symbol>', methods=['GET'])
def get_rsi(symbol):
    tf = request.args.get('timeframe', 'H1').upper()
    period = int(request.args.get('period', 14))
    count = int(request.args.get('count', 100))
    rates, error = get_rates_data(symbol, tf, count + period + 10)
    if error: return jsonify(error), 400
    closes = [float(r['close']) for r in rates]
    rsi = calc_rsi(closes, period)
    times = [datetime.fromtimestamp(r['time']).isoformat() for r in rates[period+1:]]
    n = min(len(times), len(rsi), count)
    result = [{"time": times[i], "rsi": round(rsi[i], 2)} for i in range(n)]
    return jsonify({"symbol": symbol, "timeframe": tf, "period": period, "count": len(result), "values": result[-count:]})

@app.route('/api/indicator/ma/<symbol>', methods=['GET'])
def get_ma(symbol):
    tf = request.args.get('timeframe', 'H1').upper()
    period = int(request.args.get('period', 14))
    ma_type = request.args.get('type', 'sma').lower()
    count = int(request.args.get('count', 100))
    applied = request.args.get('applied', 'close').lower()
    rates, error = get_rates_data(symbol, tf, count + period + 10)
    if error: return jsonify(error), 400
    if applied == 'hl2':
        data = [(float(r['high']) + float(r['low'])) / 2 for r in rates]
    elif applied == 'hlc3':
        data = [(float(r['high']) + float(r['low']) + float(r['close'])) / 3 for r in rates]
    elif applied == 'ohlc4':
        data = [(float(r['open']) + float(r['high']) + float(r['low']) + float(r['close'])) / 4 for r in rates]
    else:
        data = [float(r[applied]) for r in rates]
    ma_funcs = {'sma': calc_sma, 'ema': calc_ema, 'smma': calc_smma, 'lwma': calc_lwma}
    if ma_type not in ma_funcs:
        return jsonify({"error": f"Invalid MA type. Valid: {list(ma_funcs.keys())}"}), 400
    ma = ma_funcs[ma_type](data, period)
    times = [datetime.fromtimestamp(r['time']).isoformat() for r in rates[period-1:]]
    n = min(len(times), len(ma), count)
    result = [{"time": times[i], "value": round(ma[i], 5)} for i in range(n)]
    return jsonify({"symbol": symbol, "timeframe": tf, "period": period, "type": ma_type, "count": len(result), "values": result[-count:]})

@app.route('/api/indicator/macd/<symbol>', methods=['GET'])
def get_macd(symbol):
    tf = request.args.get('timeframe', 'H1').upper()
    fast = int(request.args.get('fast', 12))
    slow = int(request.args.get('slow', 26))
    signal = int(request.args.get('signal', 9))
    count = int(request.args.get('count', 100))
    rates, error = get_rates_data(symbol, tf, count + slow + signal + 50)
    if error: return jsonify(error), 400
    closes = [float(r['close']) for r in rates]
    macd = calc_macd(closes, fast, slow, signal)
    n = min(len(macd['signal']), count)
    times = [datetime.fromtimestamp(r['time']).isoformat() for r in rates[-n:]]
    result = [{"time": times[i], "macd": round(macd['macd'][-n:][i], 5), "signal": round(macd['signal'][-n:][i], 5), 
              "histogram": round(macd['histogram'][-n:][i], 5)} for i in range(n)]
    return jsonify({"symbol": symbol, "timeframe": tf, "fast": fast, "slow": slow, "signal_period": signal, "values": result})

@app.route('/api/indicator/bollinger/<symbol>', methods=['GET'])
def get_bollinger(symbol):
    tf = request.args.get('timeframe', 'H1').upper()
    period = int(request.args.get('period', 20))
    std_dev = float(request.args.get('std_dev', 2))
    count = int(request.args.get('count', 100))
    rates, error = get_rates_data(symbol, tf, count + period + 10)
    if error: return jsonify(error), 400
    closes = [float(r['close']) for r in rates]
    bb = calc_bollinger(closes, period, std_dev)
    n = min(len(bb['middle']), count)
    times = [datetime.fromtimestamp(r['time']).isoformat() for r in rates[-n:]]
    result = [{"time": times[i], "upper": round(bb['upper'][-n:][i], 5), "middle": round(bb['middle'][-n:][i], 5), 
              "lower": round(bb['lower'][-n:][i], 5)} for i in range(n)]
    return jsonify({"symbol": symbol, "timeframe": tf, "period": period, "std_dev": std_dev, "values": result})

@app.route('/api/indicator/stochastic/<symbol>', methods=['GET'])
def get_stochastic(symbol):
    tf = request.args.get('timeframe', 'H1').upper()
    k_period = int(request.args.get('k_period', 14))
    d_period = int(request.args.get('d_period', 3))
    count = int(request.args.get('count', 100))
    rates, error = get_rates_data(symbol, tf, count + k_period + d_period + 10)
    if error: return jsonify(error), 400
    high = [float(r['high']) for r in rates]
    low = [float(r['low']) for r in rates]
    close = [float(r['close']) for r in rates]
    stoch = calc_stochastic(high, low, close, k_period, d_period)
    n = min(len(stoch['d']), count)
    times = [datetime.fromtimestamp(r['time']).isoformat() for r in rates[-n:]]
    result = [{"time": times[i], "k": round(stoch['k'][-n:][i], 2), "d": round(stoch['d'][-n:][i], 2)} for i in range(n)]
    return jsonify({"symbol": symbol, "timeframe": tf, "k_period": k_period, "d_period": d_period, "values": result})

@app.route('/api/indicator/atr/<symbol>', methods=['GET'])
def get_atr(symbol):
    tf = request.args.get('timeframe', 'H1').upper()
    period = int(request.args.get('period', 14))
    count = int(request.args.get('count', 100))
    rates, error = get_rates_data(symbol, tf, count + period + 10)
    if error: return jsonify(error), 400
    high = [float(r['high']) for r in rates]
    low = [float(r['low']) for r in rates]
    close = [float(r['close']) for r in rates]
    atr = calc_atr(high, low, close, period)
    n = min(len(atr), count)
    times = [datetime.fromtimestamp(r['time']).isoformat() for r in rates[-n:]]
    result = [{"time": times[i], "atr": round(atr[-n:][i], 5)} for i in range(n)]
    return jsonify({"symbol": symbol, "timeframe": tf, "period": period, "values": result})

@app.route('/api/indicator/cci/<symbol>', methods=['GET'])
def get_cci(symbol):
    tf = request.args.get('timeframe', 'H1').upper()
    period = int(request.args.get('period', 20))
    count = int(request.args.get('count', 100))
    rates, error = get_rates_data(symbol, tf, count + period + 10)
    if error: return jsonify(error), 400
    high = [float(r['high']) for r in rates]
    low = [float(r['low']) for r in rates]
    close = [float(r['close']) for r in rates]
    cci = calc_cci(high, low, close, period)
    n = min(len(cci), count)
    times = [datetime.fromtimestamp(r['time']).isoformat() for r in rates[-n:]]
    result = [{"time": times[i], "cci": round(cci[-n:][i], 2)} for i in range(n)]
    return jsonify({"symbol": symbol, "timeframe": tf, "period": period, "values": result})

@app.route('/api/indicator/williams/<symbol>', methods=['GET'])
def get_williams(symbol):
    tf = request.args.get('timeframe', 'H1').upper()
    period = int(request.args.get('period', 14))
    count = int(request.args.get('count', 100))
    rates, error = get_rates_data(symbol, tf, count + period + 10)
    if error: return jsonify(error), 400
    high = [float(r['high']) for r in rates]
    low = [float(r['low']) for r in rates]
    close = [float(r['close']) for r in rates]
    wr = calc_williams_r(high, low, close, period)
    n = min(len(wr), count)
    times = [datetime.fromtimestamp(r['time']).isoformat() for r in rates[-n:]]
    result = [{"time": times[i], "williams_r": round(wr[-n:][i], 2)} for i in range(n)]
    return jsonify({"symbol": symbol, "timeframe": tf, "period": period, "values": result})

@app.route('/api/indicator/momentum/<symbol>', methods=['GET'])
def get_momentum(symbol):
    tf = request.args.get('timeframe', 'H1').upper()
    period = int(request.args.get('period', 10))
    count = int(request.args.get('count', 100))
    rates, error = get_rates_data(symbol, tf, count + period + 10)
    if error: return jsonify(error), 400
    closes = [float(r['close']) for r in rates]
    mom = calc_momentum(closes, period)
    n = min(len(mom), count)
    times = [datetime.fromtimestamp(r['time']).isoformat() for r in rates[-n:]]
    result = [{"time": times[i], "momentum": round(mom[-n:][i], 5)} for i in range(n)]
    return jsonify({"symbol": symbol, "timeframe": tf, "period": period, "values": result})

# ============== Live Indicators (includes current forming bar) ==============
@app.route('/api/live/rsi/<symbol>', methods=['GET'])
def get_live_rsi(symbol):
    """Get live RSI including current bar"""
    tf = request.args.get('timeframe', 'H1').upper()
    period = int(request.args.get('period', 14))
    if tf not in TIMEFRAMES:
        return jsonify({"error": "Invalid timeframe"}), 400
    rates = mt5.copy_rates_from_pos(symbol, TIMEFRAMES[tf], 0, period + 20)
    if rates is None or len(rates) < period + 1:
        return jsonify(err()), 400
    closes = [float(r['close']) for r in rates]
    rsi = calc_rsi(closes, period)
    if not rsi:
        return jsonify({"error": "Not enough data"}), 400
    tick = mt5.symbol_info_tick(symbol)
    return jsonify({
        "symbol": symbol, "timeframe": tf, "period": period, "live": True,
        "rsi": round(rsi[-1], 2),
        "price": tick.bid if tick else None,
        "time": datetime.fromtimestamp(rates[-1]['time']).isoformat()
    })

@app.route('/api/live/ma/<symbol>', methods=['GET'])
def get_live_ma(symbol):
    """Get live MA including current bar"""
    tf = request.args.get('timeframe', 'H1').upper()
    period = int(request.args.get('period', 14))
    ma_type = request.args.get('type', 'sma').lower()
    applied = request.args.get('applied', 'close').lower()
    if tf not in TIMEFRAMES:
        return jsonify({"error": "Invalid timeframe"}), 400
    rates = mt5.copy_rates_from_pos(symbol, TIMEFRAMES[tf], 0, period + 10)
    if rates is None or len(rates) < period:
        return jsonify(err()), 400
    if applied == 'hl2':
        data = [(float(r['high']) + float(r['low'])) / 2 for r in rates]
    elif applied == 'hlc3':
        data = [(float(r['high']) + float(r['low']) + float(r['close'])) / 3 for r in rates]
    elif applied == 'ohlc4':
        data = [(float(r['open']) + float(r['high']) + float(r['low']) + float(r['close'])) / 4 for r in rates]
    else:
        data = [float(r[applied]) for r in rates]
    ma_funcs = {'sma': calc_sma, 'ema': calc_ema, 'smma': calc_smma, 'lwma': calc_lwma}
    if ma_type not in ma_funcs:
        return jsonify({"error": f"Invalid MA type. Valid: {list(ma_funcs.keys())}"}), 400
    ma = ma_funcs[ma_type](data, period)
    if not ma:
        return jsonify({"error": "Not enough data"}), 400
    tick = mt5.symbol_info_tick(symbol)
    return jsonify({
        "symbol": symbol, "timeframe": tf, "period": period, "type": ma_type, "live": True,
        "value": round(ma[-1], 5),
        "price": tick.bid if tick else None,
        "time": datetime.fromtimestamp(rates[-1]['time']).isoformat()
    })

@app.route('/api/live/macd/<symbol>', methods=['GET'])
def get_live_macd(symbol):
    """Get live MACD including current bar"""
    tf = request.args.get('timeframe', 'H1').upper()
    fast = int(request.args.get('fast', 12))
    slow = int(request.args.get('slow', 26))
    signal = int(request.args.get('signal', 9))
    if tf not in TIMEFRAMES:
        return jsonify({"error": "Invalid timeframe"}), 400
    rates = mt5.copy_rates_from_pos(symbol, TIMEFRAMES[tf], 0, slow + signal + 20)
    if rates is None:
        return jsonify(err()), 400
    closes = [float(r['close']) for r in rates]
    macd = calc_macd(closes, fast, slow, signal)
    if not macd['signal']:
        return jsonify({"error": "Not enough data"}), 400
    tick = mt5.symbol_info_tick(symbol)
    return jsonify({
        "symbol": symbol, "timeframe": tf, "fast": fast, "slow": slow, "signal_period": signal, "live": True,
        "macd": round(macd['macd'][-1], 5),
        "signal": round(macd['signal'][-1], 5),
        "histogram": round(macd['histogram'][-1], 5),
        "price": tick.bid if tick else None,
        "time": datetime.fromtimestamp(rates[-1]['time']).isoformat()
    })

@app.route('/api/live/bollinger/<symbol>', methods=['GET'])
def get_live_bollinger(symbol):
    """Get live Bollinger Bands including current bar"""
    tf = request.args.get('timeframe', 'H1').upper()
    period = int(request.args.get('period', 20))
    std_dev = float(request.args.get('std_dev', 2))
    if tf not in TIMEFRAMES:
        return jsonify({"error": "Invalid timeframe"}), 400
    rates = mt5.copy_rates_from_pos(symbol, TIMEFRAMES[tf], 0, period + 10)
    if rates is None or len(rates) < period:
        return jsonify(err()), 400
    closes = [float(r['close']) for r in rates]
    bb = calc_bollinger(closes, period, std_dev)
    if not bb['middle']:
        return jsonify({"error": "Not enough data"}), 400
    tick = mt5.symbol_info_tick(symbol)
    return jsonify({
        "symbol": symbol, "timeframe": tf, "period": period, "std_dev": std_dev, "live": True,
        "upper": round(bb['upper'][-1], 5),
        "middle": round(bb['middle'][-1], 5),
        "lower": round(bb['lower'][-1], 5),
        "price": tick.bid if tick else None,
        "time": datetime.fromtimestamp(rates[-1]['time']).isoformat()
    })

@app.route('/api/live/stochastic/<symbol>', methods=['GET'])
def get_live_stochastic(symbol):
    """Get live Stochastic including current bar"""
    tf = request.args.get('timeframe', 'H1').upper()
    k_period = int(request.args.get('k_period', 14))
    d_period = int(request.args.get('d_period', 3))
    if tf not in TIMEFRAMES:
        return jsonify({"error": "Invalid timeframe"}), 400
    rates = mt5.copy_rates_from_pos(symbol, TIMEFRAMES[tf], 0, k_period + d_period + 10)
    if rates is None:
        return jsonify(err()), 400
    high = [float(r['high']) for r in rates]
    low = [float(r['low']) for r in rates]
    close = [float(r['close']) for r in rates]
    stoch = calc_stochastic(high, low, close, k_period, d_period)
    if not stoch['d']:
        return jsonify({"error": "Not enough data"}), 400
    tick = mt5.symbol_info_tick(symbol)
    return jsonify({
        "symbol": symbol, "timeframe": tf, "k_period": k_period, "d_period": d_period, "live": True,
        "k": round(stoch['k'][-1], 2),
        "d": round(stoch['d'][-1], 2),
        "price": tick.bid if tick else None,
        "time": datetime.fromtimestamp(rates[-1]['time']).isoformat()
    })

@app.route('/api/live/atr/<symbol>', methods=['GET'])
def get_live_atr(symbol):
    """Get live ATR including current bar"""
    tf = request.args.get('timeframe', 'H1').upper()
    period = int(request.args.get('period', 14))
    if tf not in TIMEFRAMES:
        return jsonify({"error": "Invalid timeframe"}), 400
    rates = mt5.copy_rates_from_pos(symbol, TIMEFRAMES[tf], 0, period + 10)
    if rates is None:
        return jsonify(err()), 400
    high = [float(r['high']) for r in rates]
    low = [float(r['low']) for r in rates]
    close = [float(r['close']) for r in rates]
    atr = calc_atr(high, low, close, period)
    if not atr:
        return jsonify({"error": "Not enough data"}), 400
    tick = mt5.symbol_info_tick(symbol)
    return jsonify({
        "symbol": symbol, "timeframe": tf, "period": period, "live": True,
        "atr": round(atr[-1], 5),
        "price": tick.bid if tick else None,
        "time": datetime.fromtimestamp(rates[-1]['time']).isoformat()
    })

@app.route('/api/live/cci/<symbol>', methods=['GET'])
def get_live_cci(symbol):
    """Get live CCI including current bar"""
    tf = request.args.get('timeframe', 'H1').upper()
    period = int(request.args.get('period', 20))
    if tf not in TIMEFRAMES:
        return jsonify({"error": "Invalid timeframe"}), 400
    rates = mt5.copy_rates_from_pos(symbol, TIMEFRAMES[tf], 0, period + 10)
    if rates is None:
        return jsonify(err()), 400
    high = [float(r['high']) for r in rates]
    low = [float(r['low']) for r in rates]
    close = [float(r['close']) for r in rates]
    cci = calc_cci(high, low, close, period)
    if not cci:
        return jsonify({"error": "Not enough data"}), 400
    tick = mt5.symbol_info_tick(symbol)
    return jsonify({
        "symbol": symbol, "timeframe": tf, "period": period, "live": True,
        "cci": round(cci[-1], 2),
        "price": tick.bid if tick else None,
        "time": datetime.fromtimestamp(rates[-1]['time']).isoformat()
    })

@app.route('/api/live/all/<symbol>', methods=['GET'])
def get_live_all(symbol):
    """Get all live indicators in one call"""
    tf = request.args.get('timeframe', 'H1').upper()
    if tf not in TIMEFRAMES:
        return jsonify({"error": "Invalid timeframe"}), 400
    
    # Get enough data for all indicators
    rates = mt5.copy_rates_from_pos(symbol, TIMEFRAMES[tf], 0, 50)
    if rates is None or len(rates) < 30:
        return jsonify(err()), 400
    
    closes = [float(r['close']) for r in rates]
    high = [float(r['high']) for r in rates]
    low = [float(r['low']) for r in rates]
    
    tick = mt5.symbol_info_tick(symbol)
    
    # Calculate all indicators
    rsi = calc_rsi(closes, 14)
    sma = calc_sma(closes, 14)
    ema = calc_ema(closes, 14)
    macd = calc_macd(closes, 12, 26, 9)
    bb = calc_bollinger(closes, 20, 2)
    stoch = calc_stochastic(high, low, closes, 14, 3)
    atr = calc_atr(high, low, closes, 14)
    
    result = {
        "symbol": symbol,
        "timeframe": tf,
        "live": True,
        "price": {"bid": tick.bid, "ask": tick.ask} if tick else None,
        "time": datetime.fromtimestamp(rates[-1]['time']).isoformat(),
        "indicators": {}
    }
    
    if rsi: result["indicators"]["rsi"] = round(rsi[-1], 2)
    if sma: result["indicators"]["sma14"] = round(sma[-1], 5)
    if ema: result["indicators"]["ema14"] = round(ema[-1], 5)
    if macd['signal']:
        result["indicators"]["macd"] = {
            "macd": round(macd['macd'][-1], 5),
            "signal": round(macd['signal'][-1], 5),
            "histogram": round(macd['histogram'][-1], 5)
        }
    if bb['middle']:
        result["indicators"]["bollinger"] = {
            "upper": round(bb['upper'][-1], 5),
            "middle": round(bb['middle'][-1], 5),
            "lower": round(bb['lower'][-1], 5)
        }
    if stoch['d']:
        result["indicators"]["stochastic"] = {
            "k": round(stoch['k'][-1], 2),
            "d": round(stoch['d'][-1], 2)
        }
    if atr: result["indicators"]["atr"] = round(atr[-1], 5)
    
    return jsonify(result)

# ============== Trading ==============
@app.route('/api/trade/open', methods=['POST'])
def open_trade():
    """Open trade with SL/TP in points or price. Use 0 for no SL/TP."""
    data = request.json or {}
    symbol = data.get('symbol')
    order_type = data.get('type', 'buy').lower()
    volume = float(data.get('volume', 0.01))
    sl = float(data.get('sl', 0))
    tp = float(data.get('tp', 0))
    sl_points = data.get('sl_points')
    tp_points = data.get('tp_points')
    price = data.get('price')
    deviation = int(data.get('deviation', 20))
    magic = int(data.get('magic', 0))
    comment = data.get('comment', 'API')
    
    if not symbol:
        return jsonify({"error": "Symbol required"}), 400
    
    info = mt5.symbol_info(symbol)
    if not info:
        return jsonify({"error": f"Symbol {symbol} not found"}), 404
    if not info.visible:
        mt5.symbol_select(symbol, True)
    
    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        return jsonify(err()), 400
    
    point = info.point
    stops_level = info.trade_stops_level  # Minimum distance for SL/TP in points
    
    # Ensure SL/TP points are at least the minimum required
    if sl_points and sl_points > 0 and sl_points < stops_level:
        sl_points = stops_level + 10  # Add buffer
        print(f"[TRADE] SL points adjusted to minimum: {sl_points}")
    if tp_points and tp_points > 0 and tp_points < stops_level:
        tp_points = stops_level + 10  # Add buffer
        print(f"[TRADE] TP points adjusted to minimum: {tp_points}")
    
    if order_type == 'buy':
        trade_type, trade_price = mt5.ORDER_TYPE_BUY, price or tick.ask
        if sl_points and sl_points > 0: sl = trade_price - sl_points * point
        if tp_points and tp_points > 0: tp = trade_price + tp_points * point
    elif order_type == 'sell':
        trade_type, trade_price = mt5.ORDER_TYPE_SELL, price or tick.bid
        if sl_points and sl_points > 0: sl = trade_price + sl_points * point
        if tp_points and tp_points > 0: tp = trade_price - tp_points * point
    elif order_type == 'buy_limit':
        trade_type, trade_price = mt5.ORDER_TYPE_BUY_LIMIT, price
        if sl_points and sl_points > 0: sl = trade_price - sl_points * point
        if tp_points and tp_points > 0: tp = trade_price + tp_points * point
    elif order_type == 'sell_limit':
        trade_type, trade_price = mt5.ORDER_TYPE_SELL_LIMIT, price
        if sl_points and sl_points > 0: sl = trade_price + sl_points * point
        if tp_points and tp_points > 0: tp = trade_price - tp_points * point
    elif order_type == 'buy_stop':
        trade_type, trade_price = mt5.ORDER_TYPE_BUY_STOP, price
        if sl_points and sl_points > 0: sl = trade_price - sl_points * point
        if tp_points and tp_points > 0: tp = trade_price + tp_points * point
    elif order_type == 'sell_stop':
        trade_type, trade_price = mt5.ORDER_TYPE_SELL_STOP, price
        if sl_points and sl_points > 0: sl = trade_price + sl_points * point
        if tp_points and tp_points > 0: tp = trade_price - tp_points * point
    else:
        return jsonify({"error": f"Invalid order type: {order_type}"}), 400
    
    if trade_price is None:
        return jsonify({"error": "Price required for pending orders"}), 400
    
    sl = round(sl, info.digits) if sl > 0 else 0.0
    tp = round(tp, info.digits) if tp > 0 else 0.0
    
    # Debug logging
    print(f"[TRADE] Opening {order_type} {symbol} @ {trade_price}")
    print(f"[TRADE] SL Points: {sl_points}, TP Points: {tp_points}")
    print(f"[TRADE] Calculated SL: {sl}, TP: {tp}, Point: {point}")
    
    req = {
        "action": mt5.TRADE_ACTION_DEAL if order_type in ['buy', 'sell'] else mt5.TRADE_ACTION_PENDING,
        "symbol": symbol, "volume": volume, "type": trade_type, "price": trade_price,
        "sl": sl, "tp": tp, "deviation": deviation, "magic": magic, "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_IOC
    }
    
    print(f"[TRADE] Request: {req}")
    
    result = mt5.order_send(req)
    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        print(f"[TRADE] Success! Ticket: {result.order}")
        return jsonify({"success": True, "ticket": result.order, "result": result_to_dict(result),
                       "debug": {"sl": sl, "tp": tp, "sl_points": sl_points, "tp_points": tp_points}})
    print(f"[TRADE] Failed: {result}")
    return jsonify({"success": False, "result": result_to_dict(result), **err()}), 400

@app.route('/api/trade/close', methods=['POST'])
def close_trade():
    """Close trade by ticket number"""
    data = request.json or {}
    ticket = data.get('ticket')
    volume = data.get('volume')
    deviation = int(data.get('deviation', 20))
    comment = data.get('comment', 'API Close')
    
    if not ticket:
        return jsonify({"error": "Ticket required"}), 400
    
    positions = mt5.positions_get(ticket=int(ticket))
    if not positions:
        return jsonify({"error": "Position not found"}), 404
    
    pos = positions[0]
    tick = mt5.symbol_info_tick(pos.symbol)
    if not tick:
        return jsonify(err()), 400
    
    close_price = tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask
    close_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
    close_volume = volume if volume else pos.volume
    
    req = {
        "action": mt5.TRADE_ACTION_DEAL, "symbol": pos.symbol, "volume": close_volume,
        "type": close_type, "position": pos.ticket, "price": close_price,
        "deviation": deviation, "comment": comment, "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC
    }
    
    result = mt5.order_send(req)
    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        return jsonify({"success": True, "result": result_to_dict(result)})
    return jsonify({"success": False, "result": result_to_dict(result), **err()}), 400

@app.route('/api/trade/close_all', methods=['POST'])
def close_all_trades():
    """Close all positions or by symbol/magic"""
    data = request.json or {}
    symbol = data.get('symbol')
    magic = data.get('magic')
    
    positions = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
    if not positions:
        return jsonify({"message": "No positions to close", "closed": 0})
    
    if magic:
        positions = [p for p in positions if p.magic == int(magic)]
    
    results = []
    for pos in positions:
        tick = mt5.symbol_info_tick(pos.symbol)
        if not tick:
            results.append({"ticket": pos.ticket, "success": False, "error": "No tick"})
            continue
        close_price = tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask
        close_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        req = {"action": mt5.TRADE_ACTION_DEAL, "symbol": pos.symbol, "volume": pos.volume,
               "type": close_type, "position": pos.ticket, "price": close_price,
               "deviation": 20, "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_IOC}
        r = mt5.order_send(req)
        results.append({"ticket": pos.ticket, "success": r and r.retcode == mt5.TRADE_RETCODE_DONE})
    
    return jsonify({"closed": sum(1 for r in results if r['success']), "results": results})

@app.route('/api/trade/modify', methods=['POST'])
def modify_trade():
    """Modify SL/TP of open position.
    For points: 0 = keep current, -1 = remove, positive = set points from price
    For price: 0 = remove SL/TP, positive = set exact price
    """
    data = request.json or {}
    ticket = data.get('ticket')
    sl = data.get('sl')
    tp = data.get('tp')
    sl_points = data.get('sl_points')
    tp_points = data.get('tp_points')
    
    if not ticket:
        return jsonify({"error": "Ticket required"}), 400
    
    positions = mt5.positions_get(ticket=int(ticket))
    if not positions:
        return jsonify({"error": "Position not found"}), 404
    
    pos = positions[0]
    info = mt5.symbol_info(pos.symbol)
    point = info.point
    
    # Start with current values
    new_sl = pos.sl
    new_tp = pos.tp
    
    # Handle SL points: 0=keep, -1=remove, >0=set (from ENTRY price)
    if sl_points is not None:
        if sl_points == 0:
            new_sl = pos.sl  # Keep current
        elif sl_points < 0:
            new_sl = 0.0  # Remove SL
        else:
            if pos.type == mt5.ORDER_TYPE_BUY:
                new_sl = pos.price_open - sl_points * point  # SL below entry for BUY
            else:
                new_sl = pos.price_open + sl_points * point  # SL above entry for SELL
    # Handle SL price directly
    elif sl is not None:
        new_sl = float(sl)
    
    # Handle TP points: 0=keep, -1=remove, >0=set (from ENTRY price)
    if tp_points is not None:
        if tp_points == 0:
            new_tp = pos.tp  # Keep current
        elif tp_points < 0:
            new_tp = 0.0  # Remove TP
        else:
            if pos.type == mt5.ORDER_TYPE_BUY:
                new_tp = pos.price_open + tp_points * point  # TP above entry for BUY
            else:
                new_tp = pos.price_open - tp_points * point  # TP below entry for SELL
    # Handle TP price directly
    elif tp is not None:
        new_tp = float(tp)
    
    # Round to symbol digits
    new_sl = round(new_sl, info.digits)
    new_tp = round(new_tp, info.digits)
    
    req = {"action": mt5.TRADE_ACTION_SLTP, "symbol": pos.symbol, "position": pos.ticket,
           "sl": new_sl, "tp": new_tp}
    
    result = mt5.order_send(req)
    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        return jsonify({"success": True, "sl": new_sl, "tp": new_tp, "result": result_to_dict(result)})
    return jsonify({"success": False, "result": result_to_dict(result), **err()}), 400

@app.route('/api/order/cancel', methods=['POST'])
@app.route('/api/order/cancel/<int:ticket>', methods=['POST', 'DELETE'])
def cancel_order(ticket=None):
    """Cancel pending order"""
    if ticket is None:
        data = request.json or {}
        ticket = data.get('ticket')
    if not ticket:
        return jsonify({"error": "Ticket required"}), 400
    req = {"action": mt5.TRADE_ACTION_REMOVE, "order": int(ticket)}
    result = mt5.order_send(req)
    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        return jsonify({"success": True, "result": result_to_dict(result)})
    return jsonify({"success": False, "result": result_to_dict(result), **err()}), 400

# Alias for pending order placement
@app.route('/api/trade/pending', methods=['POST'])
def place_pending_order():
    """Alias for /api/trade - places pending orders"""
    return open_trade()

# ============== Positions & Orders ==============
@app.route('/api/positions', methods=['GET'])
def get_positions():
    symbol = request.args.get('symbol')
    ticket = request.args.get('ticket')
    magic = request.args.get('magic')
    if ticket:
        positions = mt5.positions_get(ticket=int(ticket))
    elif symbol:
        positions = mt5.positions_get(symbol=symbol)
    else:
        positions = mt5.positions_get()
    if positions is not None:
        result = [pos_to_dict(p) for p in positions]
        if magic:
            result = [p for p in result if p['magic'] == int(magic)]
        total_profit = sum(p['profit'] for p in result)
        return jsonify({"count": len(result), "total_profit": total_profit, "positions": result})
    return jsonify(err()), 400

@app.route('/api/orders', methods=['GET'])
def get_orders():
    symbol = request.args.get('symbol')
    ticket = request.args.get('ticket')
    if ticket:
        orders = mt5.orders_get(ticket=int(ticket))
    elif symbol:
        orders = mt5.orders_get(symbol=symbol)
    else:
        orders = mt5.orders_get()
    if orders is not None:
        return jsonify({"count": len(orders), "orders": [order_to_dict(o) for o in orders]})
    return jsonify(err()), 400

@app.route('/api/history/sync', methods=['POST'])
def sync_history():
    """Force sync history from broker by re-requesting it"""
    try:
        # Try to trigger history download by requesting a very long period
        from_dt = datetime(2020, 1, 1)
        to_dt = datetime.now() + timedelta(days=1)
        
        # Request deals which forces MT5 to sync from server
        deals = mt5.history_deals_get(from_dt, to_dt)
        orders = mt5.history_orders_get(from_dt, to_dt)
        
        deal_count = len(deals) if deals else 0
        order_count = len(orders) if orders else 0
        
        return jsonify({
            "success": True,
            "message": "History synced",
            "deals_found": deal_count,
            "orders_found": order_count
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/history/deals', methods=['GET'])
def get_history_deals():
    days = int(request.args.get('days', 30))
    symbol = request.args.get('symbol')
    
    # Use a wider date range to catch all history
    from_dt = datetime(2020, 1, 1) if days > 365 else datetime.now() - timedelta(days=days)
    to_dt = datetime.now() + timedelta(days=1)  # Include today fully
    
    if symbol:
        deals = mt5.history_deals_get(from_dt, to_dt, group=f"*{symbol}*")
    else:
        deals = mt5.history_deals_get(from_dt, to_dt)
    if deals is not None:
        result = [deal_to_dict(d) for d in deals]
        total_profit = sum(d['profit'] for d in result)
        return jsonify({"count": len(result), "total_profit": total_profit, "deals": result})
    return jsonify(err()), 400

@app.route('/api/history/orders', methods=['GET'])
def get_history_orders():
    days = int(request.args.get('days', 30))
    from_dt = datetime.now() - timedelta(days=days)
    to_dt = datetime.now()
    orders = mt5.history_orders_get(from_dt, to_dt)
    if orders is not None:
        return jsonify({"count": len(orders), "orders": [order_to_dict(o) for o in orders]})
    return jsonify(err()), 400

# ============== Utilities ==============
@app.route('/api/calc/margin', methods=['POST'])
def calc_margin():
    data = request.json or {}
    symbol, volume = data.get('symbol'), float(data.get('volume', 0.01))
    order_type = data.get('type', 'buy').lower()
    if not symbol:
        return jsonify({"error": "Symbol required"}), 400
    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        return jsonify(err()), 400
    price = tick.ask if order_type == 'buy' else tick.bid
    action = mt5.ORDER_TYPE_BUY if order_type == 'buy' else mt5.ORDER_TYPE_SELL
    margin = mt5.order_calc_margin(action, symbol, volume, price)
    return jsonify({"margin": margin}) if margin else (jsonify(err()), 400)

@app.route('/api/calc/profit', methods=['POST'])
def calc_profit():
    data = request.json or {}
    symbol = data.get('symbol')
    volume = float(data.get('volume', 0.01))
    order_type = data.get('type', 'buy').lower()
    price_open = float(data.get('price_open', 0))
    price_close = float(data.get('price_close', 0))
    if not symbol or not price_open or not price_close:
        return jsonify({"error": "Symbol, price_open, price_close required"}), 400
    action = mt5.ORDER_TYPE_BUY if order_type == 'buy' else mt5.ORDER_TYPE_SELL
    profit = mt5.order_calc_profit(action, symbol, volume, price_open, price_close)
    return jsonify({"profit": profit}) if profit is not None else (jsonify(err()), 400)

@app.route('/api/market/book/<symbol>', methods=['GET'])
def get_book(symbol):
    if not mt5.market_book_add(symbol):
        return jsonify(err()), 400
    book = mt5.market_book_get(symbol)
    mt5.market_book_release(symbol)
    if book:
        return jsonify({"symbol": symbol, "book": [{"type": b.type, "price": b.price, "volume": b.volume} for b in book]})
    return jsonify(err()), 400

# ============== Trailing Stop Manager ==============
trailing_config = {
    "enabled": False,
    "points": 0,
    "magic": 0  # 0 = all positions
}

@app.route('/api/trailing/set', methods=['POST'])
@app.route('/set_trailing_sl', methods=['POST'])  # Legacy endpoint for GUI compatibility
def set_trailing_sl():
    """Enable trailing stop loss for all positions"""
    data = request.json or {}
    points = data.get('points') or request.args.get('points', 50)
    magic = data.get('magic') or request.args.get('magic', 0)
    
    trailing_config['enabled'] = True
    trailing_config['points'] = int(points)
    trailing_config['magic'] = int(magic)
    
    return jsonify({
        "ok": True,
        "status": "enabled",
        "points": trailing_config['points'],
        "magic": trailing_config['magic'],
        "message": f"Trailing SL enabled at {points} points"
    })

@app.route('/api/trailing/disable', methods=['POST'])
@app.route('/disable_trailing_sl', methods=['POST'])  # Legacy endpoint for GUI compatibility
def disable_trailing_sl():
    """Disable trailing stop loss"""
    trailing_config['enabled'] = False
    trailing_config['points'] = 0
    
    return jsonify({
        "ok": True,
        "status": "disabled",
        "message": "Trailing SL disabled"
    })

@app.route('/api/trailing/status', methods=['GET'])
@app.route('/trailing_status', methods=['GET'])  # Legacy endpoint for GUI compatibility
def get_trailing_status():
    """Get current trailing stop status"""
    return jsonify({
        "ok": True,
        "enabled": trailing_config['enabled'],
        "status": "enabled" if trailing_config['enabled'] else "disabled",
        "points": trailing_config['points'],
        "magic": trailing_config['magic']
    })

@app.route('/api/trailing/apply', methods=['POST'])
def apply_trailing_sl():
    """Apply trailing stop to all open positions (call periodically)"""
    if not trailing_config['enabled']:
        return jsonify({"ok": False, "message": "Trailing SL not enabled"})
    
    points = trailing_config['points']
    magic = trailing_config['magic']
    
    positions = mt5.positions_get()
    if not positions:
        return jsonify({"ok": True, "message": "No positions", "updated": 0})
    
    updated = 0
    for pos in positions:
        # Filter by magic if specified
        if magic and pos.magic != magic:
            continue
        
        info = mt5.symbol_info(pos.symbol)
        if not info:
            continue
        
        point = info.point
        tick = mt5.symbol_info_tick(pos.symbol)
        if not tick:
            continue
        
        new_sl = pos.sl
        
        if pos.type == mt5.ORDER_TYPE_BUY:
            # For BUY: trail SL below bid price
            trail_level = tick.bid - points * point
            if trail_level > pos.sl and trail_level > pos.price_open:
                new_sl = round(trail_level, info.digits)
        else:
            # For SELL: trail SL above ask price
            trail_level = tick.ask + points * point
            if (pos.sl == 0 or trail_level < pos.sl) and trail_level < pos.price_open:
                new_sl = round(trail_level, info.digits)
        
        if new_sl != pos.sl and new_sl > 0:
            req = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": pos.symbol,
                "position": pos.ticket,
                "sl": new_sl,
                "tp": pos.tp
            }
            result = mt5.order_send(req)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                updated += 1
    
    return jsonify({"ok": True, "updated": updated, "total_positions": len(positions)})

# ============== Trend Analysis ==============
def calculate_trend(rates):
    """Calculate trend from candle data: UP, DOWN, or NEUTRAL"""
    if rates is None or len(rates) < 3:
        return "NEUTRAL"
    
    # Use last 3 candles for trend determination
    closes = [float(r['close']) for r in rates[-3:]]
    opens = [float(r['open']) for r in rates[-3:]]
    
    # Simple trend: compare first and last close
    change = closes[-1] - closes[0]
    avg_body = sum(abs(c - o) for c, o in zip(closes, opens)) / len(closes)
    
    # Need significant movement relative to average body size
    if avg_body > 0 and abs(change) > avg_body * 0.5:
        return "UP" if change > 0 else "DOWN"
    
    # Also check EMA trend
    if len(rates) >= 5:
        ema_short = sum(float(r['close']) for r in rates[-3:]) / 3
        ema_long = sum(float(r['close']) for r in rates[-5:]) / 5
        if ema_short > ema_long * 1.001:
            return "UP"
        elif ema_short < ema_long * 0.999:
            return "DOWN"
    
    return "NEUTRAL"

@app.route('/api/trend/<symbol>', methods=['GET'])
def get_trend(symbol):
    """Get trend analysis for multiple timeframes"""
    timeframes = ['M1', 'M5', 'M15', 'M30', 'H1', 'D1']
    trends = {}
    
    # Ensure symbol is selected
    info = mt5.symbol_info(symbol)
    if not info:
        # Try with common suffixes
        for suffix in ['', '.pr', '.m', '.r', '.pro', '.raw']:
            test_symbol = symbol + suffix if suffix else symbol
            info = mt5.symbol_info(test_symbol)
            if info:
                symbol = test_symbol
                break
    
    if not info:
        return jsonify({"error": f"Symbol {symbol} not found"}), 404
    
    if not info.visible:
        mt5.symbol_select(symbol, True)
    
    for tf in timeframes:
        if tf not in TIMEFRAMES:
            trends[tf] = "NEUTRAL"
            continue
        
        rates = mt5.copy_rates_from_pos(symbol, TIMEFRAMES[tf], 0, 10)
        if rates is not None and len(rates) > 0:
            trends[tf] = calculate_trend(rates)
        else:
            trends[tf] = "NEUTRAL"
    
    # Get current price
    tick = mt5.symbol_info_tick(symbol)
    price = tick.bid if tick else 0
    
    return jsonify({
        "symbol": symbol,
        "price": price,
        "trends": trends,
        "overall": determine_overall_trend(trends)
    })

def determine_overall_trend(trends):
    """Determine overall trend from all timeframes"""
    up_count = sum(1 for t in trends.values() if t == "UP")
    down_count = sum(1 for t in trends.values() if t == "DOWN")
    
    if up_count >= 4:
        return "STRONG_UP"
    elif down_count >= 4:
        return "STRONG_DOWN"
    elif up_count > down_count:
        return "UP"
    elif down_count > up_count:
        return "DOWN"
    return "NEUTRAL"


# ============== SMC (Smart Money Concepts) Analysis ==============

def find_swing_points(rates, lookback=5):
    """Find Swing Highs and Swing Lows using fractal method"""
    swing_highs = []
    swing_lows = []
    
    if rates is None or len(rates) < lookback * 2 + 1:
        return swing_highs, swing_lows
    
    for i in range(lookback, len(rates) - lookback):
        # Check for Swing High
        is_swing_high = True
        is_swing_low = True
        
        current_high = rates[i]['high']
        current_low = rates[i]['low']
        
        for j in range(1, lookback + 1):
            if rates[i - j]['high'] >= current_high or rates[i + j]['high'] >= current_high:
                is_swing_high = False
            if rates[i - j]['low'] <= current_low or rates[i + j]['low'] <= current_low:
                is_swing_low = False
        
        if is_swing_high:
            swing_highs.append({
                'index': i,
                'time': int(rates[i]['time']),
                'price': float(current_high),
                'type': 'HIGH'
            })
        
        if is_swing_low:
            swing_lows.append({
                'index': i,
                'time': int(rates[i]['time']),
                'price': float(current_low),
                'type': 'LOW'
            })
    
    return swing_highs, swing_lows


def find_order_blocks(rates, swing_highs, swing_lows):
    """Find Order Blocks (OB) - Last bullish/bearish candle before impulsive move"""
    order_blocks = []
    
    if rates is None or len(rates) < 10:
        return order_blocks
    
    # Look for Bullish Order Blocks (last bearish candle before bullish swing low)
    for swing in swing_lows[-5:]:  # Last 5 swing lows
        idx = swing['index']
        if idx < 3:
            continue
        
        # Find the last bearish candle before the swing low
        for i in range(idx - 1, max(0, idx - 5), -1):
            if rates[i]['close'] < rates[i]['open']:  # Bearish candle
                # Check if next candles made impulsive move up
                if idx + 2 < len(rates):
                    move_up = rates[idx + 2]['close'] - rates[i]['low']
                    body_size = abs(rates[i]['close'] - rates[i]['open'])
                    if move_up > body_size * 2:  # Impulsive move
                        order_blocks.append({
                            'type': 'BULLISH_OB',
                            'time': int(rates[i]['time']),
                            'high': float(rates[i]['high']),
                            'low': float(rates[i]['low']),
                            'open': float(rates[i]['open']),
                            'close': float(rates[i]['close']),
                            'mitigated': False
                        })
                        break
    
    # Look for Bearish Order Blocks (last bullish candle before bearish swing high)
    for swing in swing_highs[-5:]:  # Last 5 swing highs
        idx = swing['index']
        if idx < 3:
            continue
        
        # Find the last bullish candle before the swing high
        for i in range(idx - 1, max(0, idx - 5), -1):
            if rates[i]['close'] > rates[i]['open']:  # Bullish candle
                # Check if next candles made impulsive move down
                if idx + 2 < len(rates):
                    move_down = rates[i]['high'] - rates[idx + 2]['close']
                    body_size = abs(rates[i]['close'] - rates[i]['open'])
                    if move_down > body_size * 2:  # Impulsive move
                        order_blocks.append({
                            'type': 'BEARISH_OB',
                            'time': int(rates[i]['time']),
                            'high': float(rates[i]['high']),
                            'low': float(rates[i]['low']),
                            'open': float(rates[i]['open']),
                            'close': float(rates[i]['close']),
                            'mitigated': False
                        })
                        break
    
    return order_blocks


def calculate_fibonacci_levels(swing_high, swing_low, direction='bullish'):
    """Calculate Fibonacci retracement levels"""
    if direction == 'bullish':
        # For bullish fib, measure from swing low to swing high
        diff = swing_high - swing_low
        return {
            '0.0': round(swing_low, 5),
            '0.236': round(swing_low + diff * 0.236, 5),
            '0.382': round(swing_low + diff * 0.382, 5),
            '0.5': round(swing_low + diff * 0.5, 5),
            '0.618': round(swing_low + diff * 0.618, 5),
            '0.786': round(swing_low + diff * 0.786, 5),
            '1.0': round(swing_high, 5),
            '1.272': round(swing_high + diff * 0.272, 5),
            '1.618': round(swing_high + diff * 0.618, 5),
        }
    else:
        # For bearish fib, measure from swing high to swing low
        diff = swing_high - swing_low
        return {
            '0.0': round(swing_high, 5),
            '0.236': round(swing_high - diff * 0.236, 5),
            '0.382': round(swing_high - diff * 0.382, 5),
            '0.5': round(swing_high - diff * 0.5, 5),
            '0.618': round(swing_high - diff * 0.618, 5),
            '0.786': round(swing_high - diff * 0.786, 5),
            '1.0': round(swing_low, 5),
            '1.272': round(swing_low - diff * 0.272, 5),
            '1.618': round(swing_low - diff * 0.618, 5),
        }


def generate_smc_signal(current_price, swing_highs, swing_lows, order_blocks, fib_levels, trend):
    """Generate trading signal based on SMC analysis"""
    signal = {
        'action': 'WAIT',
        'confidence': 0,
        'reason': '',
        'entry': None,
        'sl': None,
        'tp': None,
        'rr': None
    }
    
    if not swing_highs or not swing_lows or not order_blocks:
        signal['reason'] = 'Insufficient data for SMC analysis'
        return signal
    
    # Get recent swing points
    last_swing_high = swing_highs[-1]['price'] if swing_highs else None
    last_swing_low = swing_lows[-1]['price'] if swing_lows else None
    
    # Check for bullish setup
    bullish_obs = [ob for ob in order_blocks if ob['type'] == 'BULLISH_OB']
    bearish_obs = [ob for ob in order_blocks if ob['type'] == 'BEARISH_OB']
    
    # Bullish signal: Price near bullish OB + bullish trend + near fib 0.618/0.5
    if bullish_obs and trend in ['UP', 'STRONG_UP']:
        for ob in bullish_obs[-2:]:
            ob_zone_top = ob['high']
            ob_zone_bottom = ob['low']
            ob_range = ob_zone_top - ob_zone_bottom
            
            # Check if price is in or APPROACHING the OB zone (within 1.5x of zone height)
            zone_buffer = ob_range * 1.5  # 150% buffer for approaching
            if ob_zone_bottom - zone_buffer <= current_price <= ob_zone_top * 1.01:
                # Check if near fib 0.618 or 0.5
                fib_618 = fib_levels.get('0.618', 0)
                fib_5 = fib_levels.get('0.5', 0)
                
                near_fib = abs(current_price - fib_618) / current_price < 0.005 or \
                          abs(current_price - fib_5) / current_price < 0.005
                
                confidence = 60
                if near_fib:
                    confidence += 25
                if trend == 'STRONG_UP':
                    confidence += 15
                
                sl_distance = current_price - ob_zone_bottom
                tp_distance = sl_distance * 2  # 1:2 RR
                
                signal = {
                    'action': 'BUY',
                    'confidence': min(confidence, 95),
                    'reason': f"Bullish OB + {'FIB confluence' if near_fib else 'Trend'} + {trend}",
                    'entry': round(current_price, 5),
                    'sl': round(ob_zone_bottom - sl_distance * 0.1, 5),
                    'tp': round(current_price + tp_distance, 5),
                    'rr': '1:2',
                    'ob_zone': [ob_zone_bottom, ob_zone_top]
                }
                break
    
    # Bearish signal: Price near bearish OB + bearish trend + near fib 0.618/0.5
    elif bearish_obs and trend in ['DOWN', 'STRONG_DOWN']:
        for ob in bearish_obs[-2:]:
            ob_zone_top = ob['high']
            ob_zone_bottom = ob['low']
            ob_range = ob_zone_top - ob_zone_bottom
            
            # Check if price is in or APPROACHING the OB zone (within 1.5x of zone height)
            zone_buffer = ob_range * 1.5  # 150% buffer for approaching
            if ob_zone_bottom * 0.99 <= current_price <= ob_zone_top + zone_buffer:
                # Check if near fib 0.618 or 0.5
                fib_618 = fib_levels.get('0.618', 0)
                fib_5 = fib_levels.get('0.5', 0)
                
                near_fib = abs(current_price - fib_618) / current_price < 0.005 or \
                          abs(current_price - fib_5) / current_price < 0.005
                
                confidence = 60
                if near_fib:
                    confidence += 25
                if trend == 'STRONG_DOWN':
                    confidence += 15
                
                sl_distance = ob_zone_top - current_price
                tp_distance = sl_distance * 2  # 1:2 RR
                
                signal = {
                    'action': 'SELL',
                    'confidence': min(confidence, 95),
                    'reason': f"Bearish OB + {'FIB confluence' if near_fib else 'Trend'} + {trend}",
                    'entry': round(current_price, 5),
                    'sl': round(ob_zone_top + sl_distance * 0.1, 5),
                    'tp': round(current_price - tp_distance, 5),
                    'rr': '1:2',
                    'ob_zone': [ob_zone_bottom, ob_zone_top]
                }
                break
    
    # Fallback: Strong trend with good structure (lower confidence)
    if signal['action'] == 'WAIT':
        if trend == 'STRONG_UP' and bullish_obs and last_swing_low:
            # Strong uptrend - give BUY signal with lower confidence
            sl_distance = current_price - last_swing_low
            tp_distance = sl_distance * 1.5  # 1:1.5 RR for trend trades
            
            signal = {
                'action': 'BUY',
                'confidence': 55,  # Lower confidence for trend-based entry
                'reason': f"Strong uptrend + Bullish structure",
                'entry': round(current_price, 5),
                'sl': round(last_swing_low - sl_distance * 0.05, 5),
                'tp': round(current_price + tp_distance, 5),
                'rr': '1:1.5'
            }
        elif trend == 'STRONG_DOWN' and bearish_obs and last_swing_high:
            # Strong downtrend - give SELL signal with lower confidence
            sl_distance = last_swing_high - current_price
            tp_distance = sl_distance * 1.5  # 1:1.5 RR for trend trades
            
            signal = {
                'action': 'SELL',
                'confidence': 55,  # Lower confidence for trend-based entry
                'reason': f"Strong downtrend + Bearish structure",
                'entry': round(current_price, 5),
                'sl': round(last_swing_high + sl_distance * 0.05, 5),
                'tp': round(current_price - tp_distance, 5),
                'rr': '1:1.5'
            }
        else:
            signal['reason'] = 'No valid SMC setup found - waiting for price to reach OB zone'
    
    return signal


@app.route('/api/smc/<symbol>', methods=['GET'])
def get_smc_analysis(symbol):
    """Get Smart Money Concepts analysis: Swings, Order Blocks, Fibonacci, and Trade Signal"""
    if not mt5.symbol_select(symbol, True):
        return jsonify({"error": f"Symbol {symbol} not found"}), 404
    
    timeframe = request.args.get('timeframe', 'H1')
    tf_map = {
        'M1': mt5.TIMEFRAME_M1, 'M5': mt5.TIMEFRAME_M5, 'M15': mt5.TIMEFRAME_M15,
        'M30': mt5.TIMEFRAME_M30, 'H1': mt5.TIMEFRAME_H1, 'H4': mt5.TIMEFRAME_H4,
        'D1': mt5.TIMEFRAME_D1, 'W1': mt5.TIMEFRAME_W1
    }
    tf = tf_map.get(timeframe.upper(), mt5.TIMEFRAME_H1)
    
    # Get more candles for swing detection
    rates = mt5.copy_rates_from_pos(symbol, tf, 0, 200)
    if rates is None or len(rates) < 50:
        return jsonify({"error": "Not enough data"}), 400
    
    # Convert to list of dicts
    rates_list = [dict(zip(['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume'], r)) for r in rates]
    
    # Get current price
    tick = mt5.symbol_info_tick(symbol)
    current_price = (tick.bid + tick.ask) / 2 if tick else rates_list[-1]['close']
    
    # Find swing points
    swing_highs, swing_lows = find_swing_points(rates_list, lookback=3)
    
    # Find order blocks
    order_blocks = find_order_blocks(rates_list, swing_highs, swing_lows)
    
    # Calculate Fibonacci levels from last major swing
    fib_levels = {}
    fib_direction = 'bullish'
    if swing_highs and swing_lows:
        last_high = swing_highs[-1]
        last_low = swing_lows[-1]
        
        # Determine direction based on which came last
        if last_high['index'] > last_low['index']:
            # Swing high came after swing low - bullish move, expect retracement down
            fib_direction = 'bearish'
            fib_levels = calculate_fibonacci_levels(last_high['price'], last_low['price'], 'bearish')
        else:
            # Swing low came after swing high - bearish move, expect retracement up
            fib_direction = 'bullish'
            fib_levels = calculate_fibonacci_levels(last_high['price'], last_low['price'], 'bullish')
    
    # Get trend for signal generation
    trend_data = {}
    for tf_name, tf_val in [('H1', mt5.TIMEFRAME_H1), ('H4', mt5.TIMEFRAME_H4), ('D1', mt5.TIMEFRAME_D1)]:
        tf_rates = mt5.copy_rates_from_pos(symbol, tf_val, 0, 20)
        if tf_rates is not None:
            tf_list = [dict(zip(['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume'], r)) for r in tf_rates]
            trend_data[tf_name] = calculate_trend(tf_list)
    
    overall_trend = determine_overall_trend(trend_data) if trend_data else 'NEUTRAL'
    
    # Generate trading signal
    signal = generate_smc_signal(current_price, swing_highs, swing_lows, order_blocks, fib_levels, overall_trend)
    
    return jsonify({
        "symbol": symbol,
        "timeframe": timeframe,
        "current_price": round(current_price, 5),
        "swing_highs": swing_highs[-10:],  # Last 10 swing highs
        "swing_lows": swing_lows[-10:],    # Last 10 swing lows
        "order_blocks": order_blocks,
        "fibonacci": {
            "direction": fib_direction,
            "levels": fib_levels
        },
        "trend": overall_trend,
        "signal": signal
    })


@app.route('/api/smc/v2/<symbol>', methods=['GET'])
def get_smc_analysis_v2(symbol):
    """
    Enhanced SMC analysis v2 using official documented SMC methodology:
    
    - Market Structure (HH/HL/LH/LL with proper labeling)
    - Break of Structure (BOS) - continuation signals
    - Change of Character (CHoCH) - reversal signals  
    - Order Blocks (with mitigation status and BOS causation)
    - Fair Value Gaps (FVG) with fill percentage
    - Liquidity Zones (Equal Highs/Lows)
    - Liquidity Sweeps detection
    - Premium/Discount Zones with OTE levels
    - Fibonacci retracement for Optimal Trade Entry
    """
    if not SMC_MODULE_LOADED:
        return jsonify({"error": "SMC module not available - please ensure smc_analysis.py is in the same directory"}), 500
    
    if not mt5.symbol_select(symbol, True):
        return jsonify({"error": f"Symbol {symbol} not found"}), 404
    
    timeframe = request.args.get('timeframe', 'H1')
    tf_map = {
        'M1': mt5.TIMEFRAME_M1, 'M5': mt5.TIMEFRAME_M5, 'M15': mt5.TIMEFRAME_M15,
        'M30': mt5.TIMEFRAME_M30, 'H1': mt5.TIMEFRAME_H1, 'H4': mt5.TIMEFRAME_H4,
        'D1': mt5.TIMEFRAME_D1, 'W1': mt5.TIMEFRAME_W1
    }
    tf = tf_map.get(timeframe.upper(), mt5.TIMEFRAME_H1)
    
    # Get candles for analysis
    rates = mt5.copy_rates_from_pos(symbol, tf, 0, 200)
    if rates is None or len(rates) < 50:
        return jsonify({"error": "Not enough data"}), 400
    
    # Extract OHLC data
    opens = [float(r['open']) for r in rates]
    highs = [float(r['high']) for r in rates]
    lows = [float(r['low']) for r in rates]
    closes = [float(r['close']) for r in rates]
    times = [datetime.fromtimestamp(r['time']).isoformat() for r in rates]
    
    # Get current price and symbol info
    tick = mt5.symbol_info_tick(symbol)
    current_price = (tick.bid + tick.ask) / 2 if tick else closes[-1]
    symbol_info = mt5.symbol_info(symbol)
    decimals = symbol_info.digits if symbol_info else 5
    
    # Run official SMC analysis
    results = analyze_smc(opens, highs, lows, closes, times)
    
    # Add metadata
    results["symbol"] = symbol
    results["timeframe"] = timeframe
    results["current_price"] = round(current_price, decimals)
    results["analysis_version"] = "2.0.0"
    results["methodology"] = "Official SMC"
    
    return jsonify(results)


@app.route('/api/smc/reverse/<symbol>', methods=['GET'])
def get_reverse_smc_analysis(symbol):
    """
    Reverse SMC Analysis - Contrarian Trading Strategy
    
    For traders who believe SMC is a trap used by institutions to hunt
    retail traders' stop losses. This endpoint:
    
    1. Performs standard SMC analysis
    2. REVERSES the signal direction (BUY → SELL, SELL → BUY)
    3. SWAPS SL and TP (SMC's TP becomes your SL, SMC's SL becomes your TP)
    
    The theory: Institutions teach retail traders SMC concepts, then
    deliberately hunt their predictable stop loss placements at Order Blocks,
    FVG zones, and swing points before moving in the opposite direction.
    
    Use same parameters as /api/smc/v2/<symbol>:
    - timeframe: M1, M5, M15, M30, H1, H4, D1, W1 (default: H1)
    """
    if not SMC_MODULE_LOADED:
        return jsonify({"error": "SMC module not available - please ensure smc_analysis.py is in the same directory"}), 500
    
    if not mt5.symbol_select(symbol, True):
        return jsonify({"error": f"Symbol {symbol} not found"}), 404
    
    timeframe = request.args.get('timeframe', 'H1')
    tf_map = {
        'M1': mt5.TIMEFRAME_M1, 'M5': mt5.TIMEFRAME_M5, 'M15': mt5.TIMEFRAME_M15,
        'M30': mt5.TIMEFRAME_M30, 'H1': mt5.TIMEFRAME_H1, 'H4': mt5.TIMEFRAME_H4,
        'D1': mt5.TIMEFRAME_D1, 'W1': mt5.TIMEFRAME_W1
    }
    tf = tf_map.get(timeframe.upper(), mt5.TIMEFRAME_H1)
    
    # Get candles for analysis
    rates = mt5.copy_rates_from_pos(symbol, tf, 0, 200)
    if rates is None or len(rates) < 50:
        return jsonify({"error": "Not enough data"}), 400
    
    # Extract OHLC data
    opens = [float(r['open']) for r in rates]
    highs = [float(r['high']) for r in rates]
    lows = [float(r['low']) for r in rates]
    closes = [float(r['close']) for r in rates]
    times = [datetime.fromtimestamp(r['time']).isoformat() for r in rates]
    
    # Get current price and symbol info
    tick = mt5.symbol_info_tick(symbol)
    current_price = (tick.bid + tick.ask) / 2 if tick else closes[-1]
    symbol_info = mt5.symbol_info(symbol)
    decimals = symbol_info.digits if symbol_info else 5
    
    # Run REVERSE SMC analysis (reverse=True)
    results = analyze_smc(opens, highs, lows, closes, times, reverse=True)
    
    # Add metadata
    results["symbol"] = symbol
    results["timeframe"] = timeframe
    results["current_price"] = round(current_price, decimals)
    results["analysis_version"] = "2.0.0"
    results["methodology"] = "Reverse SMC (Contrarian)"
    results["description"] = "Trades AGAINST standard SMC signals - SL and TP are swapped"
    
    return jsonify(results)


@app.route('/api/symbols/tradable', methods=['GET'])
def get_tradable_symbols():
    """Get list of commonly traded symbols with search support"""
    search = request.args.get('search', '').upper()
    limit = int(request.args.get('limit', 100))
    
    # Common trading symbols to prioritize
    priority_base = ['XAUUSD', 'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 
                     'USDCHF', 'NZDUSD', 'GBPJPY', 'EURJPY', 'EURGBP', 'AUDJPY',
                     'XAGUSD', 'BTCUSD', 'ETHUSD', 'US30', 'US100', 'GER40', 'UK100']
    
    symbols = mt5.symbols_get()
    if not symbols:
        return jsonify({"symbols": [], "error": "Failed to get symbols"})
    
    result = []
    seen = set()
    
    # Helper to add symbol with info
    def add_symbol(s):
        if s.name in seen or not s.visible:
            return
        
        # Skip if not matching search
        if search and search not in s.name.upper() and search not in (s.description or '').upper():
            return
        
        seen.add(s.name)
        result.append({
            "symbol": s.name,
            "description": s.description,
            "bid": s.bid,
            "ask": s.ask,
            "digits": s.digits,
            "category": get_symbol_category(s.name)
        })
    
    # First add priority symbols
    symbol_dict = {s.name: s for s in symbols}
    for base in priority_base:
        # Try exact match
        if base in symbol_dict:
            add_symbol(symbol_dict[base])
        # Try with common suffixes
        for suffix in ['.pr', '.m', '.r', '.pro', '.raw', '.std']:
            key = base + suffix
            if key in symbol_dict:
                add_symbol(symbol_dict[key])
    
    # Then add remaining symbols
    for s in symbols:
        if len(result) >= limit:
            break
        add_symbol(s)
    
    return jsonify({
        "symbols": result[:limit],
        "count": len(result),
        "total_available": len(symbols)
    })

def get_symbol_category(symbol):
    """Categorize symbol for grouping"""
    symbol = symbol.upper()
    if 'XAU' in symbol or 'XAG' in symbol or 'GOLD' in symbol or 'SILVER' in symbol:
        return 'Metals'
    elif 'BTC' in symbol or 'ETH' in symbol or 'LTC' in symbol or 'CRYPTO' in symbol:
        return 'Crypto'
    elif any(idx in symbol for idx in ['US30', 'US100', 'US500', 'GER', 'UK100', 'JP225', 'DAX', 'NASDAQ', 'DOW']):
        return 'Indices'
    elif 'OIL' in symbol or 'WTI' in symbol or 'BRENT' in symbol or 'GAS' in symbol:
        return 'Energy'
    else:
        return 'Forex'

# ============== Profit Watchdog ==============
import threading
import time as time_module

watchdog_config = {
    "enabled": False,
    "mode": "FIXED",      # FIXED or AUTO
    "target_amount": 20,  # Target profit in account currency
    "step": 1,            # Step for AUTO mode
    "current_step": 0,    # Current step level for AUTO mode
    "check_interval": 1,  # Seconds between checks
    "magic": 0,           # 0 = all positions
    "last_check": None,
    "last_profit": 0,
    "positions_closed": 0
}

watchdog_thread = None
watchdog_running = False

def watchdog_worker():
    """Background worker that monitors floating profit and closes all trades when target is hit"""
    global watchdog_running
    print("[WATCHDOG] Worker started")
    
    while watchdog_running and watchdog_config["enabled"]:
        try:
            # Get all positions
            if watchdog_config["magic"] > 0:
                positions = mt5.positions_get(magic=watchdog_config["magic"])
            else:
                positions = mt5.positions_get()
            
            if positions is None or len(positions) == 0:
                watchdog_config["last_profit"] = 0
                time_module.sleep(watchdog_config["check_interval"])
                continue
            
            # Calculate total floating profit
            total_profit = sum(p.profit for p in positions)
            watchdog_config["last_profit"] = total_profit
            watchdog_config["last_check"] = datetime.now().isoformat()
            
            # Determine target based on mode
            if watchdog_config["mode"] == "AUTO":
                # In AUTO mode, target increases by step each time we hit it
                target = watchdog_config["step"] * (watchdog_config["current_step"] + 1)
            else:
                target = watchdog_config["target_amount"]
            
            # Check if we hit the target
            if total_profit >= target:
                print(f"[WATCHDOG] 🎯 Target hit! Profit: ${total_profit:.2f} >= Target: ${target:.2f}")
                print(f"[WATCHDOG] Closing {len(positions)} positions...")
                
                closed = 0
                for pos in positions:
                    # Close each position
                    tick = mt5.symbol_info_tick(pos.symbol)
                    if not tick:
                        continue
                    
                    price = tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask
                    close_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
                    
                    req = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "symbol": pos.symbol,
                        "volume": pos.volume,
                        "type": close_type,
                        "position": pos.ticket,
                        "price": price,
                        "deviation": 20,
                        "magic": pos.magic,
                        "comment": f"Watchdog TP ${target:.2f}",
                        "type_time": mt5.ORDER_TIME_GTC,
                        "type_filling": mt5.ORDER_FILLING_IOC
                    }
                    
                    result = mt5.order_send(req)
                    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                        closed += 1
                        print(f"[WATCHDOG] ✅ Closed {pos.symbol} #{pos.ticket}")
                    else:
                        print(f"[WATCHDOG] ❌ Failed to close {pos.symbol} #{pos.ticket}")
                
                watchdog_config["positions_closed"] += closed
                print(f"[WATCHDOG] Closed {closed}/{len(positions)} positions")
                
                # In AUTO mode, increment the step
                if watchdog_config["mode"] == "AUTO":
                    watchdog_config["current_step"] += 1
                    print(f"[WATCHDOG] AUTO mode: Next target = ${watchdog_config['step'] * (watchdog_config['current_step'] + 1):.2f}")
                else:
                    # In FIXED mode, disable after hitting target
                    watchdog_config["enabled"] = False
                    print("[WATCHDOG] FIXED mode: Watchdog disabled after target hit")
                    break
            
        except Exception as e:
            print(f"[WATCHDOG] Error: {e}")
        
        time_module.sleep(watchdog_config["check_interval"])
    
    print("[WATCHDOG] Worker stopped")
    watchdog_running = False

@app.route('/api/watchdog/start', methods=['POST'])
def start_watchdog():
    """Start the profit watchdog"""
    global watchdog_thread, watchdog_running
    
    data = request.json or {}
    mode = data.get('mode', 'FIXED').upper()
    
    if mode == 'AUTO':
        watchdog_config["mode"] = "AUTO"
        watchdog_config["step"] = float(data.get('step', 1))
        watchdog_config["current_step"] = 0
    else:
        watchdog_config["mode"] = "FIXED"
        watchdog_config["target_amount"] = float(data.get('amount', 20))
    
    watchdog_config["magic"] = int(data.get('magic', 0))
    watchdog_config["check_interval"] = float(data.get('interval', 1))
    watchdog_config["enabled"] = True
    watchdog_config["positions_closed"] = 0
    
    # Start worker thread if not running
    if not watchdog_running:
        watchdog_running = True
        watchdog_thread = threading.Thread(target=watchdog_worker, daemon=True)
        watchdog_thread.start()
    
    target = watchdog_config["step"] if mode == "AUTO" else watchdog_config["target_amount"]
    
    return jsonify({
        "ok": True,
        "status": "running",
        "mode": watchdog_config["mode"],
        "target": target,
        "message": f"Watchdog started in {mode} mode, target: ${target:.2f}"
    })

@app.route('/api/watchdog/stop', methods=['POST'])
def stop_watchdog():
    """Stop the profit watchdog"""
    global watchdog_running
    
    watchdog_config["enabled"] = False
    watchdog_running = False
    
    return jsonify({
        "ok": True,
        "status": "stopped",
        "positions_closed": watchdog_config["positions_closed"],
        "message": "Watchdog stopped"
    })

@app.route('/api/watchdog/status', methods=['GET'])
def watchdog_status():
    """Get watchdog status"""
    # Get current floating profit
    positions = mt5.positions_get()
    current_profit = sum(p.profit for p in positions) if positions else 0
    
    if watchdog_config["mode"] == "AUTO":
        next_target = watchdog_config["step"] * (watchdog_config["current_step"] + 1)
    else:
        next_target = watchdog_config["target_amount"]
    
    return jsonify({
        "enabled": watchdog_config["enabled"],
        "mode": watchdog_config["mode"],
        "target_amount": watchdog_config["target_amount"],
        "step": watchdog_config["step"],
        "current_step": watchdog_config["current_step"],
        "next_target": next_target,
        "floating_profit": round(current_profit, 2),
        "last_check": watchdog_config["last_check"],
        "positions_closed": watchdog_config["positions_closed"],
        "open_positions": len(positions) if positions else 0
    })

# Legacy endpoints for GUI
@app.route('/api/start_watchdog', methods=['POST'])
def legacy_start_watchdog():
    """Legacy endpoint for GUI"""
    return start_watchdog()

@app.route('/api/stop_watchdog', methods=['POST'])
def legacy_stop_watchdog():
    """Legacy endpoint for GUI"""
    return stop_watchdog()

@app.route('/api/watchdog_stats', methods=['GET'])
def legacy_watchdog_stats():
    """Legacy endpoint for GUI - returns stats in old format"""
    status = watchdog_status().get_json()
    return jsonify({
        "cpu": "N/A",
        "memory": "N/A",
        "floating_profit": f"${status['floating_profit']:.2f}",
        "status": "enabled" if status["enabled"] else "disabled",
        "target": status["next_target"],
        "mode": status["mode"]
    })


# ============== Legacy Endpoints for GUI Compatibility ==============
@app.route('/balance', methods=['GET'])
def legacy_balance():
    """Legacy endpoint for old GUI - redirects to /api/account"""
    return account_info()

@app.route('/positions', methods=['GET'])
def legacy_positions():
    """Legacy endpoint for old GUI - redirects to /api/positions"""
    return get_positions()

@app.route('/history', methods=['GET'])
def legacy_history():
    """Legacy endpoint for old GUI - adapts date range to days"""
    start = request.args.get('start')
    end = request.args.get('end')
    
    # Convert date range to days
    days = 30  # default
    if start and end:
        try:
            start_dt = datetime.strptime(start, '%Y-%m-%d')
            end_dt = datetime.strptime(end, '%Y-%m-%d')
            days = max(1, (datetime.now() - start_dt).days + 1)
        except:
            pass
    
    symbol = request.args.get('symbol')
    from_dt = datetime.now() - timedelta(days=days)
    to_dt = datetime.now()
    
    if symbol:
        deals = mt5.history_deals_get(from_dt, to_dt, group=f"*{symbol}*")
    else:
        deals = mt5.history_deals_get(from_dt, to_dt)
    
    if deals is None:
        return jsonify({"trades": [], "error": err()})
    
    # Format for GUI compatibility
    trades = []
    for d in deals:
        if d.entry == 1:  # Only OUT deals (closed trades)
            trades.append({
                "ticket": d.ticket,
                "symbol": d.symbol,
                "type": "buy" if d.type == 0 else "sell",
                "volume": d.volume,
                "price": d.price,
                "profit": d.profit,
                "time": datetime.fromtimestamp(d.time).isoformat(),
                "comment": d.comment
            })
    
    return jsonify({"trades": trades, "count": len(trades)})

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "name": "MT5 REST API Server",
        "version": "2.1.0",
        "endpoints": {
            "Connection": [
                "POST /api/init", "POST /api/shutdown", "GET /api/status", "GET /api/version",
                "POST /api/invoke/trade", "POST /api/invoke/close", "POST /api/invoke/account",
            ],
            "Account": ["GET /api/account"],
            "Symbols": ["GET /api/symbols", "GET /api/symbols/tradable?search=&limit=100", "GET /api/symbol/<symbol>", "POST /api/symbol/<symbol>/select"],
            "Trend": ["GET /api/trend/<symbol>"],
            "Price": ["GET /api/price/<symbol>", "GET /api/tick/<symbol>"],
            "Candles": ["GET /api/candles/<symbol>?timeframe=H1&count=100"],
            "Indicators": [
                "GET /api/indicator/rsi/<symbol>?period=14&timeframe=H1&count=100",
                "GET /api/indicator/ma/<symbol>?type=sma|ema|smma|lwma&period=14&applied=close",
                "GET /api/indicator/macd/<symbol>?fast=12&slow=26&signal=9",
                "GET /api/indicator/bollinger/<symbol>?period=20&std_dev=2",
                "GET /api/indicator/stochastic/<symbol>?k_period=14&d_period=3",
                "GET /api/indicator/atr/<symbol>?period=14",
                "GET /api/indicator/cci/<symbol>?period=20",
                "GET /api/indicator/williams/<symbol>?period=14",
                "GET /api/indicator/momentum/<symbol>?period=10"
            ],
            "Trading": [
                "POST /api/trade/open {symbol,type,volume,sl,tp,sl_points,tp_points,price,magic,comment}",
                "POST /api/trade/close {ticket,volume}",
                "POST /api/trade/close_all {symbol,magic}",
                "POST /api/trade/modify {ticket,sl,tp,sl_points,tp_points}",
                "POST /api/order/cancel {ticket}"
            ],
            "Trailing Stop": [
                "POST /api/trailing/set {points, magic}",
                "POST /api/trailing/disable",
                "GET /api/trailing/status",
                "POST /api/trailing/apply"
            ],
            "Positions": ["GET /api/positions?symbol=&ticket=&magic=", "GET /api/orders"],
            "History": ["GET /api/history/deals?days=30&symbol=", "GET /api/history/orders?days=30"],
            "Utilities": ["POST /api/calc/margin", "POST /api/calc/profit", "GET /api/market/book/<symbol>"],
            "Legacy (GUI)": ["GET /balance", "GET /positions", "GET /history", "POST /set_trailing_sl", "POST /disable_trailing_sl", "GET /trailing_status"]
        }
    })

def auto_init_mt5():
    """Auto-initialize MT5 connection on startup"""
    import time
    time.sleep(2)  # Wait for Flask to start
    
    for attempt in range(5):
        try:
            if mt5.initialize():
                account = mt5.account_info()
                if account:
                    print(f"✅ MT5 Auto-connected: Account {account.login}")
                    return True
                else:
                    print(f"⏳ Attempt {attempt+1}: MT5 initialized but no account")
            else:
                error = mt5.last_error()
                print(f"⏳ Attempt {attempt+1}: {error}")
        except Exception as e:
            print(f"⏳ Attempt {attempt+1}: {e}")
        time.sleep(3)
    
    print("⚠️ MT5 auto-init failed - will connect when /api/init is called")
    return False

if __name__ == '__main__':
    print("=" * 60)
    print("MT5 REST API Server v2.0")
    print("=" * 60)
    print("Server: http://0.0.0.0:5000")
    print("Docs:   http://localhost:5000/")
    print("=" * 60)
    
    # Start auto-init in background thread
    import threading
    init_thread = threading.Thread(target=auto_init_mt5, daemon=True)
    init_thread.start()
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
