import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Tuple
import ccxt
import pandas as pd
import numpy as np
from flask import Flask  # for 24/7 hack on Render
import requests  # for Telegram pings
import csv  # for logging

app = Flask(__name__)  # web server to keep awake

# CONFIG - Customize here
CONFIG = {
    "SYMBOLS": [
        "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT",
        "ADA/USDT", "AVAX/USDT", "DOT/USDT", "LINK/USDT", "BCH/USDT"
    ],
    "TIMEFRAMES": {"5m": 400, "15m": 400, "1h": 300, "4h": 200},
    "SCAN_INTERVAL": 15,  # seconds
    "LOCKED_INTERVAL_MIN": 120,  # minutes lock after signal
    "MIN_SCORE": 70,  # 0-100 for god-tier only
    "MIN_RR": 2.5,  # minimum risk-reward
    "RISK_MODEL": {
        "SL_ATR_MULT": 1.2,   # default
        "TP_ATR_MULT": 3.0,   # default
    },
    "MAX_SIGNALS_PER_DAY": 5,
    "TRADE_SESSIONS_UTC": [
        {"start": "00:00", "end": "07:00"},  # Asia (tuned for low-vol)
        {"start": "07:00", "end": "11:00"},  # London
        {"start": "13:00", "end": "17:00"},  # NY overlap
    ],
    "VOL_REGIME_MIN": 0.0015,  # normalized ATR min
    "REGIME_THRESHOLD": 0.003,  # ATR/price for trend vs range
    "RECENT_MOVE_PCT": 0.01,  # 1% recent move skip threshold
    "RECENT_MOVE_CANDLES": 3,  # last 3 candles for recent move check
    "ORDER_BUFFER_SEC": 90,  # seconds buffer for manual entry
    "TELEGRAM_TOKEN": "8441346951:AAGRjh5GQaResRakjmdre3iVPvXYdoqEP5g",  # your bot token
    "TELEGRAM_CHAT_ID": "8557187571",  # your chat ID
    "PRICE_DECIMALS": {
        "BTC/USDT": 1, "ETH/USDT": 2, "BNB/USDT": 2, "SOL/USDT": 3,
        "XRP/USDT": 4, "ADA/USDT": 4, "AVAX/USDT": 2, "DOT/USDT": 3,
        "LINK/USDT": 3, "BCH/USDT": 2
    },
    "LOG_FILE": "signals_log.csv"
}

exchange = ccxt.binance({'enableRateLimit': True})
exchange.load_markets()

STATE = {
    "signals_today": 0,
    "last_reset": datetime.now(timezone.utc).date()
}

@dataclass
class Signal:
    symbol: str
    direction: str
    entry: float
    sl: float
    tp: float
    rr: float
    score: int
    reason: str

def load_ohlcv(symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    try:
        raw = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(raw, columns=['ts','open','high','low','close','volume'])
        df['ts'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
        df = df.iloc[:-1].reset_index(drop=True)  # drop incomplete candle
        if len(df) < 100:
            return pd.DataFrame()
        return df
    except Exception as e:
        print(f"[load_ohlcv] Error for {symbol} {timeframe}: {e}")
        return pd.DataFrame()

def ema(s, n): return s.ewm(span=n, adjust=False).mean()

def atr(df, n=14):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def rsi(df, n=14):
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(n).mean()
    loss = -delta.where(delta < 0, 0).rolling(n).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def in_trade_session(now_utc: datetime) -> bool:
    current_time = now_utc.time()
    for sess in CONFIG["TRADE_SESSIONS_UTC"]:
        start = datetime.strptime(sess["start"], "%H:%M").time()
        end = datetime.strptime(sess["end"], "%H:%M").time()
        if start <= current_time <= end:
            return True
    return False

def determine_regime(df_15m: pd.DataFrame, df_1h: pd.DataFrame, df_4h: pd.DataFrame, atr15: float, rsi15: float) -> str:
    # Trend mode: EMA stack, structure breaks, VWAP holding
    ema9_1h = ema(df_1h['close'], 9).iloc[-1]
    ema21_1h = ema(df_1h['close'], 21).iloc[-1]
    ema9_4h = ema(df_4h['close'], 9).iloc[-1]
    ema21_4h = ema(df_4h['close'], 21).iloc[-1]

    ema_stack_bull = ema9_1h > ema21_1h and ema9_4h > ema21_4h
    ema_stack_bear = ema9_1h < ema21_1h and ema9_4h < ema21_4h

    # Structure: HH/HL or LH/LL
    hh = df_15m['high'].rolling(20).max().shift(1).iloc[-1]
    hl = df_15m['low'].rolling(20).max().shift(1).iloc[-1]
    lh = df_15m['high'].rolling(20).min().shift(1).iloc[-1]
    ll = df_15m['low'].rolling(20).min().shift(1).iloc[-1]
    structure_bull = df_15m['high'].iloc[-1] > hh and df_15m['low'].iloc[-1] > hl
    structure_bear = df_15m['high'].iloc[-1] < lh and df_15m['low'].iloc[-1] < ll

    # VWAP holding (simple MA as proxy)
    vwap = (df_15m['high'] + df_15m['low'] + df_15m['close']) / 3
    vwap_bull = df_15m['close'].iloc[-1] > vwap.rolling(20).mean().iloc[-1]
    vwap_bear = df_15m['close'].iloc[-1] < vwap.rolling(20).mean().iloc[-1]

    trend_points = 0
    if ema_stack_bull or ema_stack_bear: trend_points += 1
    if structure_bull or structure_bear: trend_points += 1
    if vwap_bull or vwap_bear: trend_points += 1

    range_points = 0
    if not ema_stack_bull and not ema_stack_bear: range_points += 1
    if not structure_bull and not structure_bear: range_points += 1
    if 40 < rsi15 < 60: range_points += 1

    if atr15 / df_15m['close'].iloc[-1] > CONFIG["REGIME_THRESHOLD"]:
        return "TREND" if trend_points >= 2 else "RANGE"
    else:
        return "RANGE" if range_points >= 2 else "TREND"

def calculate_score(symbol, dfs) -> Tuple[Optional[Signal], Optional[Dict]]:
    try:
        d5 = dfs.get('5m')
        d15 = dfs.get('15m')
        d1h = dfs.get('1h')
        d4h = dfs.get('4h')

        if any(d.empty for d in [d15, d1h, d4h]) or any(len(d) < 50 for d in [d15, d1h, d4h]):
            return None, None

        price = d15['close'].iloc[-1]
        atr15 = atr(d15).iloc[-1]

        vol_regime = atr15 / price  # normalized ATR
        if vol_regime < CONFIG["VOL_REGIME_MIN"]:
            return None, None

        rsi15 = rsi(d15).iloc[-1]

        regime = determine_regime(d15, d1h, d4h, atr15, rsi15)

        score_breakdown = {"htf": 0, "ob": 0, "fvg": 0, "sweep": 0, "vol": 0, "rsi": 0}

        # HTF Bias
        ema9_1h = ema(d1h['close'], 9).iloc[-1]
        ema21_1h = ema(d1h['close'], 21).iloc[-1]
        ema9_4h = ema(d4h['close'], 9).iloc[-1]
        ema21_4h = ema(d4h['close'], 21).iloc[-1]

        if ema9_1h > ema21_1h and ema9_4h > ema21_4h:
            bias = "BULL"
            score_breakdown["htf"] = 30
        elif ema9_1h < ema21_1h and ema9_4h < ema21_4h:
            bias = "BEAR"
            score_breakdown["htf"] = 30
        else:
            bias = "NEUTRAL"

        if bias == "NEUTRAL":
            return None, None

        if (bias == "BULL" and price < ema9_1h) or (bias == "BEAR" and price > ema9_1h):
            return None, None

        # OB
        ob_score = 0
        bull_ob, bear_ob = find_order_blocks(d15)
        if bull_ob and abs(price - bull_ob) / atr15 < 0.6:
            ob_score += 15
        if bear_ob and abs(price - bear_ob) / atr15 < 0.6:
            ob_score += 15
        score_breakdown["ob"] = min(ob_score, 25)

        # FVG
        fvg_up, fvg_dn = find_fvg(d15)
        if fvg_up and price <= fvg_up:
            score_breakdown["fvg"] = 20
        if fvg_dn and price >= fvg_dn:
            score_breakdown["fvg"] = 20

        # Sweep - aligned with bias, using 5m if available
        sweep_score = 0
        if d5 is not None and not d5.empty and len(d5) >= 50:
            if bias == "BULL" and sweep_liquidity(d5, "LONG"):
                sweep_score = 10
            elif bias == "BEAR" and sweep_liquidity(d5, "SHORT"):
                sweep_score = 10
        score_breakdown["sweep"] = sweep_score

        # Vol
        if d15['volume'].iloc[-1] > d15['volume'].rolling(20).mean().iloc[-1] * 2:
            score_breakdown["vol"] = 10

        # RSI weight
        rsi_score = 0
        if 40 <= rsi15 <= 60:
            rsi_score += 5
        elif (bias == "BULL" and rsi15 < 35) or (bias == "BEAR" and rsi15 > 65):
            rsi_score += 5
        score_breakdown["rsi"] = rsi_score

        score = sum(score_breakdown.values())
        if score < CONFIG["MIN_SCORE"]:
            return None, None

        # Regime-specific adjustments
        sl_mult = CONFIG["RISK_MODEL"]["SL_ATR_MULT"]
        tp_mult = CONFIG["RISK_MODEL"]["TP_ATR_MULT"]
        if regime == "TREND":
            sl_mult = 1.5  # wider in trend
        elif regime == "RANGE":
            sl_mult = 1.0  # tighter in range

        # Direction
        direction = "LONG" if bias == "BULL" else "SHORT"

        # 5m ATR for entry region, fallback to 15m
        d5_atr = atr(d5).iloc[-1] if d5 is not None and not d5.empty and len(d5) >= 50 else atr15

        if direction == "LONG":
            sl = price - sl_mult * d5_atr
            tp = price + tp_mult * d5_atr
        else:
            sl = price + sl_mult * d5_atr
            tp = price - tp_mult * d5_atr

        rr = abs(tp - price) / abs(sl - price)
        if rr < CONFIG["MIN_RR"]:
            return None, None

        sig = Signal(
            symbol=symbol,
            direction=direction,
            entry=price,
            sl=sl,
            tp=tp,
            rr=rr,
            score=score,
            reason=" | ".join([f"{k}:{v}" for k, v in score_breakdown.items() if v > 0])
        )
        return sig, score_breakdown

    except Exception as e:
        print(f"calculate_score error for {symbol}: {e}")
        return None, None

def send_tele(msg: str):
    try:
        url = f"https://api.telegram.org/bot{CONFIG['TELEGRAM_TOKEN']}/sendMessage"
        params = {"chat_id": CONFIG["TELEGRAM_CHAT_ID"], "text": msg}
        resp = requests.get(url, params=params, timeout=5)
        if resp.status_code != 200:
            print(f"[send_tele] Non-200: {resp.status_code} {resp.text}")
    except Exception as e:
        print(f"[send_tele] Error: {e}")

def main():
    send_tele("Bot is running smoothly!")  # Startup message
    print("Sniper Bot Running - Complete Engine")
    while True:
        try:
            now_utc = datetime.now(timezone.utc)
            today = now_utc.date()
            if today != STATE["last_reset"]:
                STATE["last_reset"] = today
                STATE["signals_today"] = 0

            if STATE["signals_today"] >= CONFIG["MAX_SIGNALS_PER_DAY"]:
                print("Max signals hit for today. Standing down.")
                time.sleep(CONFIG["SCAN_INTERVAL"] * 4)
                continue

            if not in_trade_session(now_utc):
                print(f"[{now_utc}] Outside trade session. Sleeping...")
                time.sleep(CONFIG["SCAN_INTERVAL"])
                continue

            best: Optional[Tuple[Signal, Dict]] = None

            for sym in CONFIG["SYMBOLS"]:
                dfs: Dict[str, pd.DataFrame] = {}
                for tf, lim in CONFIG["TIMEFRAMES"].items():
                    df = load_ohlcv(sym, tf, lim)
                    if df.empty:
                        dfs = {}
                        break
                    dfs[tf] = df

                if len(dfs) < 4:
                    continue  # all TFs required

                sig, breakdown = calculate_score(sym, dfs)
                if sig and breakdown:
                    if best is None or sig.score > best[0].score:
                        best = (sig, breakdown)

            if best:
                best_sig, best_breakdown = best
                msg = (
                    f"SNIPER CALL\n"
                    f"Coin: {best_sig.symbol}\n"
                    f"Direction: {best_sig.direction}\n"
                    f"Entry: {best_sig.entry:.{CONFIG['PRICE_DECIMALS'].get(best_sig.symbol,3)}f}\n"
                    f"SL: {best_sig.sl:.{CONFIG['PRICE_DECIMALS'].get(best_sig.symbol,3)}f}\n"
                    f"TP: {best_sig.tp:.{CONFIG['PRICE_DECIMALS'].get(best_sig.symbol,3)}f}\n"
                    f"RR: 1:{best_sig.rr}\n"
                    f"Score: {best_sig.score}/100\n"
                    f"Reason: {best_sig.reason}"
                )
                send_tele(msg)
                print(msg)
                log_signal(best_sig, best_breakdown)
                STATE["signals_today"] += 1
                time.sleep(CONFIG["LOCKED_INTERVAL_MIN"] * 60)  # lock
            else:
                print(f"[{now_utc}] Scanning - No perfect setup yet...")
                time.sleep(CONFIG["SCAN_INTERVAL"])

        except Exception as e:
            print(f"Error: {e}")
            time.sleep(60)

@app.route('/')
def home():
    return "Bot Awake"

if __name__ == "__main__":
    from threading import Thread
    Thread(target=app.run, kwargs={'host': '0.0.0.0', 'port': 8080}).start()
    main()
