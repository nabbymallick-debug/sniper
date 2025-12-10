import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict
import ccxt
import pandas as pd
import numpy as np
from flask import Flask  # for 24/7 hack on Render
import requests  # for Telegram pings

app = Flask(__name__)  # web server to keep awake

# CONFIG - Customize here
CONFIG = {
    "SYMBOLS": [
        "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT",
        "ADA/USDT", "AVAX/USDT", "DOT/USDT", "LINK/USDT", "BCH/USDT"
    ],
    "TIMEFRAMES": {"5m": 400, "15m": 400, "1h": 300, "4h": 200},
    "SCAN_INTERVAL": 15,  # seconds
    "LOCKED_INTERVAL": 8,  # seconds when trade active
    "MIN_SCORE": 90,  # 0-100 for god-tier only
    "MIN_RR": 3.0,  # minimum risk-reward
    "TELEGRAM_TOKEN": "8441346951:AAGRjh5GQaResRakjmdre3iVPvXYdoqEP5g",  # your bot token
    "TELEGRAM_CHAT_ID": "8557187571",  # your chat ID
    "PRICE_DECIMALS": {
        "BTC/USDT": 1, "ETH/USDT": 2, "BNB/USDT": 2, "SOL/USDT": 3,
        "XRP/USDT": 4, "ADA/USDT": 4, "AVAX/USDT": 2, "DOT/USDT": 3,
        "LINK/USDT": 3, "BCH/USDT": 2
    }
}

exchange = ccxt.binance({'enableRateLimit': True})
exchange.load_markets()

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

def find_order_blocks(df, lookback=30):
    df['body'] = abs(df['close'] - df['open'])
    df['bull_ob'] = np.where((df['close'] > df['open']) & 
                             (df['body'] > df['body'].shift(1)*1.8) &
                             (df['low'] < df['low'].shift(1)), df['low'], np.nan)
    df['bear_ob'] = np.where((df['close'] < df['open']) & 
                             (df['body'] > df['body'].shift(1)*1.8) &
                             (df['high'] > df['high'].shift(1)), df['high'], np.nan)
    bull = df['bull_ob'].dropna().iloc[-1] if not df['bull_ob'].dropna().empty else None
    bear = df['bear_ob'].dropna().iloc[-1] if not df['bear_ob'].dropna().empty else None
    return bull, bear

def find_fvg(df):
    df['fvg_up'] = np.where((df['low'] > df['high'].shift(2)), df['low'], np.nan)
    df['fvg_dn'] = np.where((df['high'] < df['low'].shift(2)), df['high'], np.nan)
    up = df['fvg_up'].dropna().iloc[-1] if not df['fvg_up'].dropna().empty else None
    dn = df['fvg_dn'].dropna().iloc[-1] if not df['fvg_dn'].dropna().empty else None
    return up, dn

def sweep_liquidity(df, direction):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    if direction == "LONG":
        return last['low'] < prev['low'] and last['close'] > prev['open']
    else:
        return last['high'] > prev['high'] and last['close'] < prev['open']

def is_trap(df, atr15):
    last = df.iloc[-1]
    wick = (last['high'] - last['close']) if last['close'] < last['open'] else (last['close'] - last['low'])
    if wick > atr15 * 1.5 and df['volume'].iloc[-1] < df['volume'].rolling(20).mean().iloc[-1]:
        return True
    return False

def calculate_score(symbol, dfs):
    try:
        d15 = dfs['15m']
        d1h = dfs['1h']
        d4h = dfs['4h']

        if len(d15) < 50: return None

        price = d15['close'].iloc[-1]
        atr15 = atr(d15).iloc[-1]

        if is_trap(d15, atr15): return None

        score = 0
        reason_parts = []

        ema9_1h = ema(d1h['close'], 9).iloc[-1]
        ema21_1h = ema(d1h['close'], 21).iloc[-1]
        ema9_4h = ema(d4h['close'], 9).iloc[-1]
        ema21_4h = ema(d4h['close'], 21).iloc[-1]
        if ema9_1h > ema21_1h and ema9_4h > ema21_4h and price > ema9_1h:
            score += 30; reason_parts.append("HTF Bull")
        elif ema9_1h < ema21_1h and ema9_4h < ema21_4h and price < ema9_1h:
            score += 30; reason_parts.append("HTF Bear")

        bull_ob, bear_ob = find_order_blocks(d15)
        if bull_ob and abs(price - bull_ob)/atr15 < 0.6:
            score += 25; reason_parts.append("Bull OB")
        if bear_ob and abs(price - bear_ob)/atr15 < 0.6:
            score += 25; reason_parts.append("Bear OB")

        fvg_up, fvg_dn = find_fvg(d15)
        if fvg_up and price <= fvg_up:
            score += 20; reason_parts.append("FVG Up")
        if fvg_dn and price >= fvg_dn:
            score += 20; reason_parts.append("FVG Dn")

        if sweep_liquidity(d15, "LONG"):
            score += 20; reason_parts.append("Bull Sweep")
        if sweep_liquidity(d15, "SHORT"):
            score += 20; reason_parts.append("Bear Sweep")

        if d15['volume'].iloc[-1] > d15['volume'].rolling(20).mean().iloc[-1] * 2:
            score += 10; reason_parts.append("Vol Spike")

        rsi15 = rsi(d15).iloc[-1]
        if rsi15 > 65 or rsi15 < 35: return None

        if score < CONFIG["MIN_SCORE"]: return None

        direction = "LONG" if "HTF Bull" in " ".join(reason_parts) else "SHORT"

        if d15['close'].iloc[-1] == price:  # closed candle
            next_tp = price + 3.5 * atr15 if direction == "LONG" else price - 3.5 * atr15
            sl = price - atr15 if direction == "LONG" else price + atr15
            rr = abs(next_tp - price) / abs(sl - price)
            if rr < CONFIG["MIN_RR"]: return None
            return Signal(symbol, direction, price, sl, next_tp, rr, score, " | ".join(reason_parts))
        return None
    except:
        return None

def send_tele(msg):
    try:
        url = f"https://api.telegram.org/bot{CONFIG['TELEGRAM_TOKEN']}/sendMessage?chat_id={CONFIG['TELEGRAM_CHAT_ID']}&text={msg}"
        requests.get(url)
    except:
        pass

def main():
    print("Sniper Bot Running - Complete Engine")
    while True:
        try:
            best = None
            for sym in CONFIG["SYMBOLS"]:
                dfs = {tf: pd.DataFrame(exchange.fetch_ohlcv(sym, tf, limit=CONFIG["TIMEFRAMES"][tf])) for tf in CONFIG["TIMEFRAMES"]}
                for tf in dfs:
                    if dfs[tf].empty: continue
                    dfs[tf].columns = ['ts','open','high','low','close','volume']
                    dfs[tf]['atr'] = atr(dfs[tf])
                sig = calculate_score(sym, dfs)
                if sig and (best is None or sig.score > best.score):
                    best = sig

                time.sleep(0.5)  # rate limit

            if best:
                msg = f"SNIPER CALL\nCoin: {best.symbol}\nDirection: {best.direction}\nEntry: {best.entry:.{CONFIG['PRICE_DECIMALS'].get(best.symbol,3)}f}\nSL: {best.sl:.{CONFIG['PRICE_DECIMALS'].get(best.symbol,3)}f}\nTP: {best.tp:.{CONFIG['PRICE_DECIMALS'].get(best.symbol,3)}f}\nRR: 1:{best.rr}\nScore: {best.score}/100\nReason: {best.reason}"
                send_tele(msg)
                print(msg)
                time.sleep(CONFIG["LOCKED_INTERVAL"] * 60)  # lock for 2 hours after signal
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Scanning - No perfect setup yet...")
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
