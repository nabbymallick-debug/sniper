import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict
import ccxt
import pandas as pd
import numpy as np
import winsound  # for beeps
from flask import Flask  # for 24/7 hack

app = Flask(__name__)  # web server for ping

CONFIG = {
    "SYMBOLS": [
        "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT",
        "ADA/USDT", "AVAX/USDT", "DOT/USDT", "LINK/USDT", "BCH/USDT"
    ],
    "TIMEFRAMES": {"5m": 400, "15m": 400, "1h": 300, "4h": 200},
    "SCAN_INTERVAL": 15,           # seconds when free
    "LOCKED_INTERVAL": 8,          # seconds when trade active
    "MIN_SCORE": 85,               # 0-100 → only god-tier setups
    "MIN_RR": 3.0,
    "RISK_PERCENT": 1.0,           # 1% of balance
    "ACCOUNT_BALANCE": 100.0,      # change to your real balance
    "TELEGRAM_TOKEN": "8441346951:AAGRjh5GQaResRakjmdre3iVPvXYdoqEP5g",  # your token placed
    "TELEGRAM_CHAT_ID": "8557187571",  # your chat ID placed
    "PRICE_DECIMALS": {            # clean copy-paste prices
        "BTC/USDT": 1, "ETH/USDT": 2, "BNB/USDT": 2, "SOL/USDT": 3,
        "XRP/USDT": 4, "ADA/USDT": 4, "AVAX/USDT": 2, "DOT/USDT": 3,
        "LINK/USDT": 3, "BCH/USDT": 2
    }
}

exchange = ccxt.binance({'enableRateLimit': True})
exchange.load_markets()

# Global state
active_trade = None
last_beep = 0

@dataclass
class Signal:
    symbol: str
    direction: str  # LONG or SHORT
    entry: float
    sl: float
    tp: float
    rr: float
    score: int
    reason: str

# ======================= INDICATORS =======================
def ema(s, n): return s.ewm(span=n, adjust=False).mean()
def atr(df, n=14):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(n).mean()

# ======================= SMC / ICT TOOLS =======================
def find_order_blocks(df_15m: pd.DataFrame, lookback=30):
    """Returns latest bullish & bearish OB prices"""
    df = df_15m.copy()
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

def find_fvg(df_15m: pd.DataFrame):
    """Simple FVG detection"""
    df = df_15m.copy()
    df['fvg_up'] = np.where((df['low'] > df['high'].shift(2)) , df['low'], np.nan)
    df['fvg_dn'] = np.where((df['high'] < df['low'].shift(2)), df['high'], np.nan)
    up = df['fvg_up'].dropna().iloc[-1] if not df['fvg_up'].dropna().empty else None
    dn = df['fvg_dn'].dropna().iloc[-1] if not df['fvg_dn'].dropna().empty else None
    return up, dn

def sweep_liquidity(df_15m: pd.DataFrame, direction: str):
    last = df_15m.iloc[-1]
    prev = df_15m.iloc[-2]
    if direction == "LONG":
        return last['low'] < prev['low'] and last['close'] > prev['open']
    else:
        return last['high'] > prev['high'] and last['close'] < prev['open']

# ======================= SCORING ENGINE (0-100) =======================
def calculate_score(symbol: str, dfs: Dict[str, pd.DataFrame]) -> Optional[Signal]:
    try:
        d5 = dfs['5m']
        d15 = dfs['15m']
        d1h = dfs['1h']
        d4h = dfs['4h']

        if len(d15) < 50: return None

        price = d15['close'].iloc[-1]
        atr15 = d15['atr'].iloc[-1]

        score = 0
        reason_parts = []

        # 1. HTF Trend (4h + 1h EMA stack)
        ema9_1h = ema(d1h['close'], 9).iloc[-1]
        ema21_1h = ema(d1h['close'], 21).iloc[-1]
        if ema9_1h > ema21_1h and price > ema9_1h:
            score += 20; reason_parts.append("HTF Bull")
        elif ema9_1h < ema21_1h and price < ema9_1h:
            score += 20; reason_parts.append("HTF Bear")

        # 2. Order Block touch
        bull_ob, bear_ob = find_order_blocks(d15)
        if bull_ob and abs(price - bull_ob)/atr15 < 0.6:
            score += 25; reason_parts.append("Bull OB")
        if bear_ob and abs(price - bear_ob)/atr15 < 0.6:
            score += 25; reason_parts.append("Bear OB")

        # 3. FVG fill
        fvg_up, fvg_dn = find_fvg(d15)
        if fvg_up and price <= fvg_up:
            score += 20; reason_parts.append("FVG Fill ↑")
        if fvg_dn and price >= fvg_dn:
            score += 20; reason_parts.append("FVG Fill ↓")

        # 4. Liquidity sweep + rejection
        if sweep_liquidity(d15, "LONG"):
            score += 20; reason_parts.append("Sweep + Bull Reject")
        if sweep_liquidity(d15, "SHORT"):
            score += 20; reason_parts.append("Sweep + Bear Reject")

        # 5. Volume spike
        if d15['volume'].iloc[-1] > d15['volume'].rolling(20).mean().iloc[-1] * 2:
            score += 15; reason_parts.append("Vol Spike")

        if score < CONFIG["MIN_SCORE"]:
            return None

        # Direction logic
        direction = "LONG" if ("Bull" in " ".join(reason_parts) or "FVG Fill ↑" in " ".join(reason_parts)) else "SHORT"

        # Real TP = next major liquidity or OB
        next_tp = price + 3.5 * atr15 if direction == "LONG" else price - 3.5 * atr15
        sl = price - atr15 if direction == "LONG" else price + atr15
        rr = abs(next_tp - price) / abs(sl - price)

        if rr < CONFIG["MIN_RR"]:
            return None

        return Signal(
            symbol=symbol,
            direction=direction,
            entry=price,  # market now OR limit if you want warning
            sl=sl,
            tp=next_tp,
            rr=round(rr, 2),
            score=score,
            reason=" | ".join(reason_parts[:4])
        )
    except:
        return None

# ======================= UI & LOOP =======================
def clear(): os.system('cls' if os.name == 'nt' else 'clear')

def beep(freq=1000, dur=300):
    global last_beep
    if time.time() - last_beep > 2:
        try: winsound.Beep(freq, dur)
        except: pass
        last_beep = time.time()

def send_tele(msg):
    try:
        import requests
        requests.get(f"https://api.telegram.org/bot{CONFIG['TELEGRAM_TOKEN']}/sendMessage?chat_id={CONFIG['TELEGRAM_CHAT_ID']}&text={msg}")
    except: pass

def print_signal(sig: Signal):
    clear()
    print("="*60)
    print("               GOD-TIER SETUP FOUND!")
    print("="*60)
    print(f"COIN      → {sig.symbol:12} Score: {sig.score}/100")
    print(f"Direction → {sig.direction}")
    print(f"Entry now → {sig.entry:.{CONFIG['PRICE_DECIMALS'].get(sig.symbol,3)}f}")
    print(f"Stop Loss → {sig.sl:.{CONFIG['PRICE_DECIMALS'].get(sig.symbol,3)}f}")
    print(f"TakeProfit→ {sig.tp:.{CONFIG['PRICE_DECIMALS'].get(sig.symbol,3)}f}")
    print(f"R:R       → 1:{sig.rr}")
    print(f"Reason    → {sig.reason}")
    print("="*60)
    print("   → Set limit order or market enter NOW")
    print("   → Trade locked — no new signals until TP/SL")
    print("="*60)
    beep(2000, 800)
    send_tele(f" {sig.direction} {sig.symbol}\nEntry: {sig.entry}\nSL {sig.sl}\nTP {sig.tp}\nRR 1:{sig.rr}\nScore {sig.score}")

def main():
    global active_trade
    print("Starting Sniper Brain v2 — press Ctrl+C to stop\n")
    time.sleep(2)
    
    while True:
        try:
            if active_trade:  # only monitor active trade
                ticker = exchange.fetch_ticker(active_trade.symbol)
                p = ticker['last']
                if (active_trade.direction == "LONG" and p >= active_trade.tp) or \
                   (active_trade.direction == "SHORT" and p <= active_trade.tp):
                    print(f"\n TAKE PROFIT HIT! {active_trade.symbol} {active_trade.direction}")
                    beep(2500, 1000)
                    active_trade = None
                elif (active_trade.direction == "LONG" and p <= active_trade.sl) or \
                     (active_trade.direction == "SHORT" and p >= active_trade.sl):
                    print(f"\n STOP LOSS HIT. {active_trade.symbol}")
                    beep(500, 1000)
                    active_trade = None
                else:
                    print(f"Active → {active_trade.symbol} {active_trade.direction} | Price {p:.4f} | TP {active_trade.tp:.4f}", end='\r')
                time.sleep(CONFIG["LOCKED_INTERVAL"])
                continue

            # FREE MODE — scan for new god-tier setup
            best = None
            for sym in CONFIG["SYMBOLS"]:
                try:
                    dfs = {tf: pd.DataFrame(exchange.fetch_ohlcv(sym, tf, limit=CONFIG["TIMEFRAMES"][tf])) 
                           for tf in CONFIG["TIMEFRAMES"]}
                    for tf in dfs:
                        if dfs[tf].empty: continue
                        dfs[tf].columns = ['ts','open','high','low','close','volume']
                        dfs[tf]['atr'] = atr(dfs[tf])
                    sig = calculate_score(sym, dfs)
                    if sig and (best is None or sig.score > best.score):
                        best = sig
                except: continue

            if best:
                active_trade = best
                print_signal(best)
            else:
                now = datetime.now().strftime("%H:%M:%S")
                print(f"[{now}] Scanning {len(CONFIG['SYMBOLS'])} coins — no god-tier yet...", end='\r')
                time.sleep(CONFIG["SCAN_INTERVAL"])

        except KeyboardInterrupt:
            print("\nBot stopped by user")
            break
        except Exception as e:
            print("ERROR:", e)
            time.sleep(5)

@app.route('/')  # ping endpoint for 24/7
def home():
    return "Bot alive!"

if __name__ == "__main__":
    from threading import Thread
    Thread(target=app.run, kwargs={'host': '0.0.0.0', 'port': 8080}).start()  # run web in background
    main()
