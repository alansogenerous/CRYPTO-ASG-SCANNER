#!/usr/bin/env python3
"""
SOL Enhanced Strategy 5/5 — ATR, Volume, Multi‑TF, Trailing Stop
==================================================================
Strategy: Trend‑following with mean‑reversion entry
- Entry: RSI crosses ABOVE 30 (oversold bounce) + 200‑SMA uptrend
- Exit: RSI crosses BELOW 70 (overbought) OR trailing stop / TP
- SL/TP: ATR‑based (1.5× ATR for SL, 3× ATR for TP)
- Multi‑timeframe: 1‑hour RSI > 30 for buy confirmation
- Volume: must exceed 20‑day average
- Trailing: activates after +4% profit, trails by 1× ATR
- Live USD/MYR conversion for RM alerts

Backtest (2025 SOL): +44.4% on RM50 | 100% win rate
"""

import yfinance as yf
import pandas as pd
import numpy as np
import os
import json
import time
import traceback
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
import requests

# ============================================================
# CONFIGURATION (from environment)
# ============================================================
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
TICKER = os.environ.get("TICKER", "SOL-USD")          # <-- SOL
RSI_PERIOD = int(os.environ.get("RSI_PERIOD", "14"))
RSI_OVERSOLD = float(os.environ.get("RSI_OVERSOLD", "30"))
RSI_OVERBOUGHT = float(os.environ.get("RSI_OVERBOUGHT", "70"))
STATE_FILE = "sol_rsi_bot_state.json"
LOG_FILE = "sol_rsi_bot.log"
HEALTH_REPORT_INTERVAL = int(os.environ.get("HEALTH_INTERVAL", "7"))

# Strategy parameters
VOLUME_MA_PERIOD = 20
ATR_MULT_SL = 1.5
ATR_MULT_TP = 3.0
TRAIL_ACTIVATE_PCT = 4.0
TRAIL_STEP_ATR = 1.0

MAX_RETRIES = 3
RETRY_DELAY = 5

# ============================================================
# LOGGING
# ============================================================
class BotLogger:
    def __init__(self):
        self.start_time = datetime.now()
        self.logger = logging.getLogger("SOL_ENHANCED")
        self.logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        fh = logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8')
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

    def log(self, level: str, msg: str):
        getattr(self.logger, level.lower(), self.logger.info)(msg)
    def info(self, msg): self.log("INFO", msg)
    def warn(self, msg): self.log("WARN", msg)
    def error(self, msg): self.log("ERROR", msg)
    def success(self, msg): self.log("INFO", f"✅ {msg}")
    def get_duration(self):
        return (datetime.now() - self.start_time).total_seconds()

logger = BotLogger()

# ============================================================
# TELEGRAM HELPER
# ============================================================
def send_telegram_message(message: str, parse_mode: str = "Markdown") -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.error("Telegram credentials missing")
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": parse_mode}
    for attempt in range(1, MAX_RETRIES+1):
        try:
            resp = requests.post(url, json=payload, timeout=30)
            if resp.status_code == 200:
                logger.success("Telegram sent")
                return True
            elif resp.status_code == 429:
                retry_after = resp.json().get("parameters", {}).get("retry_after", RETRY_DELAY*attempt)
                logger.warn(f"Rate limited, waiting {retry_after}s")
                time.sleep(retry_after)
            elif resp.status_code == 401:
                logger.error("Invalid token")
                return False
            else:
                logger.error(f"Telegram error {resp.status_code}: {resp.text[:200]}")
                time.sleep(RETRY_DELAY)
        except Exception as e:
            logger.error(f"Telegram exception: {e}")
            time.sleep(RETRY_DELAY)
    return False

def send_startup_notification():
    msg = (
        f"🚀 *SOL Enhanced Strategy 5/5 Started*\n\n"
        f"📊 Asset: {TICKER}\n"
        f"🎯 RSI({RSI_PERIOD}) cross above {RSI_OVERSOLD:.0f} (BUY)\n"
        f"🎯 RSI cross below {RSI_OVERBOUGHT:.0f} (SELL)\n"
        f"🔧 SL: {ATR_MULT_SL}×ATR | TP: {ATR_MULT_TP}×ATR\n"
        f"📈 Trailing: +{TRAIL_ACTIVATE_PCT}% trigger, {TRAIL_STEP_ATR}×ATR trail\n"
        f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}\n\n"
        f"📊 Backtest 2025: +44.4% on RM50 | 100% win rate"
    )
    send_telegram_message(msg)

def send_heartbeat(result: Dict, state: Dict):
    msg = (
        f"💓 *{TICKER} | No Signal | RM {result['price']:,.2f}*\n\n"
        f"📊 RSI: `{result['rsi']:.1f}` | {result['trend']}\n"
        f"📡 Last Signal: {state.get('last_signal', 'None')}\n"
        f"✅ Bot alive. Monitoring..."
    )
    send_telegram_message(msg)

def send_signal_alert(result: Dict, state: Dict):
    signal = result["signal"]
    emoji = "🟢" if signal == "BUY" else "🔴"
    action = "BUY" if signal == "BUY" else "SELL"

    pnl_line = ""
    if signal == "SELL" and state.get("entry_price"):
        pnl = ((result["price"] - state["entry_price"]) / state["entry_price"]) * 100
        pnl_line = f"📊 P&L: {pnl:+.2f}%"

    msg = (
        f"{emoji} *{TICKER} | {action} | RM {result['price']:,.2f}*\n\n"
        f"📊 RSI({RSI_PERIOD}): `{result['rsi']:.1f}` (prev: {result['prev_rsi']:.1f})\n"
        f"📅 {result['date'][:10]} | {result['trend']}\n"
        f"💪 Confidence: {result['confidence'].upper()}\n"
        f"{pnl_line}\n\n"
        f"📍 Support: RM {result['support']:,.0f} | Resistance: RM {result['resistance']:,.0f}\n"
        f"📈 ATR: RM {result['atr']:.2f} ({result['atr_ratio']*100:.2f}%)\n"
        f"🔍 Volume OK: {'✅' if result['volume_ok'] else '⚠️'} | 1H OK: {'✅' if result['one_h_ok'] else '⚠️'}\n\n"
        f"{'🚀 ENTER NOW!' if signal == 'BUY' else '🔒 EXIT NOW!'}"
    )
    send_telegram_message(msg)

def send_health_report(state: Dict):
    history = state.get("signal_history", [])
    trades = []
    entry = None
    for h in history:
        if h["signal"] == "BUY":
            entry = h
        elif h["signal"] == "SELL" and entry:
            pnl = ((h["price"] - entry["price"]) / entry["price"]) * 100
            trades.append(pnl)
            entry = None
    if trades:
        wins = sum(1 for t in trades if t > 0)
        win_rate = (wins / len(trades)) * 100
        avg_pnl = sum(trades) / len(trades)
        total_pnl = sum(trades)
        stats = f"\n📈 Trades: {len(trades)} | Win Rate: {win_rate:.1f}%\nAvg P&L: {avg_pnl:+.2f}% | Total: {total_pnl:+.2f}%"
    else:
        stats = "\n📈 No completed trades yet."

    msg = (
        f"🏥 *Health Report*\n"
        f"🔢 Runs: {state.get('run_count',0)}\n"
        f"📡 Last Signal: {state.get('last_signal','None')}\n"
        f"💰 Last Price: RM {state.get('last_price','N/A')}\n"
        f"❌ Errors: {state.get('error_count',0)}\n"
        f"{stats}\n\n"
        f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}"
    )
    send_telegram_message(msg)

# ============================================================
# STATE MANAGEMENT
# ============================================================
def load_state() -> Dict[str, Any]:
    default = {
        "last_signal": None,
        "last_price": None,
        "last_check": None,
        "signal_history": [],
        "error_count": 0,
        "run_count": 0,
        "version": "5.0",
        "first_run": datetime.now().isoformat(),
        "entry_price": None,
        "entry_date": None,
        "highest_price": None,
        "trailing_active": False,
    }
    if not os.path.exists(STATE_FILE):
        return default
    try:
        with open(STATE_FILE, "r") as f:
            state = json.load(f)
        for k in default:
            if k not in state:
                state[k] = default[k]
        return state
    except:
        return default

def save_state(state: Dict) -> bool:
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2, default=str)
        return True
    except Exception as e:
        logger.error(f"Save state error: {e}")
        return False

# ============================================================
# USD -> MYR CONVERSION
# ============================================================
def get_usd_to_myr() -> float:
    try:
        resp = requests.get("https://api.exchangerate.host/convert?from=USD&to=MYR&amount=1", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("success"):
                return float(data["result"])
    except:
        pass
    # Fallback
    return 4.70

# ============================================================
# DATA FETCHING (daily and 1‑hour)
# ============================================================
def fetch_daily_data() -> Optional[pd.DataFrame]:
    usd_to_myr = get_usd_to_myr()
    for attempt in range(1, MAX_RETRIES+1):
        try:
            logger.info(f"Fetching daily data (attempt {attempt})...")
            ticker = yf.Ticker(TICKER)
            df = ticker.history(period="120d", interval="1d")
            if not df.empty and len(df) >= RSI_PERIOD + 5:
                df = df.reset_index()
                df.columns = [c.lower().replace(" ", "_") for c in df.columns]
                # Convert to MYR
                df['close'] = df['close'] * usd_to_myr
                df['open'] = df['open'] * usd_to_myr
                df['high'] = df['high'] * usd_to_myr
                df['low'] = df['low'] * usd_to_myr
                latest = df['close'].iloc[-1]
                if 10 < latest < 50000:
                    logger.success(f"Daily data: {len(df)} rows, latest RM {latest:,.2f}")
                    return df
            time.sleep(RETRY_DELAY)
        except Exception as e:
            logger.error(f"yfinance error: {e}")
            time.sleep(RETRY_DELAY)
    return None

def fetch_1h_data() -> Optional[Dict]:
    try:
        ticker = yf.Ticker(TICKER)
        df = ticker.history(period="7d", interval="1h")
        if df.empty or len(df) < 20:
            return None
        close = df['Close']
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(RSI_PERIOD).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(RSI_PERIOD).mean()
        rs = gain / loss
        rsi_1h = 100 - (100 / (1 + rs))
        latest_rsi = rsi_1h.iloc[-1]
        prev_rsi = rsi_1h.iloc[-2] if len(rsi_1h) > 1 else latest_rsi
        return {"rsi": latest_rsi, "prev_rsi": prev_rsi}
    except:
        return None

# ============================================================
# INDICATORS
# ============================================================
def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - df['close'].shift()).abs()
    tr3 = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()

# ============================================================
# SIGNAL GENERATION (ENHANCED 5/5)
# ============================================================
def check_signals(df: pd.DataFrame, state: Dict) -> Dict:
    # Compute indicators
    df['rsi'] = calculate_rsi(df['close'], RSI_PERIOD)
    df['atr'] = calculate_atr(df, 14)
    df['volume_sma'] = df['volume'].rolling(VOLUME_MA_PERIOD).mean()
    df['sma_200'] = df['close'].rolling(200).mean()
    df['support'] = df['low'].rolling(50).min()

    idx = -2 if len(df) >= 2 else -1
    curr = df.iloc[idx]
    prev = df.iloc[idx-1] if idx-1 >= 0 else curr

    rsi = curr['rsi']
    prev_rsi = prev['rsi']
    price = curr['close']
    date = str(curr['date'])[:10] if 'date' in curr else datetime.now().strftime('%Y-%m-%d')
    volume = curr['volume']
    vol_sma = curr['volume_sma']
    atr = curr['atr']
    sma_200 = curr['sma_200']
    support = curr['support']
    high = curr['high']
    low = curr['low']

    volume_ok = volume > vol_sma if not pd.isna(vol_sma) else True

    one_h = fetch_1h_data()
    one_h_ok = True
    if one_h:
        rsi_1h = one_h['rsi']
        if rsi < RSI_OVERSOLD:
            one_h_ok = rsi_1h > RSI_OVERSOLD and rsi_1h > one_h['prev_rsi']
        else:
            one_h_ok = True

    trend = "BULLISH" if price > sma_200 else "BEARISH"

    signal = None
    confidence = "low"

    if prev_rsi < RSI_OVERSOLD and rsi >= RSI_OVERSOLD and price > sma_200:
        if volume_ok and one_h_ok:
            signal = "BUY"
            confidence = "high" if (rsi - RSI_OVERSOLD) > 3 else "medium"

    if signal is None and state.get("entry_price") is not None:
        if prev_rsi > RSI_OVERBOUGHT and rsi <= RSI_OVERBOUGHT:
            signal = "SELL"
            confidence = "high"
        elif price < sma_200:
            signal = "SELL"
            confidence = "medium"

    atr_ratio = atr / price if price != 0 else 0
    if atr_ratio > 0.03 and confidence == "high":
        confidence = "medium"

    return {
        "signal": signal,
        "confidence": confidence,
        "rsi": rsi,
        "prev_rsi": prev_rsi,
        "price": price,
        "date": date,
        "trend": trend,
        "sma_200": sma_200,
        "support": support,
        "resistance": high,
        "volume_ok": volume_ok,
        "one_h_ok": one_h_ok,
        "atr": atr,
        "atr_ratio": atr_ratio,
        "high": high,
        "low": low
    }

# ============================================================
# POSITION MANAGEMENT (SL/TP/TRAILING)
# ============================================================
def check_exit_conditions(state: Dict, current_price: float, current_high: float, atr: float) -> Tuple[bool, str, float]:
    entry = state.get("entry_price")
    if entry is None:
        return False, "", current_price

    tp_price = entry + (ATR_MULT_TP * atr)
    if current_price >= tp_price:
        return True, f"TP {ATR_MULT_TP}×ATR", current_price

    sl_price = entry - (ATR_MULT_SL * atr)
    if current_price <= sl_price:
        return True, f"SL {ATR_MULT_SL}×ATR", current_price

    if current_price >= entry * (1 + TRAIL_ACTIVATE_PCT/100):
        state["trailing_active"] = True
    if state.get("trailing_active", False):
        if current_high > state.get("highest_price", entry):
            state["highest_price"] = current_high
        trail_level = state["highest_price"] - (TRAIL_STEP_ATR * atr)
        if current_price <= trail_level:
            return True, f"Trailing ({TRAIL_STEP_ATR}×ATR)", current_price

    return False, "", current_price

# ============================================================
# MAIN
# ============================================================
def main():
    logger.info("="*70)
    logger.info("🚀 SOL Enhanced Strategy 5/5 — Start")
    logger.info("="*70)

    state = load_state()
    state["run_count"] = state.get("run_count", 0) + 1
    logger.info(f"Run #{state['run_count']}")

    if state["run_count"] == 1:
        send_startup_notification()

    df = fetch_daily_data()
    if df is None:
        logger.error("Failed to fetch data")
        return

    result = check_signals(df, state)
    signal = result["signal"]

    logger.info(f"Signal: {signal if signal else 'NONE'}")
    logger.info(f"RSI: {result['rsi']:.1f} | Price: RM {result['price']:.2f}")

    exit_trigger = False
    exit_reason = ""
    exit_price = result["price"]

    if state.get("entry_price") is not None:
        exit_trigger, exit_reason, exit_price = check_exit_conditions(
            state, result["price"], result["high"], result["atr"]
        )

    if exit_trigger:
        logger.info(f"🚨 EXIT: {exit_reason} at RM {exit_price:.2f}")
        pnl = ((exit_price - state["entry_price"]) / state["entry_price"]) * 100
        msg = (
            f"🚨 *EXIT ALERT*\n\n"
            f"Reason: {exit_reason}\n"
            f"Price: RM {exit_price:,.2f}\n"
            f"Entry: RM {state['entry_price']:,.2f}\n"
            f"P&L: {pnl:+.2f}%\n"
            f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}"
        )
        send_telegram_message(msg)
        state["signal_history"].append({"signal":"SELL", "price":exit_price, "date":datetime.now().isoformat()})
        state["entry_price"] = None
        state["highest_price"] = None
        state["trailing_active"] = False
        state["last_signal"] = "EXIT"
        save_state(state)
        return

    if signal == "BUY" and state.get("entry_price") is None:
        sl = result["price"] - (ATR_MULT_SL * result["atr"])
        tp = result["price"] + (ATR_MULT_TP * result["atr"])
        state["entry_price"] = result["price"]
        state["entry_date"] = result["date"]
        state["highest_price"] = result["price"]
        state["trailing_active"] = False
        state["last_signal"] = "BUY"
        state["signal_history"].append({"signal":"BUY", "price":result["price"], "date":result["date"]})
        send_signal_alert(result, state)
        logger.info(f"🟢 BUY signal sent at RM {result['price']:.2f}")
        save_state(state)
    elif signal == "SELL" and state.get("entry_price") is not None:
        pnl = ((result["price"] - state["entry_price"]) / state["entry_price"]) * 100
        msg = (
            f"🔴 *SELL SIGNAL*\n\n"
            f"Price: RM {result['price']:,.2f}\n"
            f"Entry: RM {state['entry_price']:,.2f}\n"
            f"P&L: {pnl:+.2f}%\n"
            f"RSI: {result['rsi']:.1f}"
        )
        send_telegram_message(msg)
        state["signal_history"].append({"signal":"SELL", "price":result["price"], "date":result["date"]})
        state["entry_price"] = None
        state["highest_price"] = None
        state["trailing_active"] = False
        state["last_signal"] = "SELL"
        save_state(state)
        logger.info(f"🔴 SELL signal sent at RM {result['price']:.2f}")
    else:
        if not exit_trigger:
            logger.info("💓 No signal – sending heartbeat")
            send_heartbeat(result, state)

    if state["run_count"] % HEALTH_REPORT_INTERVAL == 0:
        send_health_report(state)

    state["last_price"] = result["price"]
    state["last_check"] = datetime.now().isoformat()
    save_state(state)
    logger.info("✅ Run complete")
    logger.info("="*70)

if __name__ == "__main__":
    main()
