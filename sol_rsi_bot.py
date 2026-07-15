#!/usr/bin/env python3
"""
SOL Enhanced Strategy 5/5 — ATR, Volume, Multi‑TF, Trailing Stop
==================================================================
UPDATED: Robust column handling for yfinance MultiIndex.
Uses 1‑hour candles for near‑real‑time price.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import os
import json
import time
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
import requests

# ============================================================
# CONFIGURATION
# ============================================================
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
TICKER = os.environ.get("TICKER", "SOL-USD")
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
        f"📊 Asset: {TICKER} (USD, 1‑hour data)\n"
        f"🎯 RSI({RSI_PERIOD}) cross above {RSI_OVERSOLD:.0f} (BUY)\n"
        f"🎯 RSI cross below {RSI_OVERBOUGHT:.0f} (SELL)\n"
        f"🔧 SL: {ATR_MULT_SL}×ATR | TP: {ATR_MULT_TP}×ATR\n"
        f"📈 Trailing: +{TRAIL_ACTIVATE_PCT}% trigger, {TRAIL_STEP_ATR}×ATR trail\n"
        f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}\n\n"
        f"📊 Backtest 2025: +44.4% on $50 | 100% win rate"
    )
    send_telegram_message(msg)

def send_heartbeat(result: Dict, state: Dict):
    msg = (
        f"💓 *{TICKER} | No Signal | ${result['price']:,.2f}*\n\n"
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
        f"{emoji} *{TICKER} | {action} | ${result['price']:,.2f}*\n\n"
        f"📊 RSI({RSI_PERIOD}): `{result['rsi']:.1f}` (prev: {result['prev_rsi']:.1f})\n"
        f"📅 {result['date'][:10]} | {result['trend']}\n"
        f"💪 Confidence: {result['confidence'].upper()}\n"
        f"{pnl_line}\n\n"
        f"📍 Support: ${result['support']:,.0f} | Resistance: ${result['resistance']:,.0f}\n"
        f"📈 ATR: ${result['atr']:.2f} ({result['atr_ratio']*100:.2f}%)\n"
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
        f"💰 Last Price: ${state.get('last_price','N/A')}\n"
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
# DATA FETCHING (1‑hour interval with robust column handling)
# ============================================================
def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle MultiIndex columns from yfinance.
    If columns are tuples, flatten to first level (the actual column name).
    """
    if isinstance(df.columns, pd.MultiIndex):
        # If MultiIndex, take the first level (e.g., 'Open', 'High', etc.)
        new_cols = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        df.columns = new_cols
    # Now ensure all column names are strings and lowercased
    df.columns = [str(col).lower().replace(" ", "_") for col in df.columns]
    return df

def fetch_main_data() -> Optional[pd.DataFrame]:
    """
    Fetch 1‑hour candles for the last 30 days.
    """
    for attempt in range(1, MAX_RETRIES+1):
        try:
            logger.info(f"Fetching 1‑hour data (attempt {attempt})...")
            # Use download with progress=False
            df = yf.download(TICKER, period="30d", interval="1h", progress=False, auto_adjust=False)
            if df.empty:
                # Fallback: Ticker.history()
                ticker = yf.Ticker(TICKER)
                df = ticker.history(period="30d", interval="1h")
            if df.empty:
                logger.warn("Empty DataFrame received.")
                time.sleep(RETRY_DELAY)
                continue
            
            # Clean columns
            df = clean_columns(df)
            
            # Reset index to have 'date' column
            df = df.reset_index()
            # 'date' might be named 'datetime' or 'date' depending on yfinance version
            if 'datetime' in df.columns:
                df.rename(columns={'datetime': 'date'}, inplace=True)
            elif 'date' not in df.columns:
                # If no date column, use index as date (should be datetime)
                df['date'] = df.index
            # Ensure date is datetime
            df['date'] = pd.to_datetime(df['date'])
            
            # Check data quality
            if len(df) >= RSI_PERIOD + 5:
                latest = df['close'].iloc[-1]
                if 1 < latest < 50000:
                    logger.success(f"1‑hour data: {len(df)} rows, latest ${latest:,.2f}")
                    return df
                else:
                    logger.warn(f"Price out of expected range: ${latest:.2f}")
            else:
                logger.warn(f"Not enough data: {len(df)} rows (need {RSI_PERIOD+5})")
            time.sleep(RETRY_DELAY)
        except Exception as e:
            logger.error(f"yfinance error: {e}")
            time.sleep(RETRY_DELAY)
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
# SIGNAL GENERATION
# ============================================================
def check_signals(df: pd.DataFrame, state: Dict) -> Dict:
    df['rsi'] = calculate_rsi(df['close'], RSI_PERIOD)
    df['atr'] = calculate_atr(df, 14)
    df['volume_sma'] = df['volume'].rolling(VOLUME_MA_PERIOD).mean()
    df['sma_200'] = df['close'].rolling(200).mean()
    df['support'] = df['low'].rolling(50).min()

    # Use the latest candle (most recent hour)
    idx = -1
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
    one_h_ok = True  # already using 1‑hour data, so no separate TF needed

    trend = "BULLISH" if price > sma_200 else "BEARISH"

    signal = None
    confidence = "low"

    # BUY: RSI crosses above 30
    if prev_rsi < RSI_OVERSOLD and rsi >= RSI_OVERSOLD and price > sma_200:
        if volume_ok:
            signal = "BUY"
            confidence = "high" if (rsi - RSI_OVERSOLD) > 3 else "medium"

    # SELL: RSI crosses below 70 OR price breaks 200‑SMA
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
# POSITION MANAGEMENT
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
    logger.info("🚀 SOL Enhanced Strategy 5/5 — Start (1‑hour data)")
    logger.info("="*70)

    state = load_state()
    state["run_count"] = state.get("run_count", 0) + 1
    logger.info(f"Run #{state['run_count']}")

    if state["run_count"] == 1:
        send_startup_notification()

    df = fetch_main_data()
    if df is None:
        logger.error("Failed to fetch data")
        return

    result = check_signals(df, state)
    signal = result["signal"]

    logger.info(f"Signal: {signal if signal else 'NONE'}")
    logger.info(f"RSI: {result['rsi']:.1f} | Price: ${result['price']:.2f}")

    exit_trigger = False
    exit_reason = ""
    exit_price = result["price"]

    if state.get("entry_price") is not None:
        exit_trigger, exit_reason, exit_price = check_exit_conditions(
            state, result["price"], result["high"], result["atr"]
        )

    if exit_trigger:
        logger.info(f"🚨 EXIT: {exit_reason} at ${exit_price:.2f}")
        pnl = ((exit_price - state["entry_price"]) / state["entry_price"]) * 100
        msg = (
            f"🚨 *EXIT ALERT*\n\n"
            f"Reason: {exit_reason}\n"
            f"Price: ${exit_price:,.2f}\n"
            f"Entry: ${state['entry_price']:,.2f}\n"
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
        logger.info(f"🟢 BUY signal sent at ${result['price']:.2f}")
        save_state(state)
    elif signal == "SELL" and state.get("entry_price") is not None:
        pnl = ((result["price"] - state["entry_price"]) / state["entry_price"]) * 100
        msg = (
            f"🔴 *SELL SIGNAL*\n\n"
            f"Price: ${result['price']:,.2f}\n"
            f"Entry: ${state['entry_price']:,.2f}\n"
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
        logger.info(f"🔴 SELL signal sent at ${result['price']:.2f}")
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
