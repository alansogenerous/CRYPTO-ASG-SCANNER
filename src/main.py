"""Main entry point — SOL bot using yfinance only."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
from datetime import datetime
from typing import Dict, Any, Optional

import pandas as pd
import numpy as np
import yfinance as yf
import requests

# Removed Luno import — we only use yfinance now
from src.strategy import FractalMomentumStrategy
from src.alerts import (
    send_telegram_alert,
    format_signal_alert,
    format_heartbeat,
    send_startup_notification,
    send_health_report
)

STATE_FILE = "sol_bot_state.json"
LOG_FILE = "sol_bot.log"

def log_message(msg: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_msg = f"{timestamp} {msg}"
    print(full_msg)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(full_msg + "\n")

def load_state() -> Dict[str, Any]:
    default = {
        "version": "2.0",
        "run_count": 0,
        "last_signal": None,
        "last_signal_date": None,
        "last_price": 0.0,
        "last_check": None,
        "signal_history": [],
        "error_count": 0,
        "first_run": datetime.now().isoformat(),
        "entry_price": None,
        "entry_date": None,
        "highest_price": None,
        "trailing_active": False
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
    except Exception as e:
        log_message(f"⚠️ Error loading state: {e}")
        return default

def save_state(state: Dict[str, Any]) -> bool:
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2, default=str)
        return True
    except Exception as e:
        log_message(f"❌ Error saving state: {e}")
        return False

def get_usd_to_myr() -> float:
    """Get live USD/MYR exchange rate with fallback."""
    try:
        resp = requests.get("https://api.exchangerate.host/convert?from=USD&to=MYR&amount=1", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("success"):
                return float(data["result"])
    except Exception as e:
        log_message(f"⚠️ Exchange rate API error: {e}")
    # Fallback to a reasonable fixed rate
    return 4.70

def fetch_sol_data(timeframe: str = "daily") -> Optional[pd.DataFrame]:
    """Fetch SOL-USD data from Yahoo Finance and convert to MYR."""
    log_message(f"📊 Fetching SOL data from Yahoo Finance ({timeframe})...")
    
    usd_to_myr = get_usd_to_myr()
    log_message(f"💱 USD/MYR = {usd_to_myr:.4f}")
    
    interval = "1d" if timeframe == "daily" else "4h"
    period = "120d" if timeframe == "daily" else "30d"
    
    # Try yfinance download (compatible with various versions)
    try:
        # First attempt: without progress parameter (older versions)
        df = yf.download("SOL-USD", period=period, interval=interval, auto_adjust=False)
        if df.empty:
            raise ValueError("Empty DataFrame")
        # If it fails, we'll try with progress=False in except
    except Exception as e:
        log_message(f"⚠️ yfinance download (no progress) failed: {e}")
        # Second attempt: with progress=False (newer versions)
        try:
            df = yf.download("SOL-USD", period=period, interval=interval, auto_adjust=False, progress=False)
        except Exception as e2:
            log_message(f"❌ yfinance download with progress also failed: {e2}")
            # Third attempt: use Ticker.history()
            try:
                ticker = yf.Ticker("SOL-USD")
                df = ticker.history(period=period, interval=interval)
            except Exception as e3:
                log_message(f"❌ Ticker.history() failed: {e3}")
                return None
    
    if df is None or df.empty:
        log_message("❌ No data returned from Yahoo Finance")
        return None
    
    if len(df) < 50:
        log_message(f"⚠️ Only {len(df)} candles — insufficient for strategy (need 50+)")
        # Still return but strategy will handle insufficient data
        # We'll generate synthetic to keep bot alive? No, we'll just return what we have.
    
    # Reset index and clean column names
    df = df.reset_index()
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    
    # Convert price columns to MYR
    df['close'] = df['close'] * usd_to_myr
    df['open'] = df['open'] * usd_to_myr
    df['high'] = df['high'] * usd_to_myr
    df['low'] = df['low'] * usd_to_myr
    # Volume stays as is (in USD units, but we don't convert)
    
    log_message(f"✅ Fetched {len(df)} candles from Yahoo (converted to MYR)")
    return df

def main():
    log_message("=" * 70)
    log_message("🚀 SOL Fractal Momentum Bot v2.0 — 5/5 (yfinance only)")
    log_message("=" * 70)
    
    state = load_state()
    state["run_count"] = state.get("run_count", 0) + 1
    log_message(f"📋 Run #{state['run_count']}")
    
    if state["run_count"] == 1:
        send_startup_notification()
    
    try:
        timeframe = os.getenv("TIMEFRAME", "daily")
        df = fetch_sol_data(timeframe)
        if df is None:
            raise Exception("Failed to fetch data from Yahoo Finance")
        
        strategy = FractalMomentumStrategy(
            capital=50.0,
            rsi_period=14,
            rsi_oversold=30,
            rsi_overbought=70,
            atr_sl_mult=1.5,
            atr_tp_mult=3.0,
            trail_trigger_pct=4.0,
            trail_step_atr=1.0,
            timeframe=timeframe,
            min_trade_rm=5.0
        )
        
        if state.get("entry_price") is not None:
            strategy.position_open = True
            strategy.entry_price = state["entry_price"]
            strategy.entry_date = state.get("entry_date")
            strategy.highest_price = state.get("highest_price", state["entry_price"])
            strategy.trailing_active = state.get("trailing_active", False)
            # Recalculate SL/TP based on latest ATR from data
            if 'atr' in df.columns:
                latest_atr = df['atr'].iloc[-1]
            else:
                latest_atr = df['close'].pct_change().std() * df['close'].iloc[-1]
            strategy.stop_loss = strategy.entry_price - (1.5 * latest_atr)
            strategy.take_profit = strategy.entry_price + (3.0 * latest_atr)
            log_message(f"📂 Restored position: entry RM {strategy.entry_price:.2f}")
        
        action, price, tp, sl, meta = strategy.evaluate(df)
        
        current_price = df['close'].iloc[-1]
        log_message(f"📊 Current Price: RM {current_price:,.2f}")
        log_message(f"📊 RSI: {meta.get('rsi', 0):.1f}")
        log_message(f"📊 Trend: {meta.get('trend', 'N/A')}")
        log_message(f"📊 Signal: {action if action else 'NONE'}")
        
        signal_sent = False
        if action:
            log_message(f"🚨 {action} SIGNAL at RM {price:,.2f}")
            message = format_signal_alert(action, price, tp, sl, meta)
            success = send_telegram_alert(message)
            if success:
                signal_sent = True
                state["last_signal"] = action
                state["last_signal_date"] = datetime.now().isoformat()
                if "signal_history" not in state:
                    state["signal_history"] = []
                state["signal_history"].append({
                    "signal": action,
                    "price": price,
                    "date": datetime.now().isoformat(),
                    "meta": {k: v for k, v in meta.items() if isinstance(v, (str, int, float, bool))}
                })
                state["signal_history"] = state["signal_history"][-50:]
                if action == "BUY":
                    state["entry_price"] = price
                    state["entry_date"] = datetime.now().isoformat()
                    state["highest_price"] = price
                    state["trailing_active"] = False
                    log_message(f"📂 Position opened at RM {price:.2f}")
                if action in ("SELL", "TAKE_PROFIT", "STOP_LOSS", "TRAILING_EXIT"):
                    state["entry_price"] = None
                    state["entry_date"] = None
                    state["highest_price"] = None
                    state["trailing_active"] = False
                    log_message(f"📂 Position closed at RM {price:.2f}")
        else:
            log_message("💓 No signal — sending heartbeat")
            heartbeat_msg = format_heartbeat(current_price, meta, state)
            send_telegram_alert(heartbeat_msg)
        
        state["last_price"] = current_price
        state["last_check"] = datetime.now().isoformat()
        state["error_count"] = 0
        
        if state["run_count"] % 7 == 0:
            send_health_report(state)
        
        save_state(state)
        log_message("✅ Run complete")
        
    except Exception as e:
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        log_message(f"❌ Bot error: {error_msg}")
        state["error_count"] = state.get("error_count", 0) + 1
        save_state(state)
        send_telegram_alert(f"⚠️ BOT ERROR: {str(e)[:300]}")
    
    log_message("=" * 70)

if __name__ == "__main__":
    main()
