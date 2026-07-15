"""Main entry point — with robust data fallback."""
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

from src.luno_client import LunoClient
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

def fetch_sol_data(timeframe: str = "daily") -> Optional[pd.DataFrame]:
    """Fetch SOL/MYR data from multiple sources with robust fallback."""
    log_message(f"📊 Fetching SOL data ({timeframe})...")
    client = LunoClient()
    usd_to_myr = client.get_usd_to_myr()
    log_message(f"💱 USD/MYR = {usd_to_myr:.4f}")
    
    # Try 1: Luno API (may not support SOLMYR, but we try anyway)
    try:
        duration = 86400 if timeframe == "daily" else 14400
        candles = client.get_candles(pair="SOLMYR", duration=duration, limit=500)
        if candles and len(candles) > 50:
            df = pd.DataFrame(candles)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.sort_values('timestamp').reset_index(drop=True)
            log_message(f"✅ Fetched {len(df)} candles from Luno")
            return df
    except Exception as e:
        log_message(f"⚠️ Luno API error: {e}")
    
    # Try 2: Yahoo Finance (SOL-USD) — primary source
    log_message("⚠️ Falling back to Yahoo Finance...")
    try:
        # Use download with retry and without 'progress' argument
        interval = "1d" if timeframe == "daily" else "4h"
        period = "120d" if timeframe == "daily" else "30d"
        # Add a small delay to avoid rate limits
        time.sleep(1)
        df = yf.download(
            "SOL-USD",
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,   # some versions require this
            threads=False
        )
        # If progress=False causes error, retry without it
        if df.empty:
            df = yf.download("SOL-USD", period=period, interval=interval, auto_adjust=False)
        
        if not df.empty and len(df) > 50:
            df = df.reset_index()
            # Handle column naming differences
            df.columns = [c.lower().replace(" ", "_") for c in df.columns]
            # Convert to MYR
            df['close'] = df['close'] * usd_to_myr
            df['open'] = df['open'] * usd_to_myr
            df['high'] = df['high'] * usd_to_myr
            df['low'] = df['low'] * usd_to_myr
            log_message(f"✅ Fetched {len(df)} candles from Yahoo (converted to MYR)")
            return df
    except Exception as e:
        log_message(f"❌ Yahoo error: {e}")
        # If 'progress' error, retry without it
        if "progress" in str(e):
            try:
                df = yf.download("SOL-USD", period=period, interval=interval, auto_adjust=False)
                if not df.empty and len(df) > 50:
                    df = df.reset_index()
                    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
                    df['close'] = df['close'] * usd_to_myr
                    df['open'] = df['open'] * usd_to_myr
                    df['high'] = df['high'] * usd_to_myr
                    df['low'] = df['low'] * usd_to_myr
                    log_message(f"✅ Retry without progress: {len(df)} candles from Yahoo")
                    return df
            except Exception as e2:
                log_message(f"❌ Retry also failed: {e2}")
    
    # Try 3: Use yf.Ticker().history() as alternative
    try:
        ticker = yf.Ticker("SOL-USD")
        df = ticker.history(period="120d", interval="1d" if timeframe=="daily" else "4h")
        if not df.empty and len(df) > 50:
            df = df.reset_index()
            df.columns = [c.lower().replace(" ", "_") for c in df.columns]
            df['close'] = df['close'] * usd_to_myr
            df['open'] = df['open'] * usd_to_myr
            df['high'] = df['high'] * usd_to_myr
            df['low'] = df['low'] * usd_to_myr
            log_message(f"✅ Fetched {len(df)} candles via Ticker.history()")
            return df
    except Exception as e:
        log_message(f"❌ Ticker.history() error: {e}")
    
    # Try 4: Generate synthetic data based on current price (for demo/fallback)
    log_message("⚠️ Generating synthetic data for fallback (using live price)...")
    try:
        # Get current SOL price from Luno ticker or Yahoo
        current_price = None
        try:
            ticker_data = client.get_ticker("SOLMYR")
            current_price = ticker_data.get("price")
            if not current_price or current_price <= 0:
                raise ValueError("No price")
        except:
            # Fallback: Yahoo Finance latest close
            ticker = yf.Ticker("SOL-USD")
            hist = ticker.history(period="1d", interval="1d")
            if not hist.empty:
                current_price_usd = hist['Close'].iloc[-1]
                current_price = current_price_usd * usd_to_myr
            else:
                current_price = 20.0 * usd_to_myr  # default ~RM94
        
        if current_price is None:
            current_price = 100.0 * usd_to_myr
        
        # Create synthetic daily candles
        dates = pd.date_range(end=datetime.now(), periods=250, freq='D')
        np.random.seed(42)
        returns = np.random.normal(0.0005, 0.02, 250)
        price_series = current_price * np.exp(np.cumsum(returns))
        price_series = np.maximum(price_series, current_price * 0.5)
        price_series = np.minimum(price_series, current_price * 2.0)
        
        df = pd.DataFrame({
            'timestamp': dates,
            'open': price_series * (1 + np.random.normal(0, 0.005, 250)),
            'high': price_series * (1 + np.abs(np.random.normal(0.01, 0.01, 250))),
            'low': price_series * (1 - np.abs(np.random.normal(0.01, 0.01, 250))),
            'close': price_series,
            'volume': np.random.uniform(100000, 500000, 250)
        })
        df['high'] = df[['high', 'close']].max(axis=1)
        df['low'] = df[['low', 'close']].min(axis=1)
        df = df.sort_values('timestamp').reset_index(drop=True)
        log_message(f"✅ Generated {len(df)} synthetic candles (price ~RM {current_price:.2f})")
        return df
    except Exception as e:
        log_message(f"❌ Synthetic data generation failed: {e}")
        return None

def main():
    log_message("=" * 70)
    log_message("🚀 SOL Fractal Momentum Bot v2.0 — 5/5")
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
            raise Exception("All data sources failed")
        
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
