import os
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timezone
import traceback

# ========== CONFIG ==========
TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

SYMBOLS = {
    "BTC": "BTC-USD",
    "ETH": "ETH-USD"
}

STATE_FILES = {
    "BTC": "signal_state_btc.txt",
    "ETH": "signal_state_eth.txt"
}

# ========== FETCH DATA ==========
def get_daily_klines(symbol, limit=120):
    """Ambil data daily dari Yahoo Finance"""
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=f"{limit}d")
    df = df.rename(columns={
        "Open": "open", "High": "high",
        "Low": "low", "Close": "close", "Volume": "volume"
    })
    df = df[["open", "high", "low", "close"]].dropna()
    df = df.reset_index(drop=True)
    return df

def get_live_price(symbol):
    """Ambil harga terbaru (intraday) untuk rujukan live"""
    ticker = yf.Ticker(symbol)
    data = ticker.history(period="1d", interval="1m")
    if not data.empty:
        return data["Close"].iloc[-1]
    return None

# ========== INDIKATOR ==========
def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def atr(df, period=10):
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

def supertrend(df, period=10, multiplier=3):
    hl_avg = (df["high"] + df["low"]) / 2
    atr_val = atr(df, period)

    upper_band = hl_avg + multiplier * atr_val
    lower_band = hl_avg - multiplier * atr_val

    trend = [1]
    supertrend_vals = [lower_band.iloc[0]]

    for i in range(1, len(df)):
        prev_trend = trend[-1]
        prev_upper = upper_band.iloc[i-1]
        prev_lower = lower_band.iloc[i-1]
        curr_upper = upper_band.iloc[i]
        curr_lower = lower_band.iloc[i]
        close = df["close"].iloc[i]

        if prev_trend == 1:
            new_upper = min(curr_upper, prev_upper)
            new_lower = max(curr_lower, prev_lower)
        else:
            new_upper = min(curr_upper, prev_upper)
            new_lower = max(curr_lower, prev_lower)

        if close > new_upper:
            new_trend = 1
        elif close < new_lower:
            new_trend = -1
        else:
            new_trend = prev_trend

        trend.append(new_trend)
        supertrend_vals.append(new_lower if new_trend == 1 else new_upper)

    df["supertrend"] = supertrend_vals
    df["supertrend_trend"] = trend
    return df

def get_current_signal(df):
    df["ema12"] = ema(df["close"], 12)
    df["ema26"] = ema(df["close"], 26)
    df["macd"] = df["ema12"] - df["ema26"]
    df["signal"] = ema(df["macd"], 9)
    df["macd_hist"] = df["macd"] - df["signal"]

    df = supertrend(df, period=10, multiplier=3)

    last = df.iloc[-1]
    prev = df.iloc[-2]

    price_above_supertrend = last["close"] > last["supertrend"]
    macd_above_signal = last["macd"] > last["signal"]
    prev_macd_above_signal = prev["macd"] > prev["signal"]

    entry_long = (macd_above_signal and not prev_macd_above_signal) and price_above_supertrend
    exit_long = (not price_above_supertrend) or (not macd_above_signal and prev_macd_above_signal)

    if entry_long:
        return "ENTRY_LONG"
    elif exit_long:
        return "EXIT_LONG"
    else:
        return "NO_SIGNAL"

# ========== TELEGRAM ==========
def send_telegram(message):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML"}
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"Telegram send failed: {e}")

# ========== STATE MANAGEMENT ==========
def read_state(filepath):
    try:
        with open(filepath, "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        return "NO_SIGNAL"

def write_state(filepath, state):
    with open(filepath, "w") as f:
        f.write(state)

# ========== MAIN ==========
def process_symbol(label, yahoo_symbol, state_file):
    print(f"--- Processing {label} ---")
    df = get_daily_klines(yahoo_symbol)
    signal = get_current_signal(df)
    last_state = read_state(state_file)

    print(f"Signal: {signal}, Last state: {last_state}")

    if signal != "NO_SIGNAL" and signal != last_state:
        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        close_price = df["close"].iloc[-1]
        live_price = get_live_price(yahoo_symbol)
        live_str = f"${live_price:,.2f}" if live_price else "N/A"

        if signal == "ENTRY_LONG":
            msg = (
                f"<b>🚀 {label}/USDT LONG SIGNAL</b>\n\n"
                f"Harga Close (sumber isyarat): ${close_price:,.2f}\n"
                f"Harga Live: {live_str}\n"
                f"Masa: {now_utc}\n\n"
                f"<i>Entry long spot 100% modal.</i>"
            )
        elif signal == "EXIT_LONG":
            msg = (
                f"<b>🔻 {label}/USDT EXIT SIGNAL</b>\n\n"
                f"Harga Close: ${close_price:,.2f}\n"
                f"Harga Live: {live_str}\n"
                f"Masa: {now_utc}\n\n"
                f"<i>Tutup semua posisi long.</i>"
            )
        send_telegram(msg)
        print("Alert sent!")
        write_state(state_file, signal)
    else:
        # Optionally, if no change but signal is active, we could resend? No, keep silent.
        write_state(state_file, signal)  # ensure state is current

def main():
    success = True
    for label, ysym in SYMBOLS.items():
        try:
            process_symbol(label, ysym, STATE_FILES[label])
        except Exception as e:
            success = False
            error_msg = f"❌ <b>Error processing {label}</b>\n<pre>{traceback.format_exc()}</pre>"
            send_telegram(error_msg)
            print(f"Error on {label}: {traceback.format_exc()}")

    if not success:
        # force exit non-zero so GitHub Actions marks as failed
        exit(1)

if __name__ == "__main__":
    main()
