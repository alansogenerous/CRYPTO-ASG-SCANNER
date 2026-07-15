"""Technical indicators."""
import pandas as pd
import numpy as np
import yfinance as yf

def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    if len(series) < period:
        return pd.Series([np.nan] * len(series))
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window).mean()

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df['high']
    low = df['low']
    close = df['close']
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def fetch_4h_rsi(ticker: str = "SOL-USD") -> float:
    """Fetch 4‑hour RSI from Yahoo Finance using yfinance only."""
    try:
        # Try download with auto_adjust=False and progress=False (if supported)
        df = yf.download(ticker, period="7d", interval="4h", auto_adjust=False, progress=False)
        if df.empty or len(df) < 14:
            # If progress=False fails, try without it (some versions)
            df = yf.download(ticker, period="7d", interval="4h", auto_adjust=False)
        if df.empty:
            # Last fallback: Ticker.history()
            ticker_obj = yf.Ticker(ticker)
            df = ticker_obj.history(period="7d", interval="4h")
        if df.empty or len(df) < 14:
            return None
        close = df['Close']
        rsi = calculate_rsi(close, 14)
        return rsi.iloc[-1]
    except Exception as e:
        print(f"4H RSI fetch error: {e}")
        return None
