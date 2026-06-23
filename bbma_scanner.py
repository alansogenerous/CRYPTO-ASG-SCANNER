import os
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from typing import Dict, Optional
from enum import Enum

# ==========================================
# CONFIGURATION
# ==========================================
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')

if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
    raise ValueError("Missing Telegram credentials in GitHub Secrets!")

PAIRS = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD', 'XRP-USD']

STYLES = {
    'Intraday': {'big': '4h', 'small': '1h', 'lookback_days': 60},
    'Swing': {'big': '1d', 'small': '4h', 'lookback_days': 365}
}

BB_PERIOD = 20
BB_STD = 2.0

class BBMAState(Enum):
    NONE = 0
    EXTREME_BUY = 1
    MHV_BUY = 2
    CSA_BUY = 3
    REENTRY_BUY = 4

# ==========================================
# INDICATOR CALCULATIONS
# ==========================================
def calculate_lwma(series: pd.Series, period: int) -> pd.Series:
    weights = np.arange(1, period + 1)
    def lwma(window):
        return np.sum(window * weights) / np.sum(weights)
    return series.rolling(window=period).apply(lwma, raw=True)

def get_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    if 'Close' not in df.columns:
        close_col = [c for c in df.columns if 'close' in c.lower()]
        if close_col:
            df['Close'] = df[close_col[0]]
            df['Open'] = df[[c for c in df.columns if 'open' in c.lower()][0]]
            df['High'] = df[[c for c in df.columns if 'high' in c.lower()][0]]
            df['Low'] = df[[c for c in df.columns if 'low' in c.lower()][0]]
        else:
            return pd.DataFrame()

    df['bb_mid'] = df['Close'].rolling(BB_PERIOD).mean()
    bb_std = df['Close'].rolling(BB_PERIOD).std()
    df['bb_upper'] = df['bb_mid'] + (bb_std * BB_STD)
    df['bb_lower'] = df['bb_mid'] - (bb_std * BB_STD)
    
    df['ma5_high'] = calculate_lwma(df['High'], 5)
    df['ma10_high'] = calculate_lwma(df['High'], 10)
    df['ma5_low'] = calculate_lwma(df['Low'], 5)
    df['ma10_low'] = calculate_lwma(df['Low'], 10)
    
    df['ema50'] = df['Close'].ewm(span=50, adjust=False).mean()
    
    return df.dropna()

# ==========================================
# DATA FETCHER WITH 2026 VALIDATION
# ==========================================
def fetch_yfinance_data(ticker: str, interval: str, lookback_days: int) -> pd.DataFrame:
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        df = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False)
        
        if df.empty:
            print(f"❌ No data for {ticker} ({interval})")
            return pd.DataFrame()
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
            
        df.rename(columns={'Open': 'Open', 'High': 'High', 'Low': 'Low', 'Close': 'Close', 'Volume': 'Volume'}, inplace=True)
        
        # QC 1: Check if data is recent (June 2026)
        last_candle_time = df.index[-1]
        if isinstance(last_candle_time, pd.Timestamp):
            time_diff = datetime.now(last_candle_time.tzinfo) - last_candle_time
            if time_diff.total_seconds() > 172800:  # 48 hours
                print(f"️ WARNING: Data for {ticker} seems stale. Last candle: {last_candle_time}. Skipping.")
                return pd.DataFrame()
        
        # QC 2: Check for price anomalies (e.g., ETH $4400 when market is $1600)
        recent_median = df['Close'].iloc[-10:].median()
        last_close = df['Close'].iloc[-1]
        
        if abs(last_close - recent_median) / recent_median > 0.20:
            print(f"⚠️ WARNING: Price anomaly for {ticker}. Last: {last_close}, Median: {recent_median}. Skipping.")
            return pd.DataFrame()

        return df
        
    except Exception as e:
        print(f"❌ Error fetching {ticker}: {e}")
        return pd.DataFrame()

# ==========================================
# TELEGRAM ALERT
# ==========================================
def send_telegram(message: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {'chat_id': TELEGRAM_CHAT_ID, 'text': message, 'parse_mode': 'HTML'}
    try:
        resp = requests.post(url, json=payload, timeout=10)
        if resp.status_code == 200:
            print(f"✅ Alert sent")
        else:
            print(f"❌ TG Error: {resp.text}")
    except Exception as e:
        print(f"❌ Failed: {e}")

# ==========================================
# BBMA STATE MACHINE (STRICT OMA ALLY)
# ==========================================
class BBMABuyTracker:
    def __init__(self):
        self.state = BBMAState.NONE
        self.extreme_low = None
        self.mhv_confirmed = False
        self.csa_confirmed = False
    
    def reset(self):
        self.state = BBMAState.NONE
        self.extreme_low = None
        self.mhv_confirmed = False
        self.csa_confirmed = False
    
    def update(self, row: pd.Series, prev_row: pd.Series) -> Optional[Dict]:
        close = row['Close']
        open_ = row['Open']
        high = row['High']
        low = row['Low']
        bb_upper = row['bb_upper']
        bb_lower = row['bb_lower']
        bb_mid = row['bb_mid']
        ma5_high = row['ma5_high']
        ma10_high = row['ma10_high']
        ma5_low = row['ma5_low']
        ma10_low = row['ma10_low']
        
        is_bullish = close > open_
        is_bearish = close < open_
        prev_bullish = prev_row['Close'] > prev_row['Open']
        prev_bearish = prev_row['Close'] < prev_row['Open']
        
        # --- 1. CHECK EXTREME BUY ---
        # MA5/10 Low keluar Low BB + Reverse Candle (Bullish after Bearish)
        extreme_buy = (
            (ma5_low < bb_lower or ma10_low < bb_lower) and
            is_bullish and prev_bearish
        )
        
        if extreme_buy:
            self.state = BBMAState.EXTREME_BUY
            self.extreme_low = low
            self.mhv_confirmed = False
            self.csa_confirmed = False
            return None
        
        # --- 2. CHECK MHV AFTER EXTREME BUY ---
        # Price tak close bawah Low BB + Reverse Candle (Bearish)
        if self.state == BBMAState.EXTREME_BUY:
            mhv_valid = (close >= bb_lower) and is_bearish and prev_bullish
            
            if mhv_valid:
                self.state = BBMAState.MHV_BUY
                self.mhv_confirmed = True
                return None
            
            # MHV batal jika close bawah Low BB (momentum sambung)
            if close < bb_lower:
                self.reset()
                return None
        
        # --- 3. CHECK CSA AFTER MHV ---
        # Close atas MA5/10 Low (Early CSA)
        if self.state == BBMAState.MHV_BUY and self.mhv_confirmed:
            csa_early = close > ma5_low and close > ma10_low
            
            if csa_early:
                self.state = BBMAState.CSA_BUY
                self.csa_confirmed = True
                return None
        
        # --- 4. CHECK RE-ENTRY BUY (ONLY AFTER CSA) ---
        # CRITICAL QC FIX: Price mesti touch zone, tapi tak boleh crash jauh
        if self.state == BBMAState.CSA_BUY and self.csa_confirmed:
            # Price tak boleh crash lebih 2% bawah MA5/10 zone
            price_not_crashed = (close >= ma5_low * 0.98 and close >= ma10_low * 0.98)
            
            # Low candle kena touch zone (dalam 1% range)
            touch_zone = (
                (low <= ma5_low * 1.01 and low >= ma5_low * 0.99) or 
                (low <= ma10_low * 1.01 and low >= ma10_low * 0.99)
            )
            
            # Close kena recover balik dekat zone (bullish rejection)
            close_near_zone = close >= ma5_low * 0.99
            
            # Close tak boleh exceed MA5/10 High dan Mid BB (Rules BBMA)
            valid_reentry = (
                price_not_crashed and
                touch_zone and
                close_near_zone and
                close <= ma5_high and
                close <= ma10_high and
                close <= bb_mid and
                is_bullish and prev_bearish
            )
            
            if valid_reentry:
                self.state = BBMAState.REENTRY_BUY
                return {
                    'type': 'BUY',
                    'ma5_low': ma5_low,
                    'ma10_low': ma10_low,
                    'bb_lower': bb_lower,
                    'bb_upper': bb_upper,
                    'ma5_high': ma5_high,
                    'ma10_high': ma10_high,
                    'bb_mid': bb_mid
                }
        
        return None

# ==========================================
# LEVEL CALCULATION
# ==========================================
def calculate_levels_buy(setup: Dict) -> Dict:
    entry_aggressive = setup['ma5_low']
    entry_conservative = setup['ma10_low']
    entry_moderate = (entry_aggressive + entry_conservative) / 2
    
    sl = setup['bb_lower']
    
    tp1 = setup['ma5_high']
    tp2 = setup['bb_upper']
    tp3 = setup['bb_upper'] * 1.02
    
    return {
        'conservative': {'entry': entry_conservative, 'sl': sl},
        'moderate': {'entry': entry_moderate, 'sl': sl},
        'aggressive': {'entry': entry_aggressive, 'sl': sl},
        'tp1': tp1, 'tp2': tp2, 'tp3': tp3
    }

# ==========================================
# SCANNER
# ==========================================
def scan_pair(ticker: str):
    tracker = BBMABuyTracker()
    
    for style, tfs in STYLES.items():
        print(f"Scanning {ticker} ({style})...")
        
        try:
            df_big = fetch_yfinance_data(ticker, tfs['big'], tfs['lookback_days'])
            if df_big.empty:
                continue
            df_big = get_indicators(df_big)
            if df_big.empty:
                continue
                
            last_big = df_big.iloc[-1]
            uptrend = last_big['ema50'] < last_big['bb_mid']
            
            if not uptrend:
                continue
            
            df_small = fetch_yfinance_data(ticker, tfs['small'], tfs['lookback_days'])
            if df_small.empty:
                continue
            df_small = get_indicators(df_small)
            if df_small.empty:
                continue
            
            tracker.reset()
            
            setup_found = None
            for i in range(1, len(df_small)):
                result = tracker.update(df_small.iloc[i], df_small.iloc[i-1])
                if result is not None:
                    setup_found = result
                    break
            
            if setup_found is None:
                continue
            
            levels = calculate_levels_buy(setup_found)
            pair_name = ticker.replace('-USD', '/USDT')
            current_price = df_small.iloc[-1]['Close']
            
            msg = f"""
🟢 <b>BBMA BUY SETUP DETECTED</b>

 Pair: {pair_name}
💰 Current Price: ${current_price:.2f}
⏱️ Style: {style}
📈 Pattern: Bullish Re-Entry (After CSA)
✅ Cycle: Extreme → MHV → CSA → Re-Entry CONFIRMED

━━━━━━━━━━━━━━━━━━━━

🟢 <b>CONSERVATIVE ENTRY</b>
• Entry: {levels['conservative']['entry']:.2f}
• SL: {levels['conservative']['sl']:.2f}
• TP1: {levels['tp1']:.2f} | TP2: {levels['tp2']:.2f} | TP3: {levels['tp3']:.2f}

🟡 <b>MODERATE ENTRY</b>
• Entry: {levels['moderate']['entry']:.2f}
• SL: {levels['moderate']['sl']:.2f}
• TP1: {levels['tp1']:.2f} | TP2: {levels['tp2']:.2f} | TP3: {levels['tp3']:.2f}

🔴 <b>AGGRESSIVE ENTRY</b>
• Entry: {levels['aggressive']['entry']:.2f}
• SL: {levels['aggressive']['sl']:.2f}
• TP1: {levels['tp1']:.2f} | TP2: {levels['tp2']:.2f} | TP3: {levels['tp3']:.2f}

━━━━━━━━━━━━━━━━━━━━

⚠️ <i>Verify live price on exchange. Setup based on data from {datetime.now().strftime('%Y-%m-%d %H:%M')}</i>
            """
            send_telegram(msg)
            print(f"🚨 BUY SETUP FOUND: {ticker} ({style}) @ ${current_price:.2f}")
                
        except Exception as e:
            print(f"❌ Error {ticker} {style}: {e}")

# ==========================================
# MAIN
# ==========================================
def main():
    print(f"=== BBMA BUY Scanner Start: {datetime.now()} ===")
    for ticker in PAIRS:
        try:
            scan_pair(ticker)
        except Exception as e:
            print(f"❌ Error {ticker}: {e}")
    print("=== Scan Complete ===")

if __name__ == "__main__":
    main()

