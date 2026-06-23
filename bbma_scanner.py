import os
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from typing import Dict, Optional
from enum import Enum

# ==========================================
# CONFIGURATION
# ==========================================
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')

if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
    raise ValueError("Missing Telegram credentials in GitHub Secrets!")

PAIRS = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD', 'XRP-USD', 'RENDER-USD']

STYLES = {
    'Intraday': {'big': '4h', 'small': '1h', 'period_big': '300d', 'period_small': '300d'},
    'Swing': {'big': '1d', 'small': '4h', 'period_big': '730d', 'period_small': '730d'}
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
    df.columns = [col.capitalize() for col in df.columns]
    df = df.copy()
    
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
# BBMA STATE MACHINE (BUY ONLY)
# ==========================================
class BBMABuyTracker:
    """
    Tracks BBMA BUY cycle sequentially per Oma Ally:
    Extreme Buy → TPW → MHV → CSA → Re-Entry Buy
    """
    
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
        """
        Process one candle sequentially and return setup dict if valid Re-Entry Buy detected.
        Returns None otherwise.
        """
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
        
        # --- CHECK EXTREME BUY ---
        # MA5/10 Low keluar dari Low BB + Reverse Candle (Bullish after Bearish)
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
        
        # --- CHECK MHV AFTER EXTREME BUY ---
        if self.state == BBMAState.EXTREME_BUY:
            # MHV Buy: Price TAK close bawah Low BB + Reverse Candle (Bearish)
            # Price gagal teruskan momentum turun
            mhv_valid = (close >= bb_lower) and is_bearish and prev_bullish
            
            if mhv_valid:
                self.state = BBMAState.MHV_BUY
                self.mhv_confirmed = True
                return None
            
            # MHV batal jika close bawah Low BB (momentum sambung)
            if close < bb_lower:
                self.reset()
                return None
        
        # --- CHECK CSA AFTER MHV ---
        if self.state == BBMAState.MHV_BUY and self.mhv_confirmed:
            # CSA Buy: Close atas MA5/10 Low (early) atau atas Mid BB (strong)
            csa_early = close > ma5_low and close > ma10_low
            csa_strong = csa_early and close > bb_mid
            
            if csa_early:
                self.state = BBMAState.CSA_BUY
                self.csa_confirmed = True
                return None
        
        # --- CHECK RE-ENTRY BUY (ONLY AFTER CSA) ---
        if self.state == BBMAState.CSA_BUY and self.csa_confirmed:
            # Re-Entry Buy: Price retrace ke MA5/10 Low zone
            # Close TIDAK boleh exceed MA5/10 High atau Mid BB
            in_zone = (low <= ma5_low * 1.003) or (low <= ma10_low * 1.003)
            valid_reentry = (
                in_zone and
                close <= ma5_high and
                close <= ma10_high and
                close <= bb_mid and
                is_bullish and prev_bearish  # Reverse candle confirmation
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
# LEVEL CALCULATION (STRICT BBMA)
# ==========================================
def calculate_levels_buy(setup: Dict) -> Dict:
    """
    BUY Levels (Oma Ally):
    - Entry: MA5 Low (aggressive) / MA10 Low (conservative)
    - SL: Below BB Lower (bukan percentage)
    - TP1: MA5/10 High | TP2: BB Upper | TP3: BB Upper + buffer
    """
    entry_aggressive = setup['ma5_low']
    entry_conservative = setup['ma10_low']
    entry_moderate = (entry_aggressive + entry_conservative) / 2
    
    sl = setup['bb_lower']  # SL strictly below BB Lower
    
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
# SCANNER WITH STATE TRACKING
# ==========================================
def scan_pair(ticker: str):
    tracker = BBMABuyTracker()
    
    for style, tfs in STYLES.items():
        print(f"Scanning {ticker} ({style})...")
        
        try:
            # Fetch Big TF data
            df_big = yf.download(ticker, interval=tfs['big'], period=tfs['period_big'], progress=False)
            if df_big.empty or len(df_big) < 60:
                continue
            if isinstance(df_big.columns, pd.MultiIndex):
                df_big.columns = df_big.columns.get_level_values(0)
            df_big = get_indicators(df_big)
            
            # Check Uptrend on Big TF (EMA50 below Mid BB)
            last_big = df_big.iloc[-1]
            uptrend = last_big['ema50'] < last_big['bb_mid']
            
            if not uptrend:
                print(f"ℹ️ {ticker} ({style}) - No uptrend on big TF")
                continue
            
            # Fetch Small TF data
            df_small = yf.download(ticker, interval=tfs['small'], period=tfs['period_small'], progress=False)
            if df_small.empty or len(df_small) < 60:
                continue
            if isinstance(df_small.columns, pd.MultiIndex):
                df_small.columns = df_small.columns.get_level_values(0)
            df_small = get_indicators(df_small)
            
            # Reset tracker for this pair/style
            tracker.reset()
            
            # Walk through small TF candles sequentially to track BBMA cycle
            setup_found = None
            for i in range(1, len(df_small)):
                result = tracker.update(df_small.iloc[i], df_small.iloc[i-1])
                if result is not None:
                    setup_found = result
                    break  # First valid setup in current cycle
            
            if setup_found is None:
                print(f"ℹ️ {ticker} ({style}) - No valid BBMA BUY setup (cycle incomplete)")
                continue
            
            # Calculate levels and send alert
            levels = calculate_levels_buy(setup_found)
            pair_name = ticker.replace('-USD', '/USDT')
            
            msg = f"""
🟢 <b>BBMA BUY SETUP DETECTED</b>

📊 Pair: {pair_name}
⏱️ Style: {style}
📈 Pattern: Bullish Re-Entry (After CSA)
✅ Cycle: Extreme → MHV → CSA → Re-Entry CONFIRMED

━━━━━━━━━━━━━━━━━━━━

🟢 <b>CONSERVATIVE ENTRY</b>
Paling selamat, tunggu confirmation penuh
• Entry: {levels['conservative']['entry']:.4f}
• SL: {levels['conservative']['sl']:.4f}
• TP1: {levels['tp1']:.4f} | TP2: {levels['tp2']:.4f} | TP3: {levels['tp3']:.4f}

🟡 <b>MODERATE ENTRY</b>
Balance risk & reward
• Entry: {levels['moderate']['entry']:.4f}
• SL: {levels['moderate']['sl']:.4f}
• TP1: {levels['tp1']:.4f} | TP2: {levels['tp2']:.4f} | TP3: {levels['tp3']:.4f}

🔴 <b>AGGRESSIVE ENTRY</b>
Entry awal, harga terbaik, risiko tinggi
• Entry: {levels['aggressive']['entry']:.4f}
• SL: {levels['aggressive']['sl']:.4f}
• TP1: {levels['tp1']:.4f} | TP2: {levels['tp2']:.4f} | TP3: {levels['tp3']:.4f}

━━━━━━━━━━━━━━━━━━━━

⚠️ <i>Pilih 1 level je ikut risk appetite kau! Verify live price on exchange.</i>
⏰ {datetime.now().strftime('%Y-%m-%d %H:%M')}
            """
            send_telegram(msg)
            print(f"🚨 BUY SETUP FOUND: {ticker} ({style})")
                
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

