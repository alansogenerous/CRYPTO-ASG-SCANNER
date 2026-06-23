import os
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from typing import Dict, Optional

# ==========================================
# CONFIGURATION (Read from GitHub Secrets / Env Vars)
# ==========================================
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')

if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
    raise ValueError("Missing Telegram credentials in Environment Variables!")

PAIRS = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD', 'XRP-USD', 'RENDER-USD']

# ONLY INTRADAY & SWING (Scalping removed)
STYLES = {
    'Intraday': {'big': '4h', 'small': '1h', 'period_big': '300d', 'period_small': '300d'},
    'Swing': {'big': '1d', 'small': '4h', 'period_big': '730d', 'period_small': '730d'}
}

BB_PERIOD = 20
BB_STD = 2.0

# ==========================================
# INDICATOR CALCULATIONS (Strict BBMA PDF)
# ==========================================
def calculate_lwma(series: pd.Series, period: int) -> pd.Series:
    weights = np.arange(1, period + 1)
    def lwma(window):
        return np.sum(window * weights) / np.sum(weights)
    return series.rolling(window=period).apply(lwma, raw=True)

def get_indicators(df: pd.DataFrame) -> pd.DataFrame:
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
        print(f" Failed: {e}")

# ==========================================
# BBMA SCANNER LOGIC
# ==========================================
def check_uptrend(df: pd.DataFrame) -> bool:
    # PDF Spec: EMA 50 below mid BB = Uptrend
    last = df.iloc[-1]
    return last['ema50'] < last['bb_mid']

def find_reentry_buy(df: pd.DataFrame) -> Optional[Dict]:
    if len(df) < 3:
        return None
    
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    
    # Zone: Low touched MA5 Low or MA10 Low (0.3% buffer for yfinance noise)
    touch_zone = (curr['Low'] <= curr['ma5_low'] * 1.003) or \
                 (curr['Low'] <= curr['ma10_low'] * 1.003)
    
    # Validity: Close >= Low BB (Strict PDF rule: "CS tidak boleh CLOSE di luar BB")
    valid_close = curr['Close'] >= curr['bb_lower']
    
    # Trigger: Reverse Candle (Hijau selepas Merah)
    is_bullish = curr['Close'] > curr['Open']
    prev_bearish = prev['Close'] < prev['Open']
    reverse = is_bullish and prev_bearish
    
    if touch_zone and valid_close and reverse:
        return {
            'ma5_low': curr['ma5_low'],
            'ma10_low': curr['ma10_low'],
            'bb_lower': curr['bb_lower'],
            'bb_upper': curr['bb_upper'],
            'ma5_high': curr['ma5_high'],
            'ma10_high': curr['ma10_high'],
            'bb_mid': curr['bb_mid']
        }
    return None

def calculate_levels(setup: Dict) -> Dict:
    # 3 Entry Prices based on zone depth
    entry_high_risk = setup['ma5_low']  # Aggressive
    entry_mid_risk = (setup['ma5_low'] + setup['ma10_low']) / 2  # Moderate
    entry_low_risk = setup['ma10_low']  # Conservative
    
    # SL: Strict BBMA Rule - Must be below Low BB
    sl_base = setup['bb_lower']
    sl_high_risk = sl_base * 0.999
    sl_mid_risk = sl_base * 0.998
    sl_low_risk = sl_base * 0.997
    
    # TPs
    tp1 = setup['ma5_high']  # TP1: MA5 High
    tp2 = setup['bb_upper']  # TP2: Top BB
    tp3 = setup['bb_upper'] * 1.02  # TP3: Extension
    
    return {
        'high_risk': {'entry': entry_high_risk, 'sl': sl_high_risk},
        'mid_risk': {'entry': entry_mid_risk, 'sl': sl_mid_risk},
        'low_risk': {'entry': entry_low_risk, 'sl': sl_low_risk},
        'tp1': tp1, 'tp2': tp2, 'tp3': tp3
    }

def scan_pair(ticker: str):
    for style, tfs in STYLES.items():
        print(f"Scanning {ticker} ({style})...")
        
        try:
            df_big = yf.download(ticker, interval=tfs['big'], period=tfs['period_big'], progress=False)
            if df_big.empty or len(df_big) < 60: continue
            if isinstance(df_big.columns, pd.MultiIndex): df_big.columns = df_big.columns.get_level_values(0)
            df_big = get_indicators(df_big)
            
            if not check_uptrend(df_big): continue
            
            df_small = yf.download(ticker, interval=tfs['small'], period=tfs['period_small'], progress=False)
            if df_small.empty or len(df_small) < 60: continue
            if isinstance(df_small.columns, pd.MultiIndex): df_small.columns = df_small.columns.get_level_values(0)
            df_small = get_indicators(df_small)
            
            setup = find_reentry_buy(df_small)
            if not setup: continue
            
            levels = calculate_levels(setup)
            pair_name = ticker.replace('-USD', '/USDT')
            
            msg = f"""
🚨 <b>BBMA BUY SETUP DETECTED</b>

📊 Pair: {pair_name}
⏱️ Style: {style}
 Pattern: Bullish Rejection (Pinbar)

━━━━━━━━━━━━━━━━━━━━

🟢 <b>LOW RISK ENTRY (Konservatif)</b>
Paling selamat, tunggu confirmation penuh
• Entry: {levels['low_risk']['entry']:.4f}
• SL: {levels['low_risk']['sl']:.4f}
• TP1: {levels['tp1']:.4f} | TP2: {levels['tp2']:.4f} | TP3: {levels['tp3']:.4f}

🟡 <b>MID RISK ENTRY (Moderate)</b>
Balance risk & reward
• Entry: {levels['mid_risk']['entry']:.4f}
• SL: {levels['mid_risk']['sl']:.4f}
• TP1: {levels['tp1']:.4f} | TP2: {levels['tp2']:.4f} | TP3: {levels['tp3']:.4f}

🔴 <b>HIGH RISK ENTRY (Agresif)</b>
Entry awal, harga terbaik, risiko tinggi
• Entry: {levels['high_risk']['entry']:.4f}
• SL: {levels['high_risk']['sl']:.4f}
• TP1: {levels['tp1']:.4f} | TP2: {levels['tp2']:.4f} | TP3: {levels['tp3']:.4f}

━━━━━━━━━━━━━━━━━━━━

⚠️ <i>Pilih 1 level je ikut risk appetite kau! Verify live price on exchange.</i>
⏰ {datetime.now().strftime('%Y-%m-%d %H:%M')}
            """
            send_telegram(msg)
            print(f"🚨 SETUP FOUND: {ticker} ({style})")
        except Exception as e:
            print(f"❌ Error {ticker} {style}: {e}")

# ==========================================
# MAIN
# ==========================================
def main():
    print(f"=== BBMA Scan Start: {datetime.now()} ===")
    for ticker in PAIRS:
        try:
            scan_pair(ticker)
        except Exception as e:
            print(f"❌ Error {ticker}: {e}")
    print("=== Scan Complete ===")

if __name__ == "__main__":
    main()

