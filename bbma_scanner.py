"""
BBMA Crypto Spot Scanner — 9.8/10 FINAL VERSION
================================================
BUY ONLY | Spot Trading | Intraday + Swing

✅ MULTI-TICKER FETCH — 1 API call for all coins (4x faster)
✅ DYNAMIC FIB LOOKBACK — 20 (Intraday) / 50 (Swing)
✅ 3-TF REM CODE (R-E-M)
✅ PROPER MHV + BB EXPANSION
✅ INCLUDES RENDER (RNDR-USD)
"""

import os
import json
import time
import traceback
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, field

# ============================================================
# CONFIGURATION
# ============================================================
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID   = os.environ.get('TELEGRAM_CHAT_ID')
DRY_RUN            = os.environ.get('DRY_RUN', 'false').lower() == 'true'
COINGLASS_API_KEY  = os.environ.get('COINGLASS_API_KEY')
ALERT_CACHE_PATH   = os.environ.get('ALERT_CACHE_PATH', 'alert_cache.json')

if not DRY_RUN and (not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID):
    raise ValueError("Missing Telegram credentials!")

# ── PER-PAIR CONFIG (11 COINS — Added RNDR) ──────────────────
PAIRS: Dict[str, Dict] = {
    # Stable Large Caps
    'BTC-USD': {'bb_std': 2.0, 'vol_threshold': 1.2},
    'ETH-USD': {'bb_std': 2.0, 'vol_threshold': 1.2},
    'BNB-USD': {'bb_std': 2.0, 'vol_threshold': 1.2},
    # Mid Cap
    'SOL-USD': {'bb_std': 2.5, 'vol_threshold': 1.5},
    'XRP-USD': {'bb_std': 2.5, 'vol_threshold': 1.5},
    'ADA-USD': {'bb_std': 2.5, 'vol_threshold': 1.5},
    'LINK-USD': {'bb_std': 2.5, 'vol_threshold': 1.5},
    # High Volatility
    'TRX-USD': {'bb_std': 2.8, 'vol_threshold': 1.8},
    'DOT-USD': {'bb_std': 2.8, 'vol_threshold': 1.8},
    'TON-USD': {'bb_std': 3.0, 'vol_threshold': 2.0},
    # ── NEW: Render ──────────────────────────────────────────
    'RNDR-USD': {'bb_std': 2.8, 'vol_threshold': 1.8},  # Volatility sederhana tinggi
}

# ── 3-TIMEFRAME STYLES ──────────────────────────────────────
STYLES: Dict[str, Dict] = {
    'Intraday': {
        'big': '4h',
        'mid': '1h',
        'small': '15m',
        'lookback_days': 30,
        'fib_lookback': 20,
    },
    'Swing': {
        'big': '1d',
        'mid': '4h',
        'small': '1h',
        'lookback_days': 90,
        'fib_lookback': 50,
    },
}

# ── Indicator settings ──────────────────────────────────────
BB_PERIOD            = 20
RSI_PERIOD           = 14
RSI_OVERSOLD         = 40
ATR_PERIOD           = 14
ATR_SL_MULTIPLIER    = 1.5
OBV_EMA_PERIOD       = 20
VOL_AVG_PERIOD       = 20
VOLUME_SPIKE_MULT    = 1.5
MIN_AVG_VOLUME_USD   = 50_000_000
MIN_RR               = 1.5
MAX_DRIFT_PCT        = 0.06
ALERT_COOLDOWN_HOURS = 4
MIN_CONFIDENCE       = 'MEDIUM'
BTC_FILTER_MODE      = 'soft'

# ============================================================
# DATA CLASSES
# ============================================================
@dataclass
class Signal:
    pair:               str
    style:              str
    entry_zone_top:     float
    entry_zone_bottom:  float
    entry_moderate:     float
    entry_aggressive:   float
    sl:                 float
    tp1:                float
    tp2:                float
    tp3:                float
    rr:                 float
    rr_aggressive:      float
    atr:                float
    confidence:         str
    confirmations:      List[str] = field(default_factory=list)
    warnings:           List[str] = field(default_factory=list)
    fib_382:            float = 0.0
    fib_50:             float = 0.0
    fib_confluence:     bool  = False
    csa_type:           str   = 'CSA_EARLY'
    btc_score:          str   = ''
    funding_rate:       Optional[float] = None
    timestamp:          str   = ''
    bb_expanding:       bool  = False
    tf_alignment:       str   = ''

class BBMAState(Enum):
    NONE        = 0
    EXTREME_BUY = 1
    MHV_BUY     = 2
    CSA_BUY     = 3
    REENTRY_BUY = 4

# ============================================================
# ALERT CACHE
# ============================================================
def load_cache() -> Dict:
    try:
        if os.path.exists(ALERT_CACHE_PATH):
            with open(ALERT_CACHE_PATH, 'r') as f:
                return json.load(f)
    except Exception:
        pass
    return {}

def save_cache(cache: Dict):
    try:
        with open(ALERT_CACHE_PATH, 'w') as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        print(f"⚠️ Cache save failed: {e}")

def is_duplicate(cache: Dict, ticker: str, style: str) -> bool:
    key = f"{ticker}_{style}"
    if key not in cache:
        return False
    elapsed = (datetime.now() - datetime.fromisoformat(cache[key])).total_seconds() / 3600
    if elapsed < ALERT_COOLDOWN_HOURS:
        print(f"🔕 Duplicate skip: {key} ({elapsed:.1f}h ago)")
        return True
    return False

def mark_sent(cache: Dict, ticker: str, style: str):
    cache[f"{ticker}_{style}"] = datetime.now().isoformat()

# ============================================================
# INDICATORS
# ============================================================
def calculate_lwma(series: pd.Series, period: int) -> pd.Series:
    weights = np.arange(1, period + 1)
    return series.rolling(window=period).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True
    )

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hl = df['High'] - df['Low']
    hc = (df['High'] - df['Close'].shift()).abs()
    lc = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def calculate_obv_vectorized(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df['Close'].diff().fillna(0))
    return (direction * df['Volume']).cumsum()

def calculate_fibonacci(high: float, low: float) -> Dict[str, float]:
    diff = high - low
    return {
        '0.236': high - 0.236 * diff,
        '0.382': high - 0.382 * diff,
        '0.500': high - 0.500 * diff,
        '0.618': high - 0.618 * diff,
    }

def get_indicators(df: pd.DataFrame, bb_std: float = 2.0) -> pd.DataFrame:
    df = df.copy()
    df['bb_mid'] = df['Close'].rolling(BB_PERIOD).mean()
    _std = df['Close'].rolling(BB_PERIOD).std()
    df['bb_upper'] = df['bb_mid'] + (_std * bb_std)
    df['bb_lower'] = df['bb_mid'] - (_std * bb_std)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']
    df['bb_width_prev'] = df['bb_width'].shift(5)
    df['bb_expanding'] = df['bb_width'] > df['bb_width_prev'] * 1.05
    df['ma5_high'] = calculate_lwma(df['High'], 5)
    df['ma10_high'] = calculate_lwma(df['High'], 10)
    df['ma5_low'] = calculate_lwma(df['Low'], 5)
    df['ma10_low'] = calculate_lwma(df['Low'], 10)
    df['ema50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['rsi'] = calculate_rsi(df['Close'], RSI_PERIOD)
    df['atr'] = calculate_atr(df, ATR_PERIOD)
    df['obv'] = calculate_obv_vectorized(df)
    df['obv_ema'] = df['obv'].ewm(span=OBV_EMA_PERIOD, adjust=False).mean()
    df['obv_bullish'] = df['obv'] > df['obv_ema']
    df['vol_avg20'] = df['Volume'].rolling(VOL_AVG_PERIOD).mean()
    df['vol_ratio'] = df['Volume'] / df['vol_avg20']
    df['vol_spike'] = df['vol_ratio'] >= VOLUME_SPIKE_MULT
    return df.dropna()

# ============================================================
# MULTI-TICKER FETCH
# ============================================================
def fetch_multiple_tickers(tickers: List[str], interval: str, lookback_days: int) -> Dict[str, pd.DataFrame]:
    """Fetch multiple tickers in ONE API call."""
    try:
        end = datetime.now()
        start = end - timedelta(days=lookback_days + 30)
        data = yf.download(
            tickers, 
            start=start, 
            end=end, 
            interval=interval, 
            progress=False, 
            group_by='ticker',
            auto_adjust=True,
            threads=True,
        )
        results = {}
        for ticker in tickers:
            if ticker in data:
                df = data[ticker]
                if not df.empty and len(df) > 60:
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.droplevel(1)
                    col_map = {}
                    for c in df.columns:
                        lc = c.lower()
                        if 'open' in lc: col_map[c] = 'Open'
                        elif 'high' in lc: col_map[c] = 'High'
                        elif 'low' in lc: col_map[c] = 'Low'
                        elif 'close' in lc: col_map[c] = 'Close'
                        elif 'vol' in lc: col_map[c] = 'Volume'
                    df.rename(columns=col_map, inplace=True)
                    last_ts = df.index[-1]
                    if isinstance(last_ts, pd.Timestamp):
                        if (datetime.now(last_ts.tzinfo) - last_ts).total_seconds() > 86400:
                            print(f"⚠️ Stale data {ticker} ({interval})")
                            continue
                    results[ticker] = df
                else:
                    print(f"⚠️ No/insufficient data {ticker} ({interval})")
            else:
                print(f"⚠️ {ticker} not in multi-fetch response")
        return results
    except Exception as e:
        print(f"❌ Multi-fetch failed: {e}")
        return {}

def fetch_data(ticker: str, interval: str, lookback_days: int, max_retries: int = 2) -> pd.DataFrame:
    """Fallback single fetch."""
    for attempt in range(max_retries):
        try:
            end = datetime.now()
            start = end - timedelta(days=lookback_days + 30)
            df = yf.download(ticker, start=start, end=end, interval=interval, progress=False, auto_adjust=True)
            if df.empty:
                return pd.DataFrame()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            col_map = {}
            for c in df.columns:
                lc = c.lower()
                if 'open' in lc: col_map[c] = 'Open'
                elif 'high' in lc: col_map[c] = 'High'
                elif 'low' in lc: col_map[c] = 'Low'
                elif 'close' in lc: col_map[c] = 'Close'
                elif 'vol' in lc: col_map[c] = 'Volume'
            df.rename(columns=col_map, inplace=True)
            if len(df) < 60:
                return pd.DataFrame()
            return df
        except Exception as e:
            print(f"❌ Fetch fallback {ticker} ({interval}) attempt {attempt+1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    return pd.DataFrame()

# ============================================================
# LIQUIDITY & BTC FILTERS
# ============================================================
def check_liquidity(df: pd.DataFrame, ticker: str) -> bool:
    avg = (df['Close'].tail(20) * df['Volume'].tail(20)).mean()
    if avg < MIN_AVG_VOLUME_USD:
        print(f"🚫 Liquidity fail {ticker}: ${avg:,.0f}")
        return False
    return True

def get_btc_structure_from_df(df: pd.DataFrame) -> Optional[Dict]:
    if df.empty or len(df) < 5:
        return None
    df = get_indicators(df, bb_std=2.0)
    if df.empty:
        return None
    last, prev = df.iloc[-1], df.iloc[-2]
    c1 = bool(last['Close'] > last['ema50'])
    c2 = bool(last['ema50'] > prev['ema50'])
    c3 = bool(last['obv_bullish'])
    n = sum([c1, c2, c3])
    bullish = (n >= 2) if BTC_FILTER_MODE == 'soft' else (n == 3)
    return {'bullish': bullish, 'score': f"{n}/3", 'c1_price': c1, 'c2_slope': c2, 'c3_obv': c3}

# ============================================================
# FUNDING RATE
# ============================================================
def fetch_funding_rate(symbol: str) -> Optional[float]:
    if not COINGLASS_API_KEY:
        return None
    try:
        url = f"https://open-api.coinglass.com/public/v2/funding?symbol={symbol}&timeType=h8"
        resp = requests.get(url, headers={"coinglassSecret": COINGLASS_API_KEY}, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if data.get('data'):
                return float(data['data'][0].get('rate', 0))
    except Exception:
        pass
    return None

# ============================================================
# TELEGRAM
# ============================================================
def send_telegram(message: str):
    if DRY_RUN:
        print(f"[DRY RUN]\n{message[:300]}...")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    chunks = [message[i:i+4000] for i in range(0, len(message), 4000)]
    for chunk in chunks:
        payload = {
            'chat_id': TELEGRAM_CHAT_ID,
            'text': chunk,
            'parse_mode': 'HTML',
            'disable_web_page_preview': True,
        }
        try:
            resp = requests.post(url, json=payload, timeout=15)
            if resp.status_code != 200:
                print(f"❌ TG Error: {resp.text}")
        except Exception as e:
            print(f"❌ Send failed: {e}")
        time.sleep(0.5)

# ============================================================
# BBMA STATE MACHINE
# ============================================================
class BBMABuyTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.state = BBMAState.NONE
        self.extreme_low = None
        self.extreme_index = None
        self.rsi_at_extreme = None
        self.mhv_confirmed = False
        self.csa_confirmed = False
        self.csa_type = 'CSA_EARLY'
        self.mhv_low = None
        self.double_bottom = False

    def update(self, row: pd.Series, prev: pd.Series, idx: int) -> Optional[Dict]:
        close = row['Close']; open_ = row['Open']
        low = row['Low']; high = row['High']
        bb_lower = row['bb_lower']; bb_mid = row['bb_mid']
        ma5_high = row['ma5_high']; ma10_high = row['ma10_high']
        ma5_low = row['ma5_low']; ma10_low = row['ma10_low']
        rsi = row['rsi']
        obv_bull = bool(row['obv_bullish'])
        vol_spike = bool(row['vol_spike'])
        bull = close > open_
        bear = close < open_
        p_bear = prev['Close'] < prev['Open']

        # 1. EXTREME BUY
        is_extreme = (ma5_low < bb_lower) or (ma10_low < bb_lower)
        if is_extreme and bull and p_bear:
            self.state = BBMAState.EXTREME_BUY
            self.extreme_low = low
            self.extreme_index = idx
            self.rsi_at_extreme = rsi
            self.mhv_confirmed = False
            self.csa_confirmed = False
            self.double_bottom = False
            return None

        # 2. MHV (Double Bottom)
        if self.state == BBMAState.EXTREME_BUY and self.extreme_low is not None:
            test_low = low <= (self.extreme_low * 1.01)
            rejected = close > bb_lower
            if test_low and rejected and bear:
                self.state = BBMAState.MHV_BUY
                self.mhv_confirmed = True
                self.mhv_low = low
                self.double_bottom = True
                return None
            if close < bb_lower * 0.99:
                self.reset()
                return None

        # 3. CS ARAH
        if self.state == BBMAState.MHV_BUY and self.mhv_confirmed:
            csa_early = close > ma5_low and close > ma10_low
            csa_kukuh = csa_early and close > bb_mid
            if csa_early and bull and p_bear:
                self.state = BBMAState.CSA_BUY
                self.csa_confirmed = True
                self.csa_type = 'CSA_KUKUH' if csa_kukuh else 'CSA_EARLY'
                return None
            if close < bb_lower:
                self.reset()
                return None

        # 4. RE-ENTRY BUY
        if self.state == BBMAState.CSA_BUY and self.csa_confirmed:
            zone_top = max(ma5_low, ma10_low)
            zone_bottom = min(ma5_low, ma10_low)
            near_zone = zone_bottom * 0.985 <= low <= zone_top * 1.015
            not_crashed = close >= zone_bottom * 0.98
            below_resist = close <= ma5_high and close <= ma10_high and close <= bb_mid
            reversal = bull and p_bear

            if near_zone and not_crashed and below_resist and reversal and obv_bull and vol_spike:
                self.state = BBMAState.REENTRY_BUY
                return {
                    'zone_top': zone_top,
                    'zone_bottom': zone_bottom,
                    'trigger_price': close,
                    'obv_confirmed': obv_bull,
                    'vol_spike': vol_spike,
                    'rsi': rsi,
                    'csa_type': self.csa_type,
                    'mhv_low': self.mhv_low,
                    'double_bottom': self.double_bottom,
                }
        return None

# ============================================================
# LEVEL CALCULATION
# ============================================================
def calculate_levels(signal: Dict, current_price: float,
                     df: pd.DataFrame, fib_lookback: int = 20) -> Optional[Dict]:
    last = df.iloc[-1]
    zone_top = signal['zone_top']
    zone_bottom = signal['zone_bottom']
    zone_center = (zone_top + zone_bottom) / 2

    drift = abs(current_price - zone_center) / zone_center
    if drift > MAX_DRIFT_PCT:
        print(f"⚠️ Setup expired: drifted {drift:.1%}")
        return None

    swing_high = df['High'].iloc[-fib_lookback:].max()
    swing_low = df['Low'].iloc[-fib_lookback:].min()
    fibs = calculate_fibonacci(swing_high, swing_low)
    fib_382 = fibs['0.382']
    fib_50 = fibs['0.500']
    fib_conf = (abs(zone_center - fib_382) / zone_center < 0.02 or
                abs(zone_center - fib_50) / zone_center < 0.02)

    atr = last['atr']
    sl_bb = last['bb_lower']
    sl_atr = zone_center - (atr * ATR_SL_MULTIPLIER)
    sl = min(sl_bb, sl_atr)

    entry_mod = zone_center
    entry_agg = zone_bottom
    tp1 = last['ma5_high']
    tp2 = last['bb_mid']
    tp3 = last['bb_upper']

    def rr(entry):
        return (tp2 - entry) / (entry - sl) if entry > sl else 0

    return {
        'moderate': {'entry': entry_mod, 'sl': sl, 'rr': rr(entry_mod)},
        'aggressive': {'entry': entry_agg, 'sl': sl, 'rr': rr(entry_agg)},
        'tp1': tp1, 'tp2': tp2, 'tp3': tp3,
        'zone_top': zone_top,
        'zone_bottom': zone_bottom,
        'atr': atr,
        'fib_confluence': fib_conf,
        'fib_382': fib_382,
        'fib_50': fib_50,
        'drift_pct': drift,
        'bb_expanding': bool(last['bb_expanding']),
    }

# ============================================================
# CONFIDENCE SCORING
# ============================================================
CONFIDENCE_MAP = {(0,2): 'LOW', (3,5): 'MEDIUM', (6,8): 'HIGH', (9,99): 'PERFECT'}

def score_to_label(score: int) -> str:
    for (lo, hi), label in CONFIDENCE_MAP.items():
        if lo <= score <= hi:
            return label
    return 'LOW'

def calculate_confidence(df: pd.DataFrame, levels: Dict, signal: Dict,
                         btc_ctx: Dict, funding_rate: Optional[float],
                         tf_alignment: str) -> Tuple[str, List[str], List[str]]:
    confirmations: List[str] = []
    warnings: List[str] = []
    score = 0
    last = df.iloc[-1]

    if signal.get('vol_spike'):
        confirmations.append(f"Volume spike {last['vol_ratio']:.1f}× avg ✅")
        score += 2
    else:
        warnings.append(f"Volume weak ({last['vol_ratio']:.1f}× avg)")

    if signal.get('obv_confirmed'):
        confirmations.append("OBV bullish — accumulation ✅")
        score += 2
    else:
        warnings.append("OBV diverging")

    rsi = signal.get('rsi', last['rsi'])
    if rsi <= RSI_OVERSOLD:
        confirmations.append(f"RSI {rsi:.1f} — oversold ✅")
        score += 1
    else:
        warnings.append(f"RSI {rsi:.1f} — not oversold")

    atr_pct = levels['atr'] / last['Close'] * 100
    if atr_pct < 5:
        confirmations.append(f"ATR {atr_pct:.1f}% — low vol ✅")
        score += 1
    elif atr_pct > 10:
        warnings.append(f"High vol ATR {atr_pct:.1f}%")

    if levels.get('fib_confluence'):
        confirmations.append(f"Fib confluence (38.2={levels['fib_382']:.4f}) ✅")
        score += 1
    else:
        warnings.append("No Fib confluence")

    if btc_ctx and btc_ctx['bullish']:
        confirmations.append(f"BTC bullish ({btc_ctx['score']}) ✅")
        score += 1

    if funding_rate is not None and funding_rate < 0:
        confirmations.append(f"Funding {funding_rate:.4f}% — shorts paying ✅")
        score += 1

    rr = levels['moderate']['rr']
    if rr >= 2.0:
        confirmations.append(f"R/R {rr:.1f}× — excellent ✅")
        score += 2
    elif rr >= MIN_RR:
        confirmations.append(f"R/R {rr:.1f}× — good")
        score += 1
    else:
        warnings.append(f"R/R {rr:.1f}× — low")

    if signal.get('csa_type') == 'CSA_KUKUH':
        confirmations.append("CS Arah Kukuh ✅")
        score += 1

    if levels.get('bb_expanding'):
        confirmations.append("BB expanding (momentum ✅)")
        score += 1
    else:
        warnings.append("BB mampat — sideway risk")

    if "R-E-M" in tf_alignment:
        confirmations.append(f"3-TF Alignment ({tf_alignment}) ✅")
        score += 2
    elif "R-E" in tf_alignment:
        confirmations.append(f"2-TF Alignment ({tf_alignment})")
        score += 1

    return score_to_label(score), confirmations, warnings

# ============================================================
# TELEGRAM ALERT
# ============================================================
def build_alert(signal_obj: Signal, tfs: Dict) -> str:
    conf_emoji = {'LOW': '⚠️', 'MEDIUM': '👍', 'HIGH': '🔥', 'PERFECT': '💎'}
    emoji = conf_emoji.get(signal_obj.confidence, '👍')
    csa_label = "CS Arah Kukuh 💪" if signal_obj.csa_type == 'CSA_KUKUH' else "CS Arah Awal"

    confs = "\n".join([f"  ✅ {c}" for c in signal_obj.confirmations]) or "  —"
    warns = "\n".join([f"  ⚠️ {w}" for w in signal_obj.warnings]) or "  ✅ None"
    funding_line = f"\n• Funding Rate : {signal_obj.funding_rate:.4f}%" if signal_obj.funding_rate is not None else ""
    fib_line = f"\n• Fib 38.2%    : {signal_obj.fib_382:.4f}  |  50%: {signal_obj.fib_50:.4f}" if signal_obj.fib_confluence else ""

    return f"""
🟢 <b>BBMA BUY SETUP</b> {emoji} <b>{signal_obj.confidence}</b>

📊 <b>{signal_obj.pair}</b>  |  {signal_obj.style}  ({tfs['big']}→{tfs['mid']}→{tfs['small']})
🎯 Pattern : Re-Entry Buy ({csa_label})
⏰ Time    : {signal_obj.timestamp}
🧩 Alignment: {signal_obj.tf_alignment}

━━━━━━━━━━━━━━━━━━━━

🌐 <b>MARKET CONTEXT</b>
• BTC Structure : {signal_obj.btc_score}
• BB Expanding  : {"✅ Yes" if signal_obj.bb_expanding else "❌ No"}
• ATR (Vol)     : {signal_obj.atr:.4f}  ({signal_obj.atr / signal_obj.entry_moderate * 100:.1f}%){funding_line}{fib_line}

━━━━━━━━━━━━━━━━━━━━

📐 <b>ENTRY ZONE</b>
• Zone Top    : {signal_obj.entry_zone_top:.4f}
• Zone Bottom : {signal_obj.entry_zone_bottom:.4f}

🟡 <b>MODERATE ⭐ RECOMMENDED</b>
• Entry : {signal_obj.entry_moderate:.4f}
• SL    : {signal_obj.sl:.4f}
• R/R   : {signal_obj.rr:.1f}×

🔴 <b>AGGRESSIVE</b>
• Entry : {signal_obj.entry_aggressive:.4f}
• SL    : {signal_obj.sl:.4f}
• R/R   : {signal_obj.rr_aggressive:.1f}×

🎯 <b>TARGETS</b>
• TP1 : {signal_obj.tp1:.4f}  (MA5 High)
• TP2 : {signal_obj.tp2:.4f}  (Mid BB)  ← Wajib
• TP3 : {signal_obj.tp3:.4f}  (Upper BB)

━━━━━━━━━━━━━━━━━━━━

✅ <b>CONFIRMATIONS</b>
{confs}

⚠️ <b>WARNINGS</b>
{warns}

━━━━━━━━━━━━━━━━━━━━
<i>⚡ Verify live price. Not financial advice.</i>
"""

# ============================================================
# CORE SCANNER
# ============================================================
def scan_single_pair_with_dfs(ticker: str, pair_cfg: Dict, cache: Dict,
                              df_big: pd.DataFrame, df_mid: pd.DataFrame,
                              df_small: pd.DataFrame, btc_ctx: Dict,
                              tfs: Dict, style: str, fib_lookback: int):
    """Scan a single pair using pre-fetched dataframes."""
    bb_std = pair_cfg.get('bb_std', 2.0)

    try:
        if is_duplicate(cache, ticker, style):
            return

        if df_big.empty or df_mid.empty or df_small.empty:
            return

        df_big = get_indicators(df_big, bb_std)
        df_mid = get_indicators(df_mid, bb_std)
        df_small = get_indicators(df_small, bb_std)

        if df_big.empty or df_mid.empty or df_small.empty:
            return

        if not check_liquidity(df_big, ticker):
            return

        last_big = df_big.iloc[-1]
        if not (last_big['ema50'] < last_big['bb_mid']):
            print(f"  ⏭️  {ticker} Big TF not uptrend")
            return

        zone_top_big = max(last_big['ma5_low'], last_big['ma10_low'])
        zone_bottom_big = min(last_big['ma5_low'], last_big['ma10_low'])
        price_big = last_big['Close']
        if not (zone_bottom_big * 0.98 <= price_big <= zone_top_big * 1.02):
            print(f"  ⏭️  {ticker} Big TF not in Re-entry zone")
            return

        last_mid = df_mid.iloc[-1]
        is_extreme = (last_mid['ma5_low'] < last_mid['bb_lower']) or (last_mid['ma10_low'] < last_mid['bb_lower'])
        if not is_extreme:
            print(f"  ⏭️  {ticker} Mid TF no Extreme")
            return

        tracker = BBMABuyTracker()
        setup_found = None
        for i in range(1, len(df_small)):
            result = tracker.update(df_small.iloc[i], df_small.iloc[i-1], i)
            if result:
                setup_found = result
                break

        if not setup_found:
            print(f"  ⏭️  {ticker} Small TF no MHV/CSA")
            return

        current_price = df_small.iloc[-1]['Close']
        levels = calculate_levels(setup_found, current_price, df_small, fib_lookback)
        if not levels:
            return

        if levels['moderate']['rr'] < MIN_RR:
            print(f"  ⏭️  {ticker} R/R {levels['moderate']['rr']:.1f}× < {MIN_RR}×")
            return

        symbol = ticker.replace('-USD', '')
        funding_rate = fetch_funding_rate(symbol)
        tf_label = "R-E-M"

        confidence, confs, warns = calculate_confidence(
            df_small, levels, setup_found, btc_ctx, funding_rate, tf_label
        )

        CONF_ORDER = ['LOW', 'MEDIUM', 'HIGH', 'PERFECT']
        if CONF_ORDER.index(confidence) < CONF_ORDER.index(MIN_CONFIDENCE):
            print(f"  ⏭️  {ticker} Confidence {confidence} below {MIN_CONFIDENCE}")
            return

        pair_name = ticker.replace('-USD', '/USDT')
        sig = Signal(
            pair=pair_name,
            style=style,
            entry_zone_top=levels['zone_top'],
            entry_zone_bottom=levels['zone_bottom'],
            entry_moderate=levels['moderate']['entry'],
            entry_aggressive=levels['aggressive']['entry'],
            sl=levels['moderate']['sl'],
            tp1=levels['tp1'],
            tp2=levels['tp2'],
            tp3=levels['tp3'],
            rr=levels['moderate']['rr'],
            rr_aggressive=levels['aggressive']['rr'],
            atr=levels['atr'],
            confidence=confidence,
            confirmations=confs,
            warnings=warns,
            fib_382=levels['fib_382'],
            fib_50=levels['fib_50'],
            fib_confluence=levels['fib_confluence'],
            csa_type=setup_found.get('csa_type', 'CSA_EARLY'),
            btc_score=btc_ctx['score'] if btc_ctx else 'N/A',
            funding_rate=funding_rate,
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M UTC'),
            bb_expanding=levels['bb_expanding'],
            tf_alignment=tf_label,
        )

        msg = build_alert(sig, tfs)
        send_telegram(msg)
        mark_sent(cache, ticker, style)
        print(f"  🚨 ALERT: {pair_name} @ {current_price:.4f} | {confidence}")

    except Exception as e:
        print(f"  ❌ Error scanning {ticker}: {e}")
        traceback.print_exc()

# ============================================================
# MAIN
# ============================================================
def main():
    print(f"\n{'='*50}")
    print(f"  BBMA 9.8 SPOT SCANNER (BUY ONLY)")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Pairs: {len(PAIRS)} | Styles: {', '.join(STYLES)}")
    print(f"  BTC Filter: {BTC_FILTER_MODE} | Cooldown: {ALERT_COOLDOWN_HOURS}h")
    print(f"  Min R/R: {MIN_RR}× | Min Confidence: {MIN_CONFIDENCE}")
    print(f"  DRY RUN: {DRY_RUN}")
    print(f"{'='*50}")

    cache = load_cache()
    tickers = list(PAIRS.keys())
    alert_count = 0

    for style, tfs in STYLES.items():
        print(f"\n{'─'*45}")
        print(f"  STYLE: {style} ({tfs['big']}→{tfs['mid']}→{tfs['small']})")
        print(f"{'─'*45}")

        days = tfs['lookback_days']
        fib_lookback = tfs['fib_lookback']

        print(f"  📡 Fetching {len(tickers)} tickers on {tfs['big']}...")
        big_data = fetch_multiple_tickers(tickers, tfs['big'], days)
        if not big_data:
            print("  ⚠️ Big TF fetch failed, skipping style")
            continue

        print(f"  📡 Fetching {len(tickers)} tickers on {tfs['mid']}...")
        mid_data = fetch_multiple_tickers(tickers, tfs['mid'], days)
        if not mid_data:
            print("  ⚠️ Mid TF fetch failed, skipping style")
            continue

        print(f"  📡 Fetching {len(tickers)} tickers on {tfs['small']}...")
        small_data = fetch_multiple_tickers(tickers, tfs['small'], days)
        if not small_data:
            print("  ⚠️ Small TF fetch failed, skipping style")
            continue

        btc_df = big_data.get('BTC-USD')
        btc_ctx = get_btc_structure_from_df(btc_df) if btc_df is not None else None
        if btc_ctx is None:
            print("  ⚠️ BTC data unavailable, skipping style")
            continue
        if not btc_ctx['bullish']:
            print(f"  ⏭️ BTC not bullish ({btc_ctx['score']}) — skipping style")
            continue

        for ticker, cfg in PAIRS.items():
            df_big = big_data.get(ticker)
            df_mid = mid_data.get(ticker)
            df_small = small_data.get(ticker)

            if df_big is None or df_mid is None or df_small is None:
                print(f"  ⏭️ Missing data for {ticker}")
                continue

            scan_single_pair_with_dfs(
                ticker, cfg, cache,
                df_big, df_mid, df_small,
                btc_ctx, tfs, style, fib_lookback
            )

        style_keys = [k for k in cache.keys() if k.endswith(f"_{style}")]
        alert_count += len([k for k in style_keys if cache[k] > (datetime.now() - timedelta(hours=1)).isoformat()])

    save_cache(cache)

    if not DRY_RUN:
        heartbeat = f"""
🔵 <b>BBMA Scanner Heartbeat</b>
⏰ {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}
📊 Coins scanned: {len(PAIRS)}
📈 Alerts sent (last hour): {alert_count}
🟢 BTC Structure: {btc_ctx['score'] if btc_ctx else 'N/A'} ({'Bullish' if btc_ctx and btc_ctx['bullish'] else 'Bearish'})
✅ Status: Running
"""
        send_telegram(heartbeat)

    print(f"\n{'='*50}")
    print("  Scan Complete")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    main()
