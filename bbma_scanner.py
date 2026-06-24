"""
BBMA Crypto Spot Scanner — Final Merged Version
================================================
BUY ONLY | Spot Trading | Intraday + Swing

Merged best of:
  - bbma_scanner.py    (aku punya): alert cache, BTC filter, liquidity guard,
                                     volume spike, strict R/R, vectorized OBV
  - bbma_crypto_scanner_v2.py      : per-pair BB std, RSI, ATR SL, Fibonacci,
                                     confidence scoring, 3-TF alignment,
                                     retry logic, DRY_RUN, CSA Kukuh/Early,
                                     progress alerts, split long TG messages

Removed from v2:
  - All SELL logic (spot buy-only)
  - Funding rate (Coinglass API optional — keep if API key set)
  - ADX (simplified version in v2 was mathematically incorrect)
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
COINGLASS_API_KEY  = os.environ.get('COINGLASS_API_KEY')   # optional
ALERT_CACHE_PATH   = os.environ.get('ALERT_CACHE_PATH', 'alert_cache.json')

if not DRY_RUN and (not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID):
    raise ValueError("Missing Telegram credentials in GitHub Secrets!")

# ── Per-pair config: BB std tuned to volatility ─────────────────────────────
PAIRS: Dict[str, Dict] = {
    'BTC-USD': {'bb_std': 2.0, 'vol_threshold': 1.2},
    'ETH-USD': {'bb_std': 2.0, 'vol_threshold': 1.2},
    'SOL-USD': {'bb_std': 2.5, 'vol_threshold': 1.5},
    'BNB-USD': {'bb_std': 2.5, 'vol_threshold': 1.5},
    'XRP-USD': {'bb_std': 2.5, 'vol_threshold': 1.5},
}

# ── Timeframe styles (2-TF: big for trend, small for entry) ─────────────────
STYLES: Dict[str, Dict] = {
    'Intraday': {'big': '4h',  'small': '1h',  'lookback_days': 60},
    'Swing':    {'big': '1d',  'small': '4h',  'lookback_days': 365},
}

# ── Indicator settings ───────────────────────────────────────────────────────
BB_PERIOD            = 20
RSI_PERIOD           = 14
RSI_OVERSOLD         = 40    # relaxed from 30 — crypto rarely hits 30 in uptrend
ATR_PERIOD           = 14
ATR_SL_MULTIPLIER    = 1.5
OBV_EMA_PERIOD       = 20
VOL_AVG_PERIOD       = 20
VOLUME_SPIKE_MULT    = 1.5   # candle vol must be 1.5× 20-period avg

# ── Risk filters ─────────────────────────────────────────────────────────────
MIN_AVG_VOLUME_USD   = 50_000_000   # $50M liquidity guard
MIN_RR               = 1.5          # minimum R/R to send alert
MAX_DRIFT_PCT        = 0.06         # 6% drift = setup expired

# ── BTC market structure ─────────────────────────────────────────────────────
BTC_FILTER_MODE      = 'soft'       # 'soft' = 2/3, 'strict' = 3/3

# ── Alert dedup ──────────────────────────────────────────────────────────────
ALERT_COOLDOWN_HOURS = 4

# ── Confidence thresholds ────────────────────────────────────────────────────
MIN_CONFIDENCE       = 'MEDIUM'     # skip LOW confidence setups

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
    csa_type:           str   = 'CSA_EARLY'   # 'CSA_EARLY' or 'CSA_KUKUH'
    btc_score:          str   = ''
    funding_rate:       Optional[float] = None
    timestamp:          str   = ''

class BBMAState(Enum):
    NONE        = 0
    EXTREME_BUY = 1
    MHV_BUY     = 2
    CSA_BUY     = 3
    REENTRY_BUY = 4

# ============================================================
# ALERT CACHE (dedup)
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
    delta    = prices.diff()
    gain     = delta.where(delta > 0, 0).rolling(period).mean()
    loss     = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs       = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hl  = df['High'] - df['Low']
    hc  = (df['High'] - df['Close'].shift()).abs()
    lc  = (df['Low']  - df['Close'].shift()).abs()
    tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def calculate_obv_vectorized(df: pd.DataFrame) -> pd.Series:
    """Vectorized OBV — fast, no Python loop."""
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

    # Bollinger Bands
    df['bb_mid']   = df['Close'].rolling(BB_PERIOD).mean()
    _std           = df['Close'].rolling(BB_PERIOD).std()
    df['bb_upper'] = df['bb_mid'] + (_std * bb_std)
    df['bb_lower'] = df['bb_mid'] - (_std * bb_std)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']

    # LWMA zones
    df['ma5_high']  = calculate_lwma(df['High'], 5)
    df['ma10_high'] = calculate_lwma(df['High'], 10)
    df['ma5_low']   = calculate_lwma(df['Low'],  5)
    df['ma10_low']  = calculate_lwma(df['Low'],  10)

    # Trend anchor
    df['ema50'] = df['Close'].ewm(span=50, adjust=False).mean()

    # RSI
    df['rsi'] = calculate_rsi(df['Close'], RSI_PERIOD)

    # ATR
    df['atr'] = calculate_atr(df, ATR_PERIOD)

    # OBV (vectorized)
    df['obv']         = calculate_obv_vectorized(df)
    df['obv_ema']     = df['obv'].ewm(span=OBV_EMA_PERIOD, adjust=False).mean()
    df['obv_bullish'] = df['obv'] > df['obv_ema']

    # Volume spike
    df['vol_avg20']  = df['Volume'].rolling(VOL_AVG_PERIOD).mean()
    df['vol_ratio']  = df['Volume'] / df['vol_avg20']
    df['vol_spike']  = df['vol_ratio'] >= VOLUME_SPIKE_MULT

    return df.dropna()

# ============================================================
# LIQUIDITY GUARD
# ============================================================
def check_liquidity(df: pd.DataFrame, ticker: str) -> bool:
    avg = (df['Close'].tail(20) * df['Volume'].tail(20)).mean()
    if avg < MIN_AVG_VOLUME_USD:
        print(f"🚫 Liquidity fail {ticker}: ${avg:,.0f} < ${MIN_AVG_VOLUME_USD:,.0f}")
        return False
    return True

# ============================================================
# DATA FETCHER (with retry + QC)
# ============================================================
def fetch_data(ticker: str, interval: str, lookback_days: int,
               max_retries: int = 3) -> pd.DataFrame:
    for attempt in range(max_retries):
        try:
            end   = datetime.now()
            start = end - timedelta(days=lookback_days)
            df    = yf.download(ticker, start=start, end=end,
                                interval=interval, progress=False, auto_adjust=True)

            if df.empty:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                return pd.DataFrame()

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)

            # normalise column names
            col_map = {}
            for c in df.columns:
                lc = c.lower()
                if 'open'   in lc: col_map[c] = 'Open'
                elif 'high'  in lc: col_map[c] = 'High'
                elif 'low'   in lc: col_map[c] = 'Low'
                elif 'close' in lc: col_map[c] = 'Close'
                elif 'vol'   in lc: col_map[c] = 'Volume'
            df.rename(columns=col_map, inplace=True)

            # QC 1: freshness (24h for crypto)
            last_ts = df.index[-1]
            if isinstance(last_ts, pd.Timestamp):
                if (datetime.now(last_ts.tzinfo) - last_ts).total_seconds() > 86400:
                    print(f"⚠️ Stale data {ticker} ({interval})")
                    return pd.DataFrame()

            # QC 2: price anomaly (>25% from 20-candle median)
            med = df['Close'].iloc[-20:].median()
            if abs(df['Close'].iloc[-1] - med) / med > 0.25:
                print(f"⚠️ Price anomaly {ticker}")
                return pd.DataFrame()

            # QC 3: minimum candles
            if len(df) < 60:
                print(f"⚠️ Insufficient data {ticker} ({len(df)} candles)")
                return pd.DataFrame()

            return df

        except Exception as e:
            print(f"❌ Attempt {attempt+1}/{max_retries} {ticker}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)

    return pd.DataFrame()

# ============================================================
# BTC MARKET STRUCTURE FILTER
# ============================================================
def get_btc_structure(interval: str, lookback_days: int) -> Optional[Dict]:
    df = fetch_data('BTC-USD', interval, lookback_days)
    if df.empty:
        return None
    df = get_indicators(df, bb_std=2.0)
    if df.empty:
        return None

    last, prev = df.iloc[-1], df.iloc[-2]
    c1 = bool(last['Close'] > last['ema50'])
    c2 = bool(last['ema50'] > prev['ema50'])
    c3 = bool(last['obv_bullish'])
    n  = sum([c1, c2, c3])

    bullish = (n >= 2) if BTC_FILTER_MODE == 'soft' else (n == 3)

    return {
        'bullish':   bullish,
        'score':     f"{n}/3",
        'c1_price':  c1,
        'c2_slope':  c2,
        'c3_obv':    c3,
    }

# ============================================================
# FUNDING RATE (optional)
# ============================================================
def fetch_funding_rate(symbol: str) -> Optional[float]:
    if not COINGLASS_API_KEY:
        return None
    try:
        url  = f"https://open-api.coinglass.com/public/v2/funding?symbol={symbol}&timeType=h8"
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
    url     = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    # split if >4000 chars
    chunks  = [message[i:i+4000] for i in range(0, len(message), 4000)]
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
# BBMA STATE MACHINE (BUY ONLY)
# ============================================================
class BBMABuyTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.state             = BBMAState.NONE
        self.extreme_low       = None
        self.rsi_at_extreme    = None
        self.mhv_confirmed     = False
        self.csa_confirmed     = False
        self.csa_type          = 'CSA_EARLY'

    def update(self, row: pd.Series, prev: pd.Series) -> Optional[Dict]:
        close     = row['Close'];  open_    = row['Open']
        low       = row['Low'];    high     = row['High']
        bb_lower  = row['bb_lower']; bb_mid = row['bb_mid']
        ma5_high  = row['ma5_high']; ma10_high = row['ma10_high']
        ma5_low   = row['ma5_low'];  ma10_low  = row['ma10_low']
        rsi       = row['rsi']
        obv_bull  = bool(row['obv_bullish'])
        vol_spike = bool(row['vol_spike'])

        bull = close > open_;  bear = close < open_
        pbull = prev['Close'] > prev['Open']
        pbear = prev['Close'] < prev['Open']

        # 1. EXTREME BUY
        if (ma5_low < bb_lower or ma10_low < bb_lower) and bull and pbear:
            self.state          = BBMAState.EXTREME_BUY
            self.extreme_low    = low
            self.rsi_at_extreme = rsi
            self.mhv_confirmed  = False
            self.csa_confirmed  = False
            return None

        # 2. MHV
        if self.state == BBMAState.EXTREME_BUY:
            if (close >= bb_lower) and bear and pbull:
                self.state         = BBMAState.MHV_BUY
                self.mhv_confirmed = True
                return None
            if close < bb_lower:
                self.reset()
            return None

        # 3. CS ARAH (Early or Kukuh)
        if self.state == BBMAState.MHV_BUY and self.mhv_confirmed:
            csa_early = close > ma5_low and close > ma10_low
            csa_kukuh = csa_early and close > bb_mid
            if csa_early:
                self.state         = BBMAState.CSA_BUY
                self.csa_confirmed = True
                self.csa_type      = 'CSA_KUKUH' if csa_kukuh else 'CSA_EARLY'
            return None

        # 4. RE-ENTRY BUY (all filters must pass)
        if self.state == BBMAState.CSA_BUY and self.csa_confirmed:
            zone_top    = max(ma5_low, ma10_low)
            zone_bottom = min(ma5_low, ma10_low)

            near_zone     = zone_bottom * 0.985 <= low <= zone_top * 1.015
            not_crashed   = close >= zone_bottom * 0.98
            below_resist  = close <= ma5_high and close <= ma10_high and close <= bb_mid
            reversal      = bull and pbear

            valid = (
                near_zone    and
                not_crashed  and
                below_resist and
                reversal     and
                obv_bull     and   # accumulation
                vol_spike         # real buying pressure
            )

            if valid:
                self.state = BBMAState.REENTRY_BUY
                return {
                    'zone_top':      zone_top,
                    'zone_bottom':   zone_bottom,
                    'trigger_price': close,
                    'obv_confirmed': obv_bull,
                    'vol_spike':     vol_spike,
                    'rsi':           rsi,
                    'csa_type':      self.csa_type,
                }

        return None

# ============================================================
# LEVEL CALCULATION (ATR SL + Fibonacci)
# ============================================================
def calculate_levels(signal: Dict, current_price: float,
                     df: pd.DataFrame) -> Optional[Dict]:
    last         = df.iloc[-1]
    zone_top     = signal['zone_top']
    zone_bottom  = signal['zone_bottom']
    zone_center  = (zone_top + zone_bottom) / 2

    # drift check
    drift = abs(current_price - zone_center) / zone_center
    if drift > MAX_DRIFT_PCT:
        print(f"⚠️ Setup expired: drifted {drift:.1%}")
        return None

    # Fibonacci confluence
    swing_high   = df['High'].iloc[-20:].max()
    swing_low    = df['Low'].iloc[-20:].min()
    fibs         = calculate_fibonacci(swing_high, swing_low)
    fib_382      = fibs['0.382']
    fib_50       = fibs['0.500']
    fib_conf     = (abs(zone_center - fib_382) / zone_center < 0.02 or
                    abs(zone_center - fib_50)  / zone_center < 0.02)

    # ATR-based SL
    atr          = last['atr']
    sl_bb        = last['bb_lower']
    sl_atr       = zone_center - (atr * ATR_SL_MULTIPLIER)
    sl           = min(sl_bb, sl_atr)

    entry_mod    = zone_center
    entry_agg    = zone_bottom
    tp1          = last['ma5_high']
    tp2          = last['bb_mid']
    tp3          = last['bb_upper']

    def rr(entry):
        return (tp2 - entry) / (entry - sl) if entry > sl else 0

    return {
        'moderate':      {'entry': entry_mod, 'sl': sl, 'rr': rr(entry_mod)},
        'aggressive':    {'entry': entry_agg, 'sl': sl, 'rr': rr(entry_agg)},
        'tp1': tp1, 'tp2': tp2, 'tp3': tp3,
        'zone_top':      zone_top,
        'zone_bottom':   zone_bottom,
        'atr':           atr,
        'fib_confluence': fib_conf,
        'fib_382':       fib_382,
        'fib_50':        fib_50,
        'drift_pct':     drift,
    }

# ============================================================
# CONFIDENCE SCORING
# ============================================================
CONFIDENCE_MAP = {(0,3): 'LOW', (4,5): 'MEDIUM', (6,7): 'HIGH', (8,99): 'PERFECT'}

def score_to_label(score: int) -> str:
    for (lo, hi), label in CONFIDENCE_MAP.items():
        if lo <= score <= hi:
            return label
    return 'LOW'

def calculate_confidence(df: pd.DataFrame, levels: Dict,
                         signal: Dict,
                         btc_ctx: Dict,
                         funding_rate: Optional[float]) -> Tuple[str, List[str], List[str]]:
    confirmations: List[str] = []
    warnings:      List[str] = []
    score = 0
    last  = df.iloc[-1]

    # 1. Volume spike (2pts)
    if signal.get('vol_spike'):
        confirmations.append(f"Volume spike {last['vol_ratio']:.1f}× avg ✅")
        score += 2
    else:
        warnings.append(f"Volume weak ({last['vol_ratio']:.1f}× avg)")

    # 2. OBV (2pts)
    if signal.get('obv_confirmed'):
        confirmations.append("OBV bullish — accumulation confirmed ✅")
        score += 2
    else:
        warnings.append("OBV diverging from price")

    # 3. RSI near oversold (1pt)
    rsi = signal.get('rsi', last['rsi'])
    if rsi <= RSI_OVERSOLD:
        confirmations.append(f"RSI {rsi:.1f} — oversold zone ✅")
        score += 1
    elif rsi <= RSI_OVERSOLD + 10:
        confirmations.append(f"RSI {rsi:.1f} — approaching oversold")
        score += 0
    else:
        warnings.append(f"RSI {rsi:.1f} — not yet oversold")

    # 4. ATR manageable (1pt)
    atr_pct = levels['atr'] / last['Close'] * 100
    if atr_pct < 5:
        confirmations.append(f"ATR {atr_pct:.1f}% — manageable volatility ✅")
        score += 1
    elif atr_pct > 10:
        warnings.append(f"High volatility ATR {atr_pct:.1f}%")

    # 5. Fibonacci confluence (1pt)
    if levels.get('fib_confluence'):
        confirmations.append(f"Fibonacci confluence (38.2%={levels['fib_382']:.4f}) ✅")
        score += 1
    else:
        warnings.append("No Fibonacci confluence")

    # 6. BTC structure (1pt)
    if btc_ctx and btc_ctx['bullish']:
        confirmations.append(f"BTC market structure bullish ({btc_ctx['score']}) ✅")
        score += 1

    # 7. Funding rate (1pt, optional)
    if funding_rate is not None and funding_rate < 0:
        confirmations.append(f"Funding rate {funding_rate:.4f}% — shorts paying ✅")
        score += 1

    # 8. R/R (2pts excellent, 1pt good)
    rr = levels['moderate']['rr']
    if rr >= 2.0:
        confirmations.append(f"R/R {rr:.1f}× — excellent ✅")
        score += 2
    elif rr >= MIN_RR:
        confirmations.append(f"R/R {rr:.1f}× — good")
        score += 1
    else:
        warnings.append(f"R/R {rr:.1f}× — below minimum")

    # 9. CSA Kukuh bonus (1pt)
    if signal.get('csa_type') == 'CSA_KUKUH':
        confirmations.append("CS Arah Kukuh — strong directional signal ✅")
        score += 1

    # 10. BB expanding (1pt)
    if last['bb_width'] > 0.05:
        confirmations.append(f"BB expanding ({last['bb_width']:.2%}) — trending ✅")
        score += 1
    else:
        warnings.append(f"BB narrow ({last['bb_width']:.2%}) — ranging market")

    return score_to_label(score), confirmations, warnings

# ============================================================
# TELEGRAM ALERT
# ============================================================
def build_alert(signal_obj: Signal, tfs: Dict) -> str:
    conf_emoji = {'LOW': '⚠️', 'MEDIUM': '👍', 'HIGH': '🔥', 'PERFECT': '💎'}
    emoji      = conf_emoji.get(signal_obj.confidence, '👍')
    csa_label  = "CS Arah Kukuh 💪" if signal_obj.csa_type == 'CSA_KUKUH' else "CS Arah Awal"

    confs  = "\n".join([f"  ✅ {c}" for c in signal_obj.confirmations]) or "  —"
    warns  = "\n".join([f"  ⚠️ {w}" for w in signal_obj.warnings])      or "  ✅ None"

    funding_line = ""
    if signal_obj.funding_rate is not None:
        funding_line = f"\n• Funding Rate : {signal_obj.funding_rate:.4f}%"

    fib_line = ""
    if signal_obj.fib_confluence:
        fib_line = f"\n• Fib 38.2%    : {signal_obj.fib_382:.4f}  |  50%: {signal_obj.fib_50:.4f}"

    return f"""
🟢 <b>BBMA BUY SETUP</b> {emoji} <b>{signal_obj.confidence}</b>

📊 <b>{signal_obj.pair}</b>  |  {signal_obj.style}  ({tfs['big']} → {tfs['small']})
🎯 Pattern : Re-Entry Buy ({csa_label})
✅ Cycle   : Extreme → MHV → CS Arah → Re-Entry
⏰ Time    : {signal_obj.timestamp}

━━━━━━━━━━━━━━━━━━━━

🌐 <b>MARKET CONTEXT</b>
• BTC Structure : {"✅ Bullish" if signal_obj.btc_score else "—"} ({signal_obj.btc_score})
• ATR (Volatility): {signal_obj.atr:.4f}  ({signal_obj.atr / signal_obj.entry_moderate * 100:.1f}%){funding_line}{fib_line}

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
• TP2 : {signal_obj.tp2:.4f}  (Mid BB)  ← Wajib close here
• TP3 : {signal_obj.tp3:.4f}  (Upper BB)

━━━━━━━━━━━━━━━━━━━━

✅ <b>CONFIRMATIONS</b>
{confs}

⚠️ <b>WARNINGS</b>
{warns}

━━━━━━━━━━━━━━━━━━━━
<i>⚡ Verify live price before entry. Not financial advice.</i>
"""

# ============================================================
# SCANNER
# ============================================================
def scan_pair(ticker: str, pair_cfg: Dict, cache: Dict):
    bb_std = pair_cfg.get('bb_std', 2.0)

    for style, tfs in STYLES.items():
        print(f"\n{'─'*45}")
        print(f"  {ticker} | {style} | {tfs['big']}→{tfs['small']}")

        try:
            # ── Dedup ──────────────────────────────────────────────────────
            if is_duplicate(cache, ticker, style):
                continue

            # ── Big TF ─────────────────────────────────────────────────────
            df_big = fetch_data(ticker, tfs['big'], tfs['lookback_days'])
            if df_big.empty: continue
            df_big = get_indicators(df_big, bb_std)
            if df_big.empty: continue

            # ── Liquidity guard ────────────────────────────────────────────
            if not check_liquidity(df_big, ticker): continue

            # ── Uptrend on big TF ──────────────────────────────────────────
            last_big = df_big.iloc[-1]
            if not (last_big['ema50'] < last_big['bb_mid']):
                print(f"  ⏭️  Not uptrend ({tfs['big']})")
                continue

            # ── BTC structure ──────────────────────────────────────────────
            btc_ctx = get_btc_structure(tfs['big'], tfs['lookback_days'])
            if btc_ctx is None:
                print("  ⚠️  BTC data unavailable — skip")
                continue
            if not btc_ctx['bullish']:
                print(f"  ⏭️  BTC not bullish ({btc_ctx['score']})")
                continue

            # ── Small TF ───────────────────────────────────────────────────
            df_small = fetch_data(ticker, tfs['small'], tfs['lookback_days'])
            if df_small.empty: continue
            df_small = get_indicators(df_small, bb_std)
            if df_small.empty: continue

            # ── State machine ──────────────────────────────────────────────
            tracker     = BBMABuyTracker()
            setup_found = None
            for i in range(1, len(df_small)):
                result = tracker.update(df_small.iloc[i], df_small.iloc[i-1])
                if result:
                    setup_found = result
                    break

            if not setup_found:
                print("  ⏭️  No BBMA setup found")
                continue

            # ── Level calculation ──────────────────────────────────────────
            current_price = df_small.iloc[-1]['Close']
            levels        = calculate_levels(setup_found, current_price, df_small)
            if not levels: continue

            # ── R/R filter ─────────────────────────────────────────────────
            if levels['moderate']['rr'] < MIN_RR:
                print(f"  ⏭️  R/R {levels['moderate']['rr']:.1f}× < {MIN_RR}×")
                continue

            # ── Funding rate ───────────────────────────────────────────────
            symbol       = ticker.replace('-USD', '')
            funding_rate = fetch_funding_rate(symbol)

            # ── Confidence ─────────────────────────────────────────────────
            confidence, confs, warns = calculate_confidence(
                df_small, levels, setup_found, btc_ctx, funding_rate
            )

            CONF_ORDER = ['LOW', 'MEDIUM', 'HIGH', 'PERFECT']
            if CONF_ORDER.index(confidence) < CONF_ORDER.index(MIN_CONFIDENCE):
                print(f"  ⏭️  Confidence {confidence} below minimum {MIN_CONFIDENCE}")
                continue

            # ── Build signal object ────────────────────────────────────────
            pair_name  = ticker.replace('-USD', '/USDT')
            sig        = Signal(
                pair              = pair_name,
                style             = style,
                entry_zone_top    = levels['zone_top'],
                entry_zone_bottom = levels['zone_bottom'],
                entry_moderate    = levels['moderate']['entry'],
                entry_aggressive  = levels['aggressive']['entry'],
                sl                = levels['moderate']['sl'],
                tp1               = levels['tp1'],
                tp2               = levels['tp2'],
                tp3               = levels['tp3'],
                rr                = levels['moderate']['rr'],
                rr_aggressive     = levels['aggressive']['rr'],
                atr               = levels['atr'],
                confidence        = confidence,
                confirmations     = confs,
                warnings          = warns,
                fib_382           = levels['fib_382'],
                fib_50            = levels['fib_50'],
                fib_confluence    = levels['fib_confluence'],
                csa_type          = setup_found.get('csa_type', 'CSA_EARLY'),
                btc_score         = btc_ctx['score'],
                funding_rate      = funding_rate,
                timestamp         = datetime.now().strftime('%Y-%m-%d %H:%M UTC'),
            )

            # ── Send alert ─────────────────────────────────────────────────
            msg = build_alert(sig, tfs)
            send_telegram(msg)
            mark_sent(cache, ticker, style)
            print(f"  🚨 ALERT: {pair_name} @ {current_price:.4f} | {confidence}")

        except Exception as e:
            print(f"  ❌ Error: {e}")
            traceback.print_exc()

# ============================================================
# MAIN
# ============================================================
def main():
    print(f"\n{'='*50}")
    print(f"  BBMA SPOT SCANNER (BUY ONLY)")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Pairs: {len(PAIRS)} | Styles: {', '.join(STYLES)}")
    print(f"  BTC Filter: {BTC_FILTER_MODE} | Cooldown: {ALERT_COOLDOWN_HOURS}h")
    print(f"  Min R/R: {MIN_RR}× | Min Confidence: {MIN_CONFIDENCE}")
    print(f"  DRY RUN: {DRY_RUN}")
    print(f"{'='*50}")

    cache = load_cache()

    for ticker, cfg in PAIRS.items():
        try:
            scan_pair(ticker, cfg, cache)
        except Exception as e:
            print(f"❌ Fatal {ticker}: {e}")

    save_cache(cache)
    print(f"\n{'='*50}")
    print("  Scan Complete")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    main()
