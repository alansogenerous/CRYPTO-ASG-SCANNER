import os
import json
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
TELEGRAM_CHAT_ID   = os.environ.get('TELEGRAM_CHAT_ID')

if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
    raise ValueError("Missing Telegram credentials in GitHub Secrets!")

# ── Liquidity: min avg dollar volume (20-candle) ────────────────────────────
MIN_AVG_VOLUME_USD = 50_000_000          # $50M — skip low-cap / illiquid pairs

# ── Alert dedup: ignore same pair+style signal within this window ────────────
ALERT_COOLDOWN_HOURS = 4                 # Re-Entry signals are valid ~4h

# ── Volume spike: candle volume must be this many × 20-period avg ────────────
VOLUME_SPIKE_MULTIPLIER = 1.5            # Filters wash-trading & weak moves

# ── BTC filter mode: 'strict' (all 3 conditions) or 'soft' (2 of 3) ─────────
# 'soft' allows altcoin setups even when BTC is sideways (OBV weak but price ok)
BTC_FILTER_MODE = 'soft'

PAIRS = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD', 'XRP-USD']

STYLES = {
    'Intraday': {'big': '4h',  'small': '1h',  'lookback_days': 60},
    'Swing':    {'big': '1d',  'small': '4h',  'lookback_days': 365},
}

BB_PERIOD      = 20
BB_STD         = 2.0
OBV_EMA_PERIOD = 20

# Alert cache file — persists between GitHub Action runs via artifact / repo file
CACHE_FILE = os.environ.get('ALERT_CACHE_PATH', 'alert_cache.json')

# ==========================================
# ALERT DEDUP CACHE
# ==========================================
def load_cache() -> Dict:
    try:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, 'r') as f:
                return json.load(f)
    except Exception:
        pass
    return {}

def save_cache(cache: Dict):
    try:
        with open(CACHE_FILE, 'w') as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        print(f"⚠️ Cache save failed: {e}")

def is_duplicate_alert(cache: Dict, ticker: str, style: str) -> bool:
    key = f"{ticker}_{style}"
    if key not in cache:
        return False
    last_sent = datetime.fromisoformat(cache[key])
    elapsed   = (datetime.now() - last_sent).total_seconds() / 3600
    if elapsed < ALERT_COOLDOWN_HOURS:
        print(f"🔕 DUPLICATE skipped: {key} (sent {elapsed:.1f}h ago, cooldown={ALERT_COOLDOWN_HOURS}h)")
        return True
    return False

def mark_alert_sent(cache: Dict, ticker: str, style: str):
    cache[f"{ticker}_{style}"] = datetime.now().isoformat()

# ==========================================
# INDICATOR CALCULATIONS
# ==========================================
def calculate_lwma(series: pd.Series, period: int) -> pd.Series:
    weights = np.arange(1, period + 1)
    return series.rolling(window=period).apply(
        lambda w: np.sum(w * weights) / np.sum(weights), raw=True
    )

def calculate_obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df['Close'].diff().fillna(0))
    return (direction * df['Volume']).cumsum()

def get_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Bollinger Bands
    df['bb_mid']   = df['Close'].rolling(BB_PERIOD).mean()
    bb_std         = df['Close'].rolling(BB_PERIOD).std()
    df['bb_upper'] = df['bb_mid'] + (bb_std * BB_STD)
    df['bb_lower'] = df['bb_mid'] - (bb_std * BB_STD)

    # LWMA
    df['ma5_high']  = calculate_lwma(df['High'], 5)
    df['ma10_high'] = calculate_lwma(df['High'], 10)
    df['ma5_low']   = calculate_lwma(df['Low'],  5)
    df['ma10_low']  = calculate_lwma(df['Low'],  10)

    # Trend anchor
    df['ema50'] = df['Close'].ewm(span=50, adjust=False).mean()

    # OBV + OBV EMA (accumulation vs distribution)
    df['obv']         = calculate_obv(df)
    df['obv_ema']     = df['obv'].ewm(span=OBV_EMA_PERIOD, adjust=False).mean()
    df['obv_bullish'] = df['obv'] > df['obv_ema']

    # Volume spike filter: rolling 20-period avg volume
    df['vol_avg20']   = df['Volume'].rolling(20).mean()
    df['vol_spike']   = df['Volume'] > (df['vol_avg20'] * VOLUME_SPIKE_MULTIPLIER)

    return df.dropna()

# ==========================================
# LIQUIDITY GUARD
# ==========================================
def check_liquidity(df: pd.DataFrame, ticker: str) -> bool:
    recent      = df.tail(20)
    avg_vol_usd = (recent['Close'] * recent['Volume']).mean()
    if avg_vol_usd < MIN_AVG_VOLUME_USD:
        print(f"🚫 LIQUIDITY FAIL {ticker}: avg ${avg_vol_usd:,.0f} < ${MIN_AVG_VOLUME_USD:,.0f}")
        return False
    return True

# ==========================================
# DATA FETCHER WITH QC
# ==========================================
def fetch_yfinance_data(ticker: str, interval: str, lookback_days: int) -> pd.DataFrame:
    try:
        end_date   = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)

        df = yf.download(ticker, start=start_date, end=end_date,
                         interval=interval, progress=False)
        if df.empty:
            print(f"❌ No data: {ticker} ({interval})")
            return pd.DataFrame()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)

        # QC 1 — staleness (48h)
        last_ts = df.index[-1]
        if isinstance(last_ts, pd.Timestamp):
            if (datetime.now(last_ts.tzinfo) - last_ts).total_seconds() > 172800:
                print(f"⚠️ Stale: {ticker} ({interval}), last={last_ts}")
                return pd.DataFrame()

        # QC 2 — price anomaly (>20% from 10-candle median)
        recent_median = df['Close'].iloc[-10:].median()
        last_close    = df['Close'].iloc[-1]
        if abs(last_close - recent_median) / recent_median > 0.20:
            print(f"⚠️ Price anomaly: {ticker} last={last_close:.2f} median={recent_median:.2f}")
            return pd.DataFrame()

        return df

    except Exception as e:
        print(f"❌ Fetch error {ticker}: {e}")
        return pd.DataFrame()

# ==========================================
# BTC MARKET STRUCTURE FILTER (SOFT MODE)
# ==========================================
def get_btc_structure(interval: str, lookback_days: int) -> Optional[Dict]:
    """
    SOFT mode  — pass if 2 of 3 conditions true (allows sideways BTC).
    STRICT mode — all 3 must be true.

    Conditions:
      C1: BTC Close > EMA50
      C2: EMA50 slope rising
      C3: BTC OBV bullish (accumulation)
    """
    try:
        df = fetch_yfinance_data('BTC-USD', interval, lookback_days)
        if df.empty:
            return None
        df = get_indicators(df)
        if df.empty:
            return None

        last = df.iloc[-1]
        prev = df.iloc[-2]

        c1 = bool(last['Close'] > last['ema50'])    # price above trend
        c2 = bool(last['ema50'] > prev['ema50'])    # trend rising
        c3 = bool(last['obv_bullish'])              # accumulation

        conditions_met = sum([c1, c2, c3])

        if BTC_FILTER_MODE == 'strict':
            bullish = (conditions_met == 3)
        else:  # soft
            bullish = (conditions_met >= 2)

        return {
            'bullish':      bullish,
            'conditions':   conditions_met,
            'c1_price':     c1,
            'c2_ema_slope': c2,
            'c3_obv':       c3,
            'close':        last['Close'],
            'ema50':        last['ema50'],
        }
    except Exception as e:
        print(f"❌ BTC structure error: {e}")
        return None

# ==========================================
# TELEGRAM
# ==========================================
def send_telegram(message: str):
    url     = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {'chat_id': TELEGRAM_CHAT_ID, 'text': message, 'parse_mode': 'HTML'}
    try:
        resp = requests.post(url, json=payload, timeout=10)
        if resp.status_code == 200:
            print("✅ Alert sent")
        else:
            print(f"❌ TG Error: {resp.text}")
    except Exception as e:
        print(f"❌ Send failed: {e}")

# ==========================================
# BBMA STATE MACHINE
# ==========================================
class BBMAState(Enum):
    NONE        = 0
    EXTREME_BUY = 1
    MHV_BUY     = 2
    CSA_BUY     = 3
    REENTRY_BUY = 4

class BBMABuyTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.state           = BBMAState.NONE
        self.extreme_low     = None
        self.mhv_confirmed   = False
        self.csa_confirmed   = False
        self.setup_timestamp = None

    def update(self, row: pd.Series, prev_row: pd.Series) -> Optional[Dict]:
        close      = row['Close']
        open_      = row['Open']
        low        = row['Low']
        bb_lower   = row['bb_lower']
        bb_mid     = row['bb_mid']
        ma5_high   = row['ma5_high']
        ma10_high  = row['ma10_high']
        ma5_low    = row['ma5_low']
        ma10_low   = row['ma10_low']
        obv_bull   = bool(row['obv_bullish'])
        vol_spike  = bool(row['vol_spike'])       # ← volume spike filter

        is_bullish   = close > open_
        is_bearish   = close < open_
        prev_bullish = prev_row['Close'] > prev_row['Open']
        prev_bearish = prev_row['Close'] < prev_row['Open']

        # ── 1. EXTREME BUY ──────────────────────────────────────────────────
        extreme_buy = (
            (ma5_low < bb_lower or ma10_low < bb_lower) and
            is_bullish and prev_bearish
        )
        if extreme_buy:
            self.state         = BBMAState.EXTREME_BUY
            self.extreme_low   = low
            self.mhv_confirmed = False
            self.csa_confirmed = False
            return None

        # ── 2. MHV ──────────────────────────────────────────────────────────
        if self.state == BBMAState.EXTREME_BUY:
            if (close >= bb_lower) and is_bearish and prev_bullish:
                self.state         = BBMAState.MHV_BUY
                self.mhv_confirmed = True
                return None
            if close < bb_lower:
                self.reset()
                return None

        # ── 3. CS ARAH ──────────────────────────────────────────────────────
        if self.state == BBMAState.MHV_BUY and self.mhv_confirmed:
            if close > ma5_low and close > ma10_low:
                self.state         = BBMAState.CSA_BUY
                self.csa_confirmed = True
                return None

        # ── 4. RE-ENTRY BUY ─────────────────────────────────────────────────
        # All conditions must pass — strict precision-first approach
        if self.state == BBMAState.CSA_BUY and self.csa_confirmed:
            price_not_crashed = (
                close >= ma5_low  * 0.98 and
                close >= ma10_low * 0.98
            )
            touch_zone = (
                (low <= ma5_low  * 1.01 and low >= ma5_low  * 0.99) or
                (low <= ma10_low * 1.01 and low >= ma10_low * 0.99)
            )
            close_near_zone = close >= ma5_low * 0.99

            valid_reentry = (
                price_not_crashed and
                touch_zone        and
                close_near_zone   and
                close <= ma5_high and
                close <= ma10_high and
                close <= bb_mid   and
                is_bullish        and
                prev_bearish      and
                obv_bull          and   # OBV: accumulation confirmed
                vol_spike              # Volume spike: real buying pressure
            )

            if valid_reentry:
                self.state           = BBMAState.REENTRY_BUY
                self.setup_timestamp = datetime.now()
                return {
                    'type':          'BUY',
                    'trigger_price': close,
                    'trigger_time':  datetime.now().isoformat(),
                    'obv_confirmed': obv_bull,
                    'vol_spike':     vol_spike,
                }

        return None

# ==========================================
# LEVEL CALCULATION
# ==========================================
def calculate_levels_buy(current_price: float, df: pd.DataFrame) -> Optional[Dict]:
    last = df.iloc[-1]

    ma5_low   = last['ma5_low']
    ma10_low  = last['ma10_low']
    ma5_high  = last['ma5_high']
    bb_lower  = last['bb_lower']
    bb_mid    = last['bb_mid']
    bb_upper  = last['bb_upper']

    zone_top    = max(ma5_low, ma10_low)
    zone_bottom = min(ma5_low, ma10_low)
    zone_center = (zone_top + zone_bottom) / 2

    drift_pct = abs(current_price - zone_center) / zone_center
    if drift_pct > 0.05:
        print(f"⚠️ Setup EXPIRED: drifted {drift_pct:.1%} from zone")
        return None

    entry_mod  = zone_center
    entry_agg  = zone_bottom
    sl         = min(bb_lower, zone_bottom * 0.985)
    tp1        = ma5_high
    tp2        = bb_mid
    tp3        = bb_upper

    def rr(entry):
        return (tp2 - entry) / (entry - sl) if entry > sl else 0

    return {
        'moderate':    {'entry': entry_mod, 'sl': sl, 'rr': rr(entry_mod)},
        'aggressive':  {'entry': entry_agg, 'sl': sl, 'rr': rr(entry_agg)},
        'tp1': tp1, 'tp2': tp2, 'tp3': tp3,
        'zone_top':    zone_top,
        'zone_bottom': zone_bottom,
        'drift_pct':   drift_pct,
    }

# ==========================================
# SCANNER
# ==========================================
def scan_pair(ticker: str, cache: Dict):
    tracker = BBMABuyTracker()

    for style, tfs in STYLES.items():
        print(f"\n{'─'*40}")
        print(f"Scanning {ticker} ({style}) | TF: {tfs['big']}→{tfs['small']}")

        try:
            # ── DEDUP CHECK ─────────────────────────────────────────────────
            if is_duplicate_alert(cache, ticker, style):
                continue

            # ── BIG TF ──────────────────────────────────────────────────────
            df_big = fetch_yfinance_data(ticker, tfs['big'], tfs['lookback_days'])
            if df_big.empty: continue
            df_big = get_indicators(df_big)
            if df_big.empty: continue

            # ── LIQUIDITY GUARD ──────────────────────────────────────────────
            if not check_liquidity(df_big, ticker): continue

            # ── UPTREND CHECK ────────────────────────────────────────────────
            last_big = df_big.iloc[-1]
            if not (last_big['ema50'] < last_big['bb_mid']):
                print(f"⏭️  Not uptrend on {tfs['big']}")
                continue

            # ── BTC MARKET STRUCTURE ─────────────────────────────────────────
            btc_ctx = get_btc_structure(tfs['big'], tfs['lookback_days'])
            if btc_ctx is None:
                print(f"⚠️  BTC data unavailable — skip")
                continue
            if not btc_ctx['bullish']:
                print(
                    f"⏭️  BTC not bullish ({btc_ctx['conditions']}/3 conditions) "
                    f"[price>{btc_ctx['c1_price']} slope>{btc_ctx['c2_ema_slope']} obv>{btc_ctx['c3_obv']}]"
                )
                continue

            # ── SMALL TF ────────────────────────────────────────────────────
            df_small = fetch_yfinance_data(ticker, tfs['small'], tfs['lookback_days'])
            if df_small.empty: continue
            df_small = get_indicators(df_small)
            if df_small.empty: continue

            # ── STATE MACHINE ────────────────────────────────────────────────
            tracker.reset()
            setup_found = None
            for i in range(1, len(df_small)):
                result = tracker.update(df_small.iloc[i], df_small.iloc[i-1])
                if result:
                    setup_found = result
                    break

            if not setup_found:
                print(f"⏭️  No BBMA setup found")
                continue

            # ── LEVEL CALC ───────────────────────────────────────────────────
            current_price = df_small.iloc[-1]['Close']
            levels = calculate_levels_buy(current_price, df_small)
            if not levels:
                print(f"⚠️  Setup expired (price drifted)")
                continue

            # ── SKIP LOW R/R ─────────────────────────────────────────────────
            # Minimum 1.5x R/R on moderate entry — else not worth the trade
            if levels['moderate']['rr'] < 1.5:
                print(f"⏭️  R/R too low ({levels['moderate']['rr']:.1f}x < 1.5x) — skip")
                continue

            # ── BUILD & SEND ALERT ───────────────────────────────────────────
            pair_name  = ticker.replace('-USD', '/USDT')
            now_str    = datetime.now().strftime('%Y-%m-%d %H:%M')
            btc_score  = f"{btc_ctx['conditions']}/3"

            msg = f"""
🟢 <b>BBMA BUY SETUP — {pair_name}</b>

💰 Price  : ${current_price:,.4f}
⏱️ Style  : {style}  ({tfs['big']} → {tfs['small']})
📈 Pattern: Re-Entry Buy (Full Cycle Confirmed)
✅ Cycle  : Extreme → MHV → CS Arah → Re-Entry

━━━━━━━━━━━━━━━━━━━━

🌐 <b>MARKET CONTEXT</b>
BTC Structure ({tfs['big']}): {"✅ Bullish" if btc_ctx["bullish"] else "⚠️ Weak"} ({btc_score} conditions)
  ├ Price > EMA50  : {"✅" if btc_ctx["c1_price"] else "❌"}
  ├ EMA50 Rising   : {"✅" if btc_ctx["c2_ema_slope"] else "❌"}
  └ OBV Accumulate : {"✅" if btc_ctx["c3_obv"] else "❌"}

{pair_name} Volume:
  ├ OBV Bullish    : {"✅" if setup_found["obv_confirmed"] else "❌"}
  └ Volume Spike   : {"✅ Confirmed" if setup_found["vol_spike"] else "❌"}

━━━━━━━━━━━━━━━━━━━━

📐 <b>ENTRY ZONE</b>
Zone Top    : ${levels['zone_top']:,.4f}
Zone Bottom : ${levels['zone_bottom']:,.4f}

🟡 <b>MODERATE ⭐ RECOMMENDED</b>
Entry : ${levels['moderate']['entry']:,.4f}
SL    : ${levels['moderate']['sl']:,.4f}
TP1/2/3 : ${levels['tp1']:,.4f} / ${levels['tp2']:,.4f} / ${levels['tp3']:,.4f}
R/R   : {levels['moderate']['rr']:.1f}x

🔴 <b>AGGRESSIVE</b>
Entry : ${levels['aggressive']['entry']:,.4f}
SL    : ${levels['aggressive']['sl']:,.4f}
TP1/2/3 : ${levels['tp1']:,.4f} / ${levels['tp2']:,.4f} / ${levels['tp3']:,.4f}
R/R   : {levels['aggressive']['rr']:.1f}x

━━━━━━━━━━━━━━━━━━━━
⚠️ <i>Verify on exchange before entry. Scanned: {now_str} UTC</i>
"""
            send_telegram(msg)
            mark_alert_sent(cache, ticker, style)
            print(f"🚨 ALERT SENT: {ticker} ({style}) @ ${current_price:,.4f}")

        except Exception as e:
            print(f"❌ Error {ticker} {style}: {e}")

# ==========================================
# MAIN
# ==========================================
def main():
    print(f"\n{'='*50}")
    print(f"BBMA Scanner Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode: BTC Filter={BTC_FILTER_MODE} | Cooldown={ALERT_COOLDOWN_HOURS}h | Vol spike={VOLUME_SPIKE_MULTIPLIER}x")
    print(f"{'='*50}")

    cache = load_cache()

    for ticker in PAIRS:
        try:
            scan_pair(ticker, cache)
        except Exception as e:
            print(f"❌ Fatal error {ticker}: {e}")

    save_cache(cache)
    print(f"\n{'='*50}")
    print("Scan Complete")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    main()
