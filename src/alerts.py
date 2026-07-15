"""Telegram alert system."""
import os
import requests
from datetime import datetime
from typing import Optional, Dict

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

def send_telegram_alert(message: str, parse_mode: str = "Markdown") -> bool:
    """Send alert via Telegram bot with retry."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Telegram credentials missing. Printing alert to console:")
        print(message)
        return False
    
    if ":" not in TELEGRAM_BOT_TOKEN:
        print("❌ Invalid Telegram token format")
        return False
    
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": parse_mode,
        "disable_web_page_preview": True
    }
    
    for attempt in range(3):
        try:
            resp = requests.post(url, json=payload, timeout=30)
            if resp.status_code == 200:
                print("✅ Telegram alert sent")
                return True
            elif resp.status_code == 429:
                retry_after = resp.json().get("parameters", {}).get("retry_after", 5)
                print(f"⏳ Rate limited, waiting {retry_after}s...")
                import time
                time.sleep(retry_after)
            elif resp.status_code == 401:
                print("❌ Telegram token invalid")
                return False
            else:
                print(f"❌ Telegram error {resp.status_code}: {resp.text[:200]}")
                if attempt < 2:
                    import time
                    time.sleep(3)
        except Exception as e:
            print(f"❌ Telegram exception: {e}")
            if attempt < 2:
                import time
                time.sleep(3)
    return False

def format_signal_alert(
    action: str,
    price: float,
    tp: Optional[float] = None,
    sl: Optional[float] = None,
    meta: Optional[Dict] = None
) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M MYT")
    
    emoji_map = {
        "BUY": "🟢",
        "SELL": "🔴",
        "TAKE_PROFIT": "✅",
        "STOP_LOSS": "❌",
        "TRAILING_EXIT": "📊"
    }
    emoji = emoji_map.get(action, "📌")
    
    action_names = {
        "BUY": "BUY",
        "SELL": "SELL",
        "TAKE_PROFIT": "TAKE PROFIT",
        "STOP_LOSS": "STOP LOSS",
        "TRAILING_EXIT": "TRAILING EXIT"
    }
    action_name = action_names.get(action, action)
    
    message = f"""
{emoji} *SOL/MYR | {action_name} | RM {price:,.2f}*
📅 {timestamp}

📊 RSI(14): `{meta.get('rsi', 0):.1f}` | Trend: {meta.get('trend', 'N/A')}
"""
    
    if meta.get('prev_rsi') is not None:
        message += f"📉 Previous RSI: `{meta['prev_rsi']:.1f}`\n"
    
    if meta.get('confidence'):
        message += f"💪 Confidence: {meta['confidence']}\n"
    
    if tp:
        message += f"🎯 Take-Profit: RM {tp:,.2f} (+{((tp/price)-1)*100:.1f}%)\n"
    if sl:
        message += f"🛑 Stop-Loss: RM {sl:,.2f} ({((sl/price)-1)*100:.1f}%)\n"
    
    if meta.get('atr'):
        message += f"📈 ATR: RM {meta['atr']:.2f} ({meta['atr']/price*100:.2f}%)\n"
    
    if meta.get('volume_ok') is not None:
        message += f"🔊 Volume: {'✅ Confirmed' if meta['volume_ok'] else '⚠️ Below Avg'}\n"
    if meta.get('four_h_ok') is not None:
        message += f"🕓 4H RSI: {'✅ OK' if meta['four_h_ok'] else '⚠️ Not aligned'}\n"
    
    if meta.get('sma_200'):
        message += f"📊 200-SMA: RM {meta['sma_200']:,.2f}\n"
    
    if meta.get('high') and meta.get('low'):
        message += f"📊 Range: RM {meta['low']:,.2f} - RM {meta['high']:,.2f}\n"
    
    message += "\n#SOL #Luno #TradingSignal"
    return message

def format_heartbeat(price: float, meta: Dict, state: Dict) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M MYT")
    last_signal = state.get('last_signal', 'None')
    last_date = state.get('last_signal_date', '-')
    
    message = f"""
💓 *SOL/MYR | No Signal | RM {price:,.2f}*
📅 {timestamp}

📊 RSI(14): `{meta.get('rsi', 0):.1f}` | Trend: {meta['trend']}
📡 Last Signal: {last_signal} ({last_date})
🔢 Runs: {state.get('run_count', 0)}

✅ Bot healthy. Monitoring every 4 hours.
"""
    return message

def send_startup_notification():
    message = """
🚀 *SOL Fractal Momentum Bot 5/5 Started*

📊 Asset: SOL/MYR (Luno Malaysia)
🎯 Strategy: Fractal Momentum v2.0
📈 Entry: RSI crosses ABOVE 30 + Trend + Volume + 4H confirmation
📉 Exit: RSI crosses BELOW 70 OR Trailing Stop

🔧 SL: 1.5× ATR | TP: 3.0× ATR
📊 Trailing: +4% trigger, 1× ATR trail
💱 Live USD/MYR conversion

📈 Backtest (2025): +44.4% on RM50 | 100% win rate
⏰ {datetime.now().strftime('%Y-%m-%d %H:%M MYT')}
"""
    send_telegram_alert(message)

def send_health_report(state: Dict):
    history = state.get('signal_history', [])
    trades = []
    entry = None
    for h in history:
        if h.get('signal') == 'BUY':
            entry = h
        elif h.get('signal') in ('SELL', 'TAKE_PROFIT', 'STOP_LOSS', 'TRAILING_EXIT') and entry:
            pnl = ((h['price'] - entry['price']) / entry['price']) * 100
            trades.append(pnl)
            entry = None
    
    stats = ""
    if trades:
        wins = sum(1 for t in trades if t > 0)
        win_rate = (wins / len(trades)) * 100
        avg_pnl = sum(trades) / len(trades)
        total_pnl = sum(trades)
        stats = f"""
📈 *Trade Stats:*
   Win Rate: {win_rate:.1f}% ({wins}/{len(trades)})
   Avg P&L: {avg_pnl:+.2f}%
   Total P&L: {total_pnl:+.2f}%
   Total Trades: {len(trades)}"""
    else:
        stats = "\n📈 No completed trades yet."
    
    message = f"""
🏥 *Health Report*

🔢 Runs: {state.get('run_count', 0)}
📡 Last Signal: {state.get('last_signal', 'None')}
💰 Last Price: RM {state.get('last_price', 0):,.2f}
❌ Errors: {state.get('error_count', 0)}
{stats}

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M MYT')}
"""
    send_telegram_alert(message)
