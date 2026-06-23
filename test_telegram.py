import os
import requests
from datetime import datetime

# ==========================================
# CONFIGURATION (Read from GitHub Secrets)
# ==========================================
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')

def test_telegram():
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("❌ ERROR: TELEGRAM_BOT_TOKEN atau TELEGRAM_CHAT_ID tidak dijumpai dalam GitHub Secrets!")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    
    # Mesej test
    msg = f"""
✅ <b>TEST TELEGRAM BERJAYA!</b>

Bot BBMA Crypto Scanner anda telah disambungkan.
Sekarang anda boleh deploy code scanner yang sebenar.

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    """
    
    payload = {'chat_id': TELEGRAM_CHAT_ID, 'text': msg, 'parse_mode': 'HTML'}
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 200:
            print("✅ SUCCESS: Mesej test berjaya dihantar ke Telegram!")
        else:
            print(f"❌ FAILED: Telegram API reject. Status: {response.status_code}")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"❌ FAILED: Network error. {e}")

if __name__ == "__main__":
    test_telegram()

