"""Luno API client with robust fallback."""
import os
import requests
import time
import json
from typing import Optional, List, Dict
from datetime import datetime

class LunoClient:
    BASE_URL = "https://api.luno.com/api/1"
    EXCHANGE_RATE_URL = "https://api.exchangerate.host/convert"
    
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        self.api_key = api_key or os.getenv("LUNO_API_KEY")
        self.api_secret = api_secret or os.getenv("LUNO_API_SECRET")
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    
    def _auth(self):
        if self.api_key and self.api_secret:
            return (self.api_key, self.api_secret)
        return None
    
    def get_usd_to_myr(self) -> float:
        """Get live USD/MYR exchange rate with fallback."""
        try:
            resp = self.session.get(self.EXCHANGE_RATE_URL, params={"from": "USD", "to": "MYR", "amount": 1}, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("success"):
                    return float(data["result"])
        except Exception as e:
            print(f"Exchange rate error: {e}")
        # Try alternative API
        try:
            resp = requests.get("https://api.frankfurter.app/latest?from=USD&to=MYR", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                return float(data["rates"]["MYR"])
        except:
            pass
        # Fallback
        return 4.70
    
    def get_ticker(self, pair: str = "SOLMYR") -> Dict:
        """Get current ticker price."""
        url = f"{self.BASE_URL}/ticker"
        params = {"pair": pair}
        auth = self._auth()
        for attempt in range(3):
            try:
                resp = self.session.get(url, params=params, auth=auth, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                return {
                    "pair": pair,
                    "price": float(data.get("last_trade", 0)),
                    "bid": float(data.get("bid", 0)),
                    "ask": float(data.get("ask", 0)),
                    "volume": float(data.get("rolling_24_hour_volume", 0)),
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                print(f"Ticker error (attempt {attempt+1}): {e}")
                time.sleep(2)
        raise Exception(f"Failed to fetch ticker for {pair}")
    
    def get_candles(self, pair: str = "SOLMYR", duration: int = 86400, limit: int = 500) -> List[Dict]:
        """Get candlestick data."""
        url = f"{self.BASE_URL}/candles"
        params = {"pair": pair, "duration": duration, "limit": limit}
        auth = self._auth()
        for attempt in range(3):
            try:
                resp = self.session.get(url, params=params, auth=auth, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                candles = data.get("candles", [])
                result = []
                for c in candles:
                    result.append({
                        "timestamp": int(c.get("timestamp", 0)),
                        "open": float(c.get("open", 0)),
                        "high": float(c.get("high", 0)),
                        "low": float(c.get("low", 0)),
                        "close": float(c.get("close", 0)),
                        "volume": float(c.get("volume", 0))
                    })
                return result
            except Exception as e:
                print(f"Candles error (attempt {attempt+1}): {e}")
                time.sleep(2)
        # Fallback: return empty list
        return []
