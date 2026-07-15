"""Fractal Momentum Strategy v2.0 — 5/5."""
import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict
from src.indicators import calculate_rsi, calculate_sma, calculate_atr, fetch_4h_rsi

class FractalMomentumStrategy:
    def __init__(
        self,
        capital: float = 50.0,
        rsi_period: int = 14,
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
        atr_sl_mult: float = 1.5,
        atr_tp_mult: float = 3.0,
        trail_trigger_pct: float = 4.0,
        trail_step_atr: float = 1.0,
        volume_ma_period: int = 20,
        timeframe: str = "daily",
        min_trade_rm: float = 5.0
    ):
        self.capital = capital
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.atr_sl_mult = atr_sl_mult
        self.atr_tp_mult = atr_tp_mult
        self.trail_trigger_pct = trail_trigger_pct
        self.trail_step_atr = trail_step_atr
        self.volume_ma_period = volume_ma_period
        self.timeframe = timeframe
        self.min_trade_rm = min_trade_rm
        
        self.position_open = False
        self.entry_price = 0.0
        self.entry_date = None
        self.highest_price = 0.0
        self.trailing_active = False
        self.stop_loss = 0.0
        self.take_profit = 0.0
    
    def evaluate(self, df: pd.DataFrame) -> Tuple[Optional[str], Optional[float], Optional[float], Optional[float], Dict]:
        if df is None or len(df) < 200:
            return (None, None, None, None, {"error": "Insufficient data", "df_len": len(df) if df is not None else 0})
        
        df = df.copy()
        df['rsi'] = calculate_rsi(df['close'], self.rsi_period)
        df['sma_200'] = calculate_sma(df['close'], 200)
        df['atr'] = calculate_atr(df, 14)
        df['volume_sma'] = df['volume'].rolling(self.volume_ma_period).mean()
        
        idx = -2 if len(df) >= 2 else -1
        curr = df.iloc[idx]
        prev = df.iloc[idx-1] if idx-1 >= 0 else curr
        
        current_price = curr['close']
        current_rsi = curr['rsi']
        prev_rsi = prev['rsi']
        current_atr = curr['atr']
        volume = curr['volume']
        vol_sma = curr['volume_sma']
        sma_200 = curr['sma_200']
        
        if self.capital < self.min_trade_rm:
            return (None, None, None, None, {"error": f"Insufficient capital: RM{self.capital:.2f} < RM{self.min_trade_rm:.2f}"})
        
        volume_ok = volume > vol_sma if not pd.isna(vol_sma) else True
        
        four_h_ok = True
        if self.timeframe == "daily":
            rsi_4h = fetch_4h_rsi()
            if rsi_4h is not None:
                if current_rsi < self.rsi_oversold:
                    four_h_ok = rsi_4h > self.rsi_oversold
                else:
                    four_h_ok = True
        
        trend = "BULLISH" if current_price > sma_200 else "BEARISH"
        
        if self.position_open:
            if current_price >= self.take_profit:
                self.position_open = False
                return ("TAKE_PROFIT", current_price, None, None, {"rsi": current_rsi, "trend": trend, "atr": current_atr})
            if current_price <= self.stop_loss:
                self.position_open = False
                return ("STOP_LOSS", current_price, None, None, {"rsi": current_rsi, "trend": trend, "atr": current_atr})
            if current_price >= self.entry_price * (1 + self.trail_trigger_pct / 100):
                self.trailing_active = True
            if self.trailing_active:
                if current_price > self.highest_price:
                    self.highest_price = current_price
                trail_level = self.highest_price - (self.trail_step_atr * current_atr)
                if current_price <= trail_level:
                    self.position_open = False
                    return ("TRAILING_EXIT", current_price, None, None, {"rsi": current_rsi, "trend": trend, "atr": current_atr})
            if prev_rsi > self.rsi_overbought and current_rsi <= self.rsi_overbought:
                self.position_open = False
                return ("SELL", current_price, None, None, {"rsi": current_rsi, "trend": trend, "atr": current_atr})
            if current_price < sma_200:
                self.position_open = False
                return ("SELL", current_price, None, None, {"rsi": current_rsi, "trend": trend, "atr": current_atr})
            return (None, None, None, None, {"rsi": current_rsi, "trend": trend, "atr": current_atr, "position": "open"})
        
        buy_signal = (
            prev_rsi < self.rsi_oversold and
            current_rsi >= self.rsi_oversold and
            current_price > sma_200 and
            volume_ok and
            four_h_ok
        )
        
        if buy_signal:
            self.entry_price = current_price
            self.entry_date = str(curr.get('date', ''))
            self.highest_price = current_price
            self.trailing_active = False
            self.stop_loss = current_price - (self.atr_sl_mult * current_atr)
            self.take_profit = current_price + (self.atr_tp_mult * current_atr)
            self.position_open = True
            
            confidence = "HIGH" if (current_rsi - self.rsi_oversold) > 3 else "MEDIUM"
            return ("BUY", current_price, self.take_profit, self.stop_loss, {
                "rsi": current_rsi,
                "prev_rsi": prev_rsi,
                "trend": trend,
                "atr": current_atr,
                "volume_ok": volume_ok,
                "four_h_ok": four_h_ok,
                "confidence": confidence,
                "sma_200": sma_200,
                "high": curr['high'],
                "low": curr['low']
            })
        
        return (None, None, None, None, {
            "rsi": current_rsi,
            "prev_rsi": prev_rsi,
            "trend": trend,
            "atr": current_atr,
            "volume_ok": volume_ok,
            "four_h_ok": four_h_ok,
            "sma_200": sma_200,
            "high": curr['high'],
            "low": curr['low'],
            "position": "closed"
        })
