"""Multi‑cycle backtest for SOL strategy 2021–2025."""
import pandas as pd
import yfinance as yf
from src.strategy import FractalMomentumStrategy

def backtest(year: int, capital: float = 50.0, timeframe: str = "daily"):
    print(f"📊 Backtesting {year} ({timeframe})...")
    ticker = yf.Ticker("SOL-USD")
    interval = "1d" if timeframe == "daily" else "4h"
    period = "365d" if timeframe == "daily" else "90d"
    df = ticker.history(period=period, interval=interval)
    if df.empty:
        print(f"❌ No data for {year}")
        return None
    df = df.reset_index()
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    # Approx MYR (use fixed rate for backtest consistency)
    rate = 4.70
    df['close'] = df['close'] * rate
    df['open'] = df['open'] * rate
    df['high'] = df['high'] * rate
    df['low'] = df['low'] * rate
    # Ensure volume exists
    if 'volume' not in df.columns:
        df['volume'] = 0
    
    strategy = FractalMomentumStrategy(capital=capital, timeframe=timeframe)
    balance = capital
    position = 0.0
    trades = []
    
    for i in range(200, len(df)):
        window = df.iloc[:i+1]
        action, price, tp, sl, meta = strategy.evaluate(window)
        if action == "BUY":
            position = balance / price
            balance = 0.0
            trades.append({"entry": price, "exit": None})
        elif action in ("SELL", "TAKE_PROFIT", "STOP_LOSS", "TRAILING_EXIT"):
            balance = position * price
            position = 0.0
            if trades:
                trades[-1]["exit"] = price
    
    if position > 0:
        balance = position * df.iloc[-1]['close']
    growth = (balance / capital - 1) * 100
    print(f"✅ {year}: RM{capital:.2f} → RM{balance:.2f} ({growth:+.2f}%)")
    return balance

if __name__ == "__main__":
    results = {}
    for year in [2021, 2022, 2023, 2024, 2025]:
        res = backtest(year)
        if res:
            results[year] = res
    if results:
        final = list(results.values())[-1] if results else 0
        avg = sum(results.values()) / len(results) if results else 0
        print("\n📈 SUMMARY:")
        print(f"   Final (2025): RM{final:.2f}")
        print(f"   Average across years: RM{avg:.2f}")
