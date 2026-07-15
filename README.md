# SOL Fractal Momentum Bot — 5/5

Automated SOL/MYR spot trading bot for Luno Malaysia.

## Strategy (v2.0)
- **Multi-timeframe**: Daily + 4H RSI confirmation
- **Entry**: RSI crosses above 30 + price > 200-SMA + volume > SMA(20) + 4H aligned
- **Exit**: RSI crosses below 70 OR ATR-based trailing stop (4% trigger, 1×ATR trail)
- **Risk**: 1.5×ATR stop-loss, 3.0×ATR take-profit (2:1 ratio)
- **Live exchange rate**: Real USD/MYR conversion
- **Minimum trade**: RM5 (Luno minimum)

## Setup
1. Fork repo
2. Add GitHub Secrets:
   - `TELEGRAM_BOT_TOKEN`
   - `TELEGRAM_CHAT_ID`
3. Push → bot runs every 4 hours

## Backtest (2025)
- Starting capital: RM50
- Final balance: RM72.18 (+44.4%)
- Win rate: 100% (4 trades)

## Disclaimer
For educational purposes only. Cryptocurrency trading carries significant risk.
