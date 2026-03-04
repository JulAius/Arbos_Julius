Your goal is in `GOAL.md` read it. 

You are running inside of a git repository on a computer.

You have access to the env variables in .env

You are fed this prompt over and over again in steps, each step you are asked to plan and then execute that plan using cursors agent harness. You can read `agent.py` to fully understand how you are working.

Each time you are run, each step, your plan and execution rollouts are stored in history/<timestamp>/ under `plan.md` and `rollout.md`. The logs from the execution of your running are also found there under `logs.txt`. 

It is IMPORTANT to remember that at the beginning of each step you are fed this file. Therefore you are welcome to edit this file to pass yourself hints. Be kind to your later self and make your job easier by passing yourself information in this way.  Be careful about your context length.

Try to keep things clean when achieving your goal. Put the files you write in the correct places preferrably in the latest history folder is they are temporary. Think long term.

Your goal is in `GOAL.md` read it, and execute it. Dont stop.

---

## Current Status (Updated: 2026-03-04)

**Built Components:**
- `coinglass_client.py` - CoinGlass API client for funding rates, OI, liquidations (with fallback to Hyperliquid)
- `hyperliquid_client.py` - Hyperliquid SDK wrapper with price fetching, position management, order placement, retry logic
- `strategy.py` - Trading strategy primarily using Hyperliquid funding rates + price momentum analysis
- `trader.py` - Main trading loop with stop-loss/take-profit, trailing stops, max drawdown protection
- `performance.py` - Trade logging and performance tracking
- `main.py` - Entry point script with configurable parameters

**Features Implemented:**
- ✅ Accurate price fetching for position sizing with retry logic
- ✅ Stop-loss and take-profit logic (default 2% SL, 5% TP)
- ✅ **Trailing stop-loss** (default 1.5%) - automatically trails profitable positions
- ✅ **Max drawdown protection** (default 20%) - stops trading if drawdown exceeds threshold
- ✅ Position closing on signal reversal or SL/TP/trailing stop triggers
- ✅ Enhanced error handling with retry logic (3 attempts for critical calls)
- ✅ Primary reliance on Hyperliquid funding rates (more reliable than CoinGlass)
- ✅ Price momentum analysis using recent price history
- ✅ Proper PnL calculation and trade tracking
- ✅ Position high tracking for trailing stop functionality

**Strategy Improvements:**
- Strategy now primarily uses Hyperliquid funding rates (60% weight) - more reliable
- Added price momentum analysis (30% weight) for trend confirmation
- **Lower confidence threshold (0.4)** - captures more trading opportunities
- **Lower funding rate threshold (0.005% vs 0.01%)** - more sensitive to funding rate signals
- **More aggressive strength scaling** - better captures funding rate opportunities
- Better position sizing based on confidence scores
- Minimum position size of $10 to ensure trades are meaningful

**Risk Management:**
- Trailing stop-loss: Automatically adjusts stop-loss as position becomes profitable
- Max drawdown: Stops trading if account drops 20% from peak value
- Position size limits: Max 15% of account per position (scaled by confidence)
- Multiple exit conditions: SL, TP, trailing stop, signal reversal, confidence drop

**Usage:**
```bash
# Default: BTC,ETH,SOL with 5min interval, 2% SL, 5% TP, 1.5% trailing stop, 20% max drawdown
python main.py

# Custom: python main.py [coins] [interval_sec] [stop_loss_pct] [take_profit_pct] [trailing_stop_pct] [max_drawdown_pct]
python main.py BTC,ETH 300 0.02 0.05 0.015 0.20
```

**Recent Improvements (2026-03-04):**
- ✅ Fixed funding rate fetching - now correctly retrieves rates from Hyperliquid
- ✅ Lowered funding rate threshold from 0.01% to 0.005% for more signals
- ✅ Lowered confidence threshold from 0.5 to 0.4 for more opportunities
- ✅ Improved strength calculation (2000x scaling vs 1000x) for better signal sensitivity
- ✅ System tested and verified working - generates signals correctly

**Known Issues:**
- CoinGlass API endpoints return 404 - strategy works with Hyperliquid data (primary source)
- Account value is $0.00 - need funds in account to trade (testnet or mainnet)
- WebSocket connection limit warnings - non-critical, SDK internal connections

**Next Steps:**
1. **Fund account** - System is ready to trade once account is funded
2. Monitor strategy performance and adjust confidence thresholds if needed
3. Fine-tune trailing stop percentage based on volatility
4. Consider adding more sophisticated momentum indicators
5. WebSocket warnings are non-critical (SDK internal) - can be ignored
