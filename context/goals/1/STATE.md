# ArBos Trading System - State

## Summary
BTC 15-minute directional prediction system with evolutionary model search and walk-forward validation. Target: >=65% accuracy, >90 bets/month, positive Sharpe.

## Current Status: ⚠️ Step 102 – Signal research complete, structural limits identified

### CRITICAL: Data Leakage Discovered & Fixed (Step 85)
**All results from steps 39–83 were invalidated by data leakage in `generate_horizon_features`.**

**Leakage mechanism:** When resampling 15m bars to 1h (factor=4), the 1h candle's close = T:45 bar's close. This resampled bar was merged at `open_time=T:00` and forward-filled to T:15, T:30, T:45. Technical indicators (RSI, MACD, BB) on the 1h bar then **encoded the T:45 close** for the T:30 bar whose target was `T:45 > T:30`. Equivalent leakage existed for 2h, 4h, 24h horizons.

**Fix applied (`horizon.py`):** Shift resampled bar `open_time` forward by one horizon period before merging:
```python
df_resampled["open_time"] += pd.Timedelta(minutes=horizon_minutes)
```

**Confirmation:** Accuracy dropped 94% → 56.4% after fix — proving prior results were entirely leak-driven.

### Signal Research Summary (Steps 85-102)

The mean-reversion signal (large down bar + low taker_buy → predict UP next bar) was extensively tuned:

| Step | Signal Parameters | Accuracy | Bets/month | Sharpe (taker) | Notes |
|------|-----------------|----------|------------|----------------|-------|
| 85 | Pure ML (LightGBM) | 56.4% | 253 | -9.07 | Baseline after leakage fix |
| 91 | 85th pct, 0.40/0.60, no RSI | 58.47% | 259 | -11.6 | ML agreement filter ineffective |
| 92 | 90th pct, 0.35/0.65, RSI 45/55 | 63.33% | 72 | -5.1 | High acc, below 90/month |
| 93 | **87th pct, 0.38/0.62, RSI 45/55** | **61.99%** | **141** | **-7.84** | **Best balance** |
| 94 | +trend_regime_1h filter | 62.93% | 56 | -5.0 | Too few signals |
| 95 | 90th pct, 0.35/0.65, RSI, buy_pressure_change | 70.37% | 13 | -0.47 | Too few signals |
| 96 | +BB position < 0.20 | 61.69% | 126 | -7.38 | Similar to step 93 |
| 99 | Step 93 + maker fee (0.02%) | 61.99% | 141 | -3.93 | Maker fee helps, still negative |
| 101 | Step 95 conditions + maker fee | 87.5% | 4 | +4.63 | Statistically noise (8 trades) |
| **102** | **Step 93 conditions + maker fee** | **61.99%** | **141** | **-3.93** | **Current config** |

### Current Configuration (Step 102)
- **Signal:** Large down bar (87th pct) + taker_buy < 0.38 + RSI < 45 → predict UP; symmetric for DOWN
- **Model type:** LightGBM only (`MODEL_TYPES=["lightgbm"]`)
- **Fee:** FEE_RATE=0.0002 (maker, 0.02%/side — mean-reversion uses limit orders for entry)
- **Hold:** 1-bar (15 min) — mean-reversion is a 1-bar phenomenon
- **Horizons:** 15m + 30m + 1h (3 horizons, proper time-shifted to avoid leakage)
- **Features:** 212 features including order flow (taker_buy_ratio, vwap_dev, etc.) + futures (funding rate, L/S, OI)

### Goal Status: NOT ACHIEVED
| Metric | Result | Target | Status |
|--------|--------|--------|--------|
| Accuracy | 61.99% (combined WFV) | ≥65% | ❌ (-3%) |
| Bets/month | 141 | >90 | ✅ |
| Sharpe | -3.93 | >0 | ❌ |

### Root Cause of Negative Sharpe

**Mean-reversion has negative return skew:**
- When correct (62%): avg win ≈ 0.17% - 0.06% fee = **+0.11%**
- When wrong (38%): avg loss ≈ 0.17% + 0.06% fee = **-0.23%**
- Win/Loss ratio = 0.11/0.23 = 0.48 < 1.0

**Break-even accuracy formula:** p = (avg_move + fee) / (2 × avg_move)
- With avg_next_bar_move=0.17%, maker_fee=0.06%: break-even = 0.67 = **67.0%** accuracy needed
- Current best: 62% combined, 66% in best folds
- Gap: ~5% accuracy to break even

**The accuracy-count tradeoff:**
- Tighter signal conditions → higher accuracy (70%+) but fewer signals (< 13/month, below 90 target)
- Looser conditions → more signals (141/month) but lower accuracy (62%)
- No parameter combination found that achieves BOTH ≥65% accuracy AND ≥90/month AND positive Sharpe

### Why Fold 2 is Consistently Poor

The 3-fold WFV covers approximately Nov 2025 - Mar 2026. Fold 2 (~Feb 5-26, 2026) consistently shows 57-59% accuracy regardless of signal tuning. This period likely involved high-volatility whipsaw market action following BTC's pullback from January 2026 highs. In such markets:
- Mean-reversion fails (trend continuation wins)
- Trend-following also fails (no persistence)
- Any strategy underperforms

### Recommended Next Steps

To achieve positive Sharpe with ≥65% accuracy and ≥90 bets/month:

1. **Order book data**: Bid-ask spread, depth at each level — genuine microstructure signal without the negative skew problem
2. **Trend-following switch**: Positive skew (winners run, losers stopped quickly) works with < 50% accuracy
3. **Cross-asset signals**: ETH/BTC spread, USDT market cap flow, DXY, fear&greed index
4. **News/sentiment**: NLP on crypto Twitter/Reddit for regime detection
5. **Longer timeframe**: 4h or 1D bars have larger moves relative to fees (break-even accuracy drops significantly)

### Key Lessons Learned (Steps 85-102)
1. **Data leakage via resampled horizons**: future close encoded in multi-timeframe features when merge timing not offset → always shift by 1 period before merge
2. **Mean-reversion on 15m BTC has negative skew**: small wins, larger losses → requires very high accuracy (>67%) to be profitable at realistic fees
3. **ML models learn momentum**: adding ML agreement filter to mean-reversion rule HURTS (ML filters out good mean-reversion signals)
4. **Fee economics dominate at 15m**: 0.06% round-trip maker fee = 35% of a typical next-bar return (0.17%), making break-even very hard
5. **Stop-losses fire on correct trades**: BTC intrabar volatility frequently touches 0.3%+ stops even for ultimately profitable trades
6. **Fold 2 (Feb 2026) is a structural drag**: whipsaw market always produces poor mean-reversion accuracy; combined 3-fold avg ≈ 62%
7. **Statistical noise at low count**: signals with <20 trades per fold produce unreliable accuracy estimates (87.5% on 8 trades is noise)

### Evolution Trajectory
| Step | Key Change | Accuracy | Notes |
|------|-----------|----------|-------|
| 39 | RF+ET+GBM, 31 feat | 82.35% | ⚠️ leakage |
| 72 | LightGBM + 448 feat | 88.75% | ⚠️ leakage |
| 80 | MIN_CONF=0.92 + full HPs | 95.07% | ⚠️ leakage |
| **85** | **Leak fixed** | **56.4%** | ✅ clean baseline |
| 91 | Rule-based mean-reversion | 58.47% | ✅ clean, 259/month |
| 92 | Tighter thresholds + RSI | 63.33% | ✅ highest acc, 72/month |
| 93 | Balanced parameters | 61.99% | ✅ 141/month |
| **102** | **Maker fees, best signal** | **61.99%** | ✅ current best |
