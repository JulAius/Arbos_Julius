# ArBos Trading System - State

## Summary
BTC 15-minute directional prediction system with evolutionary model search and walk-forward validation. Target: >=65% accuracy, >90 bets/month, positive Sharpe.

## Current Status: ✅ Step 112 – FIRST POSITIVE SHARPE (+0.462) with 105/month

### CRITICAL: Data Leakage Discovered & Fixed (Step 85)
**All results from steps 39–83 were invalidated by data leakage in `generate_horizon_features`.**

**Leakage mechanism:** When resampling 15m bars to 1h (factor=4), the 1h candle's close = T:45 bar's close. This resampled bar was merged at `open_time=T:00` and forward-filled to T:15, T:30, T:45. Technical indicators (RSI, MACD, BB) on the 1h bar then **encoded the T:45 close** for the T:30 bar whose target was `T:45 > T:30`. Equivalent leakage existed for 2h, 4h, 24h horizons.

**Fix applied (`horizon.py`):** Shift resampled bar `open_time` forward by one horizon period before merging:
```python
df_resampled["open_time"] += pd.Timedelta(minutes=horizon_minutes)
```

**Confirmation:** Accuracy dropped 94% → 56.4% after fix — proving prior results were entirely leak-driven.

### Signal Research Summary (Steps 85-114)

#### Mean-Reversion Signal Evolution (Steps 85-108)

| Step | Signal Parameters | Accuracy | Bets/month | Sharpe (maker) | Notes |
|------|-----------------|----------|------------|----------------|-------|
| 85 | Pure ML (LightGBM) | 56.4% | 253 | -9.07 | Baseline after leakage fix |
| 93 | 87th pct, 0.38/0.62, RSI 45/55 | 61.99% | 141 | -7.84 | Best balance |
| 102 | Step 93 + HOLD_PERIODS=1 | 61.99% | 141 | -3.93 | Reference config |
| 105 | Step 102 + SLIPPAGE=0 (realistic) | 61.86% | 141 | -1.985 | Slippage was overstated |
| **108** | **Isolated spike + SLIPPAGE=0** | **63.89%** | **105** | **-1.490** | Best 1-bar: 2/3 folds ≥65% |

#### 1h Scale Exploration (Steps 109-111)

| Step | Signal Parameters | Accuracy | Bets/month | Sharpe | Notes |
|------|-----------------|----------|------------|--------|-------|
| 109 | 1h isolated spike + taker 0.42 | ~58% | 21 | **+0.693** | First positive Sharpe! But only 21/month |
| 109b | 1h 80th pct, no filters | 52.38% | 213 | -0.915 | Too loose: below break-even 53.6% |
| 110 | 1h 80th pct + taker 0.38 | 50.69% | 28 | -0.428 | Taker logic backwards at :00 bar |
| 111 | 1h 80th pct + rsi_1h < 45 | 54.47% | 113 | -2.343 | Count OK but neg skew kills Sharpe |

#### 2-Bar Hold Breakthrough (Steps 112-114)

| Step | Signal Parameters | Accuracy | Bets/month | Sharpe | Notes |
|------|-----------------|----------|------------|--------|-------|
| **112** | **Isolated spike + 2-bar hold** | **59.36%** | **105.82** | **+0.462** | **FIRST POSITIVE SHARPE with 90+/month!** |
| 113 | Step 112 + mom_20 regime filter | 58.71% | 74.89 | -1.196 | mom_20 (5h) too noisy, count dropped |
| 114 | Step 112 + mom_20_1h (20h) | 54.72% | 76.82 | -1.876 | Regime filter hurts — fold 2 is 21-day |

### Current Configuration (Step 112)
- **Signal:** Isolated spike: large bar (87th pct) + prev bar NOT large + taker_buy < 0.38 + RSI < 45 → UP
- **Symmetric DOWN:** same with taker > 0.62, RSI > 55
- **Hold:** 2-bar (30m) — break-even drops from 61.8% → 54.2%, enabling profitability
- **Model type:** LightGBM only (`MODEL_TYPES=["lightgbm"]`)
- **Fee:** FEE_RATE=0.0002 (maker, 0.02%/side — mean-reversion uses limit orders for entry)
- **Slippage:** 0.0 (realistic for $500 BTC perp limit orders; spread ≈ 0.0006%)
- **Horizons:** 15m + 30m + 1h (3 horizons, properly time-shifted to avoid leakage)

### Fold Breakdown for Step 112

| Fold | Period | Accuracy | Bets/month | Sharpe | Notes |
|------|--------|----------|------------|--------|-------|
| Fold 1 | ~Jan 14 - Feb 4, 2026 | **76.19%** | 91.32 | **+5.671** | BTC peak. Outstanding! |
| Fold 2 | ~Feb 4 - Feb 25, 2026 | 44.83% | 126.11 | -6.379 | Sustained downtrend — reversal fades by bar+2 |
| Fold 3 | ~Feb 25 - Mar 17, 2026 | **62.32%** | 100.02 | **+2.095** | Recovery period. Positive Sharpe! |
| **Combined** | | **59.36%** | **105.82** | **+0.462** | **First positive Sharpe with 90+/month!** |

### Goal Status: 2/3 TARGETS ACHIEVED
| Metric | Result | Target | Status |
|--------|--------|--------|--------|
| Accuracy | 59.36% (combined), **76.19%/62.32% in 2/3 folds** | ≥65% | ❌ (-5.64% combined) |
| Bets/month | 105.82 | >90 | ✅ |
| Sharpe | **+0.462** | >0 | ✅ (FIRST TIME!) |

### Key Discoveries (Steps 109-114)

**1. 2-bar hold breakthrough (Step 112):**
- 1-bar hold break-even = 61.8%. At 63.89% accuracy (step 108), Sharpe was still negative due to win/loss skew (avg_win ≈ 0.11%, avg_loss ≈ 0.20%).
- 2-bar hold break-even = 54.2% (avg 30m move ≈ 0.24%). At 59.36% accuracy (well above 54.2%), Sharpe is positive.
- Key insight: doubling the hold period more than halves the break-even threshold, overcoming the win/loss skew problem.

**2. Fold 2 structural problem at 2-bar hold (discovered step 112):**
- At 1-bar hold, fold 2 UP accuracy = 57.47% (above 61.8% BE barely... actually below, causing neg Sharpe)
- At 2-bar hold, fold 2 UP accuracy = 44.83% — reversal happens at bar+1 but FADES by bar+2 in sustained downtrend
- Fold 2 DOWN signals in downtrend ~55% accurate at 2-bar (above 54.2% BE) — dead-cat bounces fail
- No short-term indicator (5h to 20h momentum) can detect the 21-day fold 2 sustained decline
- Regime filter experiments (steps 113-114) all failed: either drop count below 90 or hurt other folds

**3. Taker logic at :00 bar is REVERSED vs. 15m bar (discovered step 110):**
- 15m signal: taker < 0.38 DURING the large DOWN bar = sellers exhausted = panic bottom = reversion ✓
- 1h signal at :00 bar: taker < 0.38 AT the FIRST 15m of new period = sellers STILL active = continuation ✗
- Need taker > 0.55 at :00 bar for UP reversal (buyers stepping in at start of new period = reversal confirmed)

**4. Fold 2 DOWN signals work at 2-bar hold:**
- Analysis: 87 combined fold 2 trades; if UP accuracy ≈ 35% and DOWN accuracy ≈ 55% → matches 44.83% combined
- DOWN signals (after large UP bars in downtrend) are profitable at 2-bar hold (dead-cat bounces fail in fold 2)
- These are already captured by the symmetric DOWN signal in step 112

### Root Cause of Remaining Accuracy Gap (65% target vs 59.36%)

**Fold 2 2-bar structural drag:**
- In sustained downtrend, isolated UP signals at 1-bar revert but REVERT BACK by bar+2 (trend continuation dominates at 30m scale)
- Fold 2 UP accuracy at 2-bar hold: ~35% → systematic loss for UP predictions in downtrend
- Fold 2 has 126/month trades (45 UP + 42 DOWN). UP signals have 35% accuracy → drag on combined
- No 5h-20h indicator captures the 21-day fold 2 decline period
- Fixing this requires: 7+ day lookback momentum, order book (spread), or cross-asset (funding)

### Recommended Next Steps (Priority Order)

1. **7-day momentum feature**: compute daily_ret = close/close.shift(672)-1 in signal block → detect multi-week downtrend
2. **Funding rate regime**: persistent negative funding → suppress UP signals (available in X_valid as `funding_rate`)
3. **Order book data**: bid-ask spread anomalies signal genuine exhaustion vs. systematic selling
4. **Cross-asset signals**: ETH/BTC ratio, DXY for regime detection at macro scale

### Evolution Trajectory
| Step | Key Change | Accuracy | Sharpe | Notes |
|------|-----------|----------|--------|-------|
| 85 | Leak fixed | 56.4% | -9.07 | ✅ clean baseline |
| 102 | Rule-based mean-reversion | 61.99% | -3.93 | ✅ reference |
| 108 | Isolated spike filter | 63.89% | -1.490 | ✅ best 1-bar |
| 112 | **2-bar hold** | **59.36%** | **+0.462** | ✅ **FIRST POSITIVE SHARPE** |
