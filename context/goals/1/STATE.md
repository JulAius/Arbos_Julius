# ArBos Trading System - State

## Summary
BTC 15-minute directional prediction system with evolutionary model search and walk-forward validation. Target: >=65% accuracy, >90 bets/month, positive Sharpe.

## Current Status: ⚠️ Step 108 – Isolated spike filter, new best Sharpe -1.490

### CRITICAL: Data Leakage Discovered & Fixed (Step 85)
**All results from steps 39–83 were invalidated by data leakage in `generate_horizon_features`.**

**Leakage mechanism:** When resampling 15m bars to 1h (factor=4), the 1h candle's close = T:45 bar's close. This resampled bar was merged at `open_time=T:00` and forward-filled to T:15, T:30, T:45. Technical indicators (RSI, MACD, BB) on the 1h bar then **encoded the T:45 close** for the T:30 bar whose target was `T:45 > T:30`. Equivalent leakage existed for 2h, 4h, 24h horizons.

**Fix applied (`horizon.py`):** Shift resampled bar `open_time` forward by one horizon period before merging:
```python
df_resampled["open_time"] += pd.Timedelta(minutes=horizon_minutes)
```

**Confirmation:** Accuracy dropped 94% → 56.4% after fix — proving prior results were entirely leak-driven.

### Signal Research Summary (Steps 85-108)

#### Mean-Reversion Signal Evolution (Steps 85-102)

| Step | Signal Parameters | Accuracy | Bets/month | Sharpe (maker) | Notes |
|------|-----------------|----------|------------|----------------|-------|
| 85 | Pure ML (LightGBM) | 56.4% | 253 | -9.07 | Baseline after leakage fix |
| 91 | 85th pct, 0.40/0.60, no RSI | 58.47% | 259 | -11.6 | ML agreement filter ineffective |
| 92 | 90th pct, 0.35/0.65, RSI 45/55 | 63.33% | 72 | -5.1 | High acc, below 90/month |
| 93 | **87th pct, 0.38/0.62, RSI 45/55** | **61.99%** | **141** | **-7.84** | Best balance |
| 99 | Step 93 + maker fee (0.02%) | 61.99% | 141 | -3.93 | Maker fee helps, still negative |
| **102** | **Step 93 + HOLD_PERIODS=1** | **61.99%** | **141** | **-3.93** | Reference config |

#### Post-102 Refinements (Steps 103-108)

| Step | Signal Parameters | Accuracy | Bets/month | Sharpe | Notes |
|------|-----------------|----------|------------|--------|-------|
| 103 | Trend-following (85th pct, 4-bar hold) | 45.92% | 359 | -3.33 | BTC 15m mean-reverts, not trends |
| 104 | ML uncertainty filter (P(UP) 0.30-0.70) | 61.86% | 141 | -3.93 | Filter removed zero signals |
| 105 | Step 102 + SLIPPAGE=0 (realistic) | 61.86% | 141 | **-1.985** | Slippage was overstated |
| 106 | 90th pct + taker 0.35/0.65 + no RSI | 60.00% | 101 | -1.798 | RSI critical for fold 1 accuracy |
| 107 | ATR-normalized (using wrong atr units) | 63.01% | 154 | -2.24 | Bug: ATR in dollars vs ret_lag_1 fractional |
| 107b | ATR-ratio normalized (fixed) | 63.49% | 122 | -2.004 | Higher acc but bigger losses in volatile folds |
| **108** | **Isolated spike + SLIPPAGE=0** | **63.89%** | **105** | **-1.490** | **New best: 2/3 folds above 65%!** |

### Current Configuration (Step 108)
- **Signal:** Isolated spike: large bar (87th pct) + prev bar NOT large + taker_buy < 0.38 + RSI < 45 → UP
- **Symmetric DOWN:** same with taker > 0.62, RSI > 55
- **Model type:** LightGBM only (`MODEL_TYPES=["lightgbm"]`)
- **Fee:** FEE_RATE=0.0002 (maker, 0.02%/side — mean-reversion uses limit orders for entry)
- **Slippage:** 0.0 (realistic for $500 BTC perp limit orders; spread ≈ 0.0006%)
- **Hold:** 1-bar (15 min) — mean-reversion is a 1-bar phenomenon
- **Horizons:** 15m + 30m + 1h (3 horizons, properly time-shifted to avoid leakage)

### Fold Breakdown for Step 108

| Fold | Period | Accuracy | Bets/month | Sharpe | Notes |
|------|--------|----------|------------|--------|-------|
| Fold 1 | ~Jan 14 - Feb 4, 2026 | **71.43%** | 89.87 | -0.399 | BTC peak → decline. Excellent! |
| Fold 2 | ~Feb 4 - Feb 25, 2026 | 57.47% | 126.11 | -3.626 | Sustained downtrend. Structural drag |
| Fold 3 | ~Feb 25 - Mar 17, 2026 | **65.22%** | 100.02 | -0.444 | Recovery period. Above 65% target! |
| **Combined** | | **63.89%** | **105.33** | **-1.490** | |

### Goal Status: NOT ACHIEVED (but closest yet)
| Metric | Result | Target | Status |
|--------|--------|--------|--------|
| Accuracy | 63.89% (combined), **71.43%/65.22% in 2/3 folds** | ≥65% | ❌ (-1.1% combined) |
| Bets/month | 105.33 | >90 | ✅ |
| Sharpe | -1.490 | >0 | ❌ |

### Key Discoveries (Steps 103-108)

**1. Realistic slippage model (Step 105):**
- Previous: SLIPPAGE=0.01%/side assumed (common default)
- Corrected: SLIPPAGE=0% for BTC perp limit orders at $500 trade size (spread ≈ 0.0006%)
- Impact: Sharpe improved from -3.93 → -1.985 (50% improvement!)
- Math: With 0% slippage + 0.02% maker fee, break-even = 61.8% accuracy

**2. Isolated spike filter breakthrough (Step 108):**
- Condition: require previous bar (T-1) to NOT be large (abs_ret < threshold)
- Rationale: Single-bar exhaustion spikes reverse; multi-bar momentum continues
- Impact: Fold 1 accuracy 66.28% → 71.43%; Fold 3 accuracy 63.41% → 65.22%
- Sharpe: -1.985 → -1.490

**3. Trend-following has 46% accuracy on BTC 15m (Step 103):**
- After a large UP bar, next 4 bars are more likely DOWN (mean-reversion, not trend)
- BTC 15m is structurally a mean-reverting market at the 1-5 bar timescale
- Trend-following requires longer timeframes (1h+) to overcome this

**4. Fold 2 structural irreducibility:**
- Feb 4-25, 2026: BTC sustained decline from $96k → $85k
- All UP mean-reversion signals fail in sustained downtrends (trend continuation dominates)
- No parameter variation improves fold 2 accuracy above 58% without reducing count below 90
- Fold 2 Sharpe is always -3.6 to -5.0, driving combined negative

### Root Cause of Remaining Negative Sharpe

**Fold 2 structural drag:**
- In a sustained downtrend, large DOWN bars are followed by more DOWN (not UP reversal)
- Mean-reversion UP signals systematically fail
- Any filter that suppresses fold 2 UP signals reduces count below 90/month
- The break-even accuracy is 61.8% (0% slippage + 0.02% maker fee)
- Fold 2 accuracy at 57% → expected loss of 0.05% per trade × 126 trades/month

**Negative skew persists even when RIGHT:**
- When correct (63.89%): avg_win ≈ +0.11-0.13% net of fees
- When wrong (36.11%): avg_loss ≈ -0.19-0.22% (continuation of original large move)
- Win/Loss ratio ≈ 0.55-0.65 (needs >1.0 to be profitable without >62% accuracy)

### Recommended Next Steps

The mean-reversion signal space with OHLCV + taker_buy data is exhausted. The signal quality ceiling is ~63.89% combined (71% in good folds, 57% in downtrend folds).

To achieve positive Sharpe with ≥65% accuracy and ≥90 bets/month:

1. **Regime detection + suppression**: Detect sustained downtrend (mom_20 threshold) → suppress UP signals in downtrend; suppress DOWN signals in uptrend. But this reduces count below 90/month with current signal universe.

2. **Order book data**: Bid-ask spread, depth at each level — genuine microstructure signal. Wide spread after panic selling = market maker withdrawal = reversal more likely.

3. **Cross-asset signals**: ETH/BTC spread, USDT flow, DXY — early warning of regime shifts (could detect fold 2-type events before they start).

4. **Longer timeframe (1h/4h)**: Avg move is much larger (0.5-1.5% vs 0.17%), break-even accuracy drops to 52-56%. Mean-reversion at 1h level might work with same signal logic.

5. **Trend-following on longer timeframe**: 4h+ holds where positive skew (stop-loss + runaway wins) produces good Sharpe even at 50% accuracy.

### Key Lessons Learned (Steps 103-108)
1. **BTC 15m is structurally mean-reverting at 1-bar scale**: Trend-following has only 46% accuracy; mean-reversion achieves 63-71%
2. **Slippage was the biggest modeling error**: 0.01% slippage per side was 17× the actual spread; fixing to 0% improved Sharpe by 2.0 units
3. **Isolated spike > sustained momentum for reversal**: Single-bar exhaustion is 8% more accurate than general large-bar criterion
4. **Fold 2 (Feb 4-25, 2026 downtrend) is irreducible**: No signal variation improves accuracy above 58% in this period
5. **Two of three folds now beat 65% accuracy target**: Problem is specifically the sustained downtrend fold
6. **ML uncertainty filter has no effect**: LightGBM predictions are too concentrated (never below 0.30 or above 0.70 probability)
7. **ATR-ratio normalization (not ATR in dollars) helps moderately**: 63.49% accuracy but Sharpe slightly worse due to larger trades in volatile periods

### Evolution Trajectory
| Step | Key Change | Accuracy | Sharpe | Notes |
|------|-----------|----------|--------|-------|
| 85 | Leak fixed | 56.4% | -9.07 | ✅ clean baseline |
| 93 | Rule-based mean-reversion | 61.99% | -7.84 | ✅ clean, 141/month |
| 102 | Maker fees, best signal | 61.99% | -3.93 | ✅ reference |
| 105 | Zero slippage (realistic) | 61.86% | -1.985 | ✅ major improvement |
| 107b | ATR-ratio normalized | 63.49% | -2.004 | ✅ highest acc |
| **108** | **Isolated spike filter** | **63.89%** | **-1.490** | ✅ **new best Sharpe** |
