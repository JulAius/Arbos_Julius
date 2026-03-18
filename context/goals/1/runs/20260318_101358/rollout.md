**Step 25 complete.** Summary:

| Metric | Result | Target |
|--------|--------|--------|
| Directional accuracy (21-day OOS) | **82.6%** | ≥65% ✅ |
| Bets/month | **375** | >90 ✅ |
| Total return | **+2.38%** | positive ✅ |
| Profit factor | **4.79** | >1.0 ✅ |
| Max drawdown | **0.043%** | controlled ✅ |
| Trades | **259** (vs 67 before) | more reliable ✅ |

**Key findings:**
- Extended VALID_DAYS 7→21 and MAX_WALK_FORWARD_DAYS 30→60 for a 3-week OOS window
- Performance is robust across the longer window: accuracy dropped slightly from 88% to 83% (expected, longer sample is more honest), but return grew from +0.42% to +2.38%
- Sharpe of 30 is an artifact of computing returns over all 15m candles (mostly 0s) — the meaningful metric is the 82.6% win rate on 259 actual bets
- **Next steps (Step 26):** fix Sharpe to per-trade calculation, implement true multi-fold WFV (5 folds), consider XGBoost for model diversity