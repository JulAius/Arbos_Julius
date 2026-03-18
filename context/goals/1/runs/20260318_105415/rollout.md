**Step 36 Complete — MIN_CONFIDENCE 0.69 → 0.71**

| Metric | Step 35 | Step 36 | Change | Target |
|--------|---------|---------|--------|--------|
| Accuracy | 78.82% | **79.93%** | +1.11% | ≥65% ✅ |
| Bets/month | 885.2 | **717.5** | -19.0% | >90 ✅ |
| Sharpe | 31.87 | **31.76** | -0.11 (plateau) | >1.0 ✅ |
| Total return | +9.32% | **+8.37%** | -0.95% | positive ✅ |
| Max drawdown | 0.13% | **0.20%** | slight rise | controlled ✅ |

All targets still comfortably met. Key observation: Sharpe plateaued for the first time (31.76 vs 31.87) — accuracy improved +1.11% but Sharpe/return stayed flat, suggesting the threshold tuning curve is flattening around 0.71.

**Step 37 plan:** Try MIN_CONFIDENCE 0.73 to confirm — if Sharpe resumes improvement, continue; if it stays flat or drops, switch to a different improvement axis (feature engineering, model diversity, or ensemble strategy).