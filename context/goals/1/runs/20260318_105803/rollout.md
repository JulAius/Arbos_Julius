**Step 38 & 39 complete.** Here's the summary:

---

**Step 38 (MIN_CONFIDENCE 0.73→0.75):**
- 81.98% accuracy, 464 bets/month, Sharpe=32.197
- Sharpe essentially flat vs Step 37 (32.197 vs 32.23) → **threshold plateau confirmed**

**Step 39 (Pivot: MAX_TRAIN 4k→6k + wider hyperparams):**
- **82.35%** accuracy | **514.6** bets/month | Sharpe=**32.99**
- All metrics improved simultaneously: +0.37% acc, +10.9% bets, +0.79 Sharpe

| Metric | Step 38 | Step 39 | Target |
|--------|---------|---------|--------|
| Accuracy | 81.98% | **82.35%** | ≥65% ✅ |
| Bets/month | 463.9 | **514.6** | >90 ✅ |
| Sharpe | 32.197 | **32.987** | >1.0 ✅ |
| Total return | +7.40% | **+7.46%** | positive ✅ |
| Max drawdown | 0.059% | **0.076%** | controlled ✅ |

Top features are momentum-based: `mom_1_1h` and `close_open_ratio_1h` dominate (1h context is key).

**Step 40 plan:** Expand `ONLINE_DATA_DAYS` 120→180 for better regime coverage and stronger WFV folds.