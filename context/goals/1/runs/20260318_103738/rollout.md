Step 30 complete. Results summary:

| Metric | Step 29 | Step 30 | Target |
|--------|---------|---------|--------|
| Accuracy | 75.4% | **81.1%** | ≥65% ✅ |
| Bets/month | 974 | **612** | >90 ✅ |
| Sharpe | 23.2 | **35.7** | >1.0 ✅ |
| Total return | +7.4% | **+9.97%** | positive ✅ |
| Max drawdown | 0.60% | **0.093%** | controlled ✅ |
| Fold 1 accuracy | 70.4% | **82.1%** | fixed ✅ |

**Key win:** The main fix worked — increasing `ONLINE_DATA_DAYS` 90→120 gave Fold 1 2593 training samples (vs 746 before), boosting its accuracy from 70.4% → 82.1%. Max drawdown dropped 6.5× to just 0.093%.

**Step 31 plan:** Replace logistic regression with `ExtraTreesClassifier` — logistic never wins in population evolution, still causes saga convergence warnings at iter=2000, and adds training overhead. ExtraTrees gives better ensemble diversity (RF + ExtraTrees + GBM) without any convergence issues.