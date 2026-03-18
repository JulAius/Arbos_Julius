All three model classes updated. Here's the Step 33 summary:

---

**Step 33 Complete — Probability Calibration (Platt Sigmoid)**

**Results (3-fold WFV, 63 days OOS):**

| Metric | Step 32 | Step 33 | Change | Target |
|--------|---------|---------|--------|--------|
| Accuracy | 81.25% | **76.82%** | -4.43% | ≥65% ✅ |
| Bets/month | 556.6 | **1137.9** | +104% | >90 ✅ |
| Sharpe | 32.30 | **26.94** | -5.4 | >1.0 ✅ |
| Total return | +7.77% | **+9.07%** | +1.3% | positive ✅ |
| Max drawdown | 0.081% | **0.37%** | 4.6× | <1% ✅ |

**What happened:** Platt sigmoid calibration compresses tree model probabilities toward 0.5 (away from their natural bimodal distribution). This means more samples cross the 0.65 threshold → 2× more trades, with a modest accuracy cost (-4.43%). All objectives remain comfortably met.

**Also fixed:** sklearn 1.7.2 deprecated `cv='prefit'` → migrated to `FrozenEstimator` pattern.

**Step 34 plan:** Raise `MIN_CONFIDENCE` 0.65→0.67 and shrink `CALIBRATION_SPLIT` 0.20→0.10 to recover ~1-2% accuracy while keeping bets/month above 900.