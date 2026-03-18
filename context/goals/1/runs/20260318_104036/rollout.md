Step 31 is complete. Here's the summary:

## Step 31 Results

**Changes implemented:**
- Replaced `logistic` with `ExtraTreesClassifier` in `MODEL_TYPES` (RF + ExtraTrees + GBM)
- Added `ExtraTreesModel` class to `base.py` and `MODEL_REGISTRY`
- Added ExtraTrees hyperparams to `config.py` and mutation/selection logic in `population.py`

**3-fold WFV (combined, 63 days OOS):**

| Metric | Step 30 | Step 31 | Target |
|--------|---------|---------|--------|
| Accuracy | 81.1% | **84.2%** (+3.1%) | ≥65% ✅ |
| Bets/month | 612 | **250.8** | >90 ✅ |
| Sharpe | 35.7 | 28.2 | >1.0 ✅ |
| Total return | +9.97% | +4.81% | positive ✅ |
| Max drawdown | 0.093% | **0.059%** | controlled ✅ |

**Key finding:** Removing logistic improved accuracy by +3.1% (best accuracy yet: 84.2%), but bets/month dropped from 612 → 250.8. Logistic's bimodal probability distribution was responsible for ~60% of trades. Tree models produce smoother probabilities that cross the 0.70 confidence gate less often.

**Population at end of step 31:** 3× RF (best fitness 0.810), 1× ExtraTrees (0.735), 1× GBM (0.685). ExtraTrees is established in the population but competing below RF.

**Step 32 plan:** Lower `MIN_CONFIDENCE` from 0.70 → 0.65 to recover bet frequency to ~400/month. With 84.2% underlying accuracy, there's ample headroom above the 65% target.