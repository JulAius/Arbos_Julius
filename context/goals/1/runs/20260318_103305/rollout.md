Step 29 is complete. Here's the summary:

## Step 29 Results

**Changes implemented:**
- Added `gradient_boosting` (HistGradientBoostingClassifier — fast, no external deps) to MODEL_TYPES
- Added `24h` horizon (factor=96) → 186 features (+31 from step 28's 155)
- Fixed logistic solver: `lbfgs` → `saga` (better for large feature sets)

**3-fold WFV (3 × 21-day non-overlapping folds):**

| Metric | Fold 1 | Fold 2 | Fold 3 | Combined | Target |
|--------|--------|--------|--------|---------|--------|
| Accuracy | 70.4% | 84.0% | 74.4% | **75.4%** ✅ | ≥65% |
| Bets/month | 1032 | 726 | 1164 | **974** ✅ | >90 |
| Sharpe | 1.5 | 44.7 | 23.3 | **23.2** ✅ | >1.0 |
| Total return | — | — | — | **+7.4%** ✅ | positive |

**Root cause of Fold 1 weakness:** With 90-day window and 3 folds × 21-day validation = 63 days OOS, Fold 1 only has 27 days (~746 samples) of training data for 186 features. HistGBM overfits on small datasets.

**Step 30 plan:** Increase `ONLINE_DATA_DAYS` from 90 → 120 to give Fold 1 ~57 days (~3000 samples) of training data, increase logistic `max_iter` to 2000, tighten GBM depth to [2,3].