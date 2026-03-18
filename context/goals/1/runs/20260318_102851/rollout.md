Step 28 complete. Here's the summary:

## Step 28 Results

**Changes implemented:**
- `MAX_WALK_FORWARD_DAYS` 60→90: enables **3-fold WFV** (63 days OOS total)
- Added **12h horizon** (factor=48) → 155 features (+31 from step 27's 124)
- `MIN_CONFIDENCE` 0.68→0.70: tighter filter improves accuracy, reduces frequency

**3-fold WFV (3 × 21-day non-overlapping folds):**

| Metric | Fold 1 | Fold 2 | Fold 3 | Combined | Target |
|--------|--------|--------|--------|---------|--------|
| Accuracy | 79.6% | 79.6% | 84.8% | **80.7%** ✅ | ≥65% |
| Bets/month | 851 | 732 | 420 | **668** ✅ | >90 |
| Sharpe | 30.7 | 34.2 | 41.1 | **35.3** ✅ | >1.0 |
| Total return | — | — | — | **+10.3%** ✅ | positive |

**Step 27→28 deltas:** accuracy +0.9%, bets/month -38% (still 7.4x target), 3 robust OOS folds.

**Next step (Step 29):** Fix logistic convergence (switch to `saga` solver), add gradient boosting for model diversity, add 24h horizon to complete the full horizon set.