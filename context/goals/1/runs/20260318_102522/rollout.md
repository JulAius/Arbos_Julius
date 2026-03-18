Step 27 is complete. Here's the summary:

## Step 27 Results

**Changes implemented:**
- Added **8h horizon** (`"8h": 32`) for longer-term trend context features
- Added **logistic regression** alongside Random Forest in the population (`MODEL_TYPES = ["random_forest", "logistic"]`)
- Reduced logistic `MAX_ITER` from 5000→1000 (sufficient with standard scaling)
- Added **3-fold WFV support** in code (not triggered at 60-day window, will need 90-day expansion)

**Multi-fold WFV (2 × 21-day folds):**

| Metric | Fold 1 | Fold 2 | Combined | Target |
|--------|--------|--------|---------|--------|
| Accuracy | 78.0% | 81.7% | **79.8%** ✅ | ≥65% |
| Bets/month | 1115 | 1045 | **1080** ✅ | >90 |
| Sharpe | 38.2 | 45.9 | **42.1** ✅ | >1.0 |
| Total return | — | — | **+10.4%** ✅ | positive |
| Max drawdown | — | — | **0.10%** ✅ | controlled |

**Key findings:**
- Logistic models added 3.2× more trading frequency (334→1080 bets/month) by producing more decisive confidence scores
- Accuracy dropped 4.1% (83.9%→79.8%) as logistic models dilute RF quality — but the extra profitable trades more than compensate, doubling total return (+4.2%→+10.4%)
- **Next step (28):** Tighten `MIN_CONFIDENCE` 0.68→0.70 to recover 1-2% accuracy while maintaining 500+ bets/month, and expand WFV to 3 folds (63 days OOS)