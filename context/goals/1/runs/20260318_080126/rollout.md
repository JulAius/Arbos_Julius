# Step 14 Rollout: Relax Confidence to 0.65

## 1. Configuration at Start of Step 14

- **Action from Step 13:** Lower MIN_CONFIDENCE from 0.75 → 0.65 (already applied in config file during Step 13 reflection)
- Config values:
  - `ConsensusConfig.MIN_CONFIDENCE = 0.65`
  - `ConsensusConfig.MIN_MODELS_AGREE = 2`
  - `ModelConfig.MODEL_TYPES = ["random_forest"]`
  - `FeatureConfig.HORIZONS = {"15m": 1}`
  - RF hyperparams: `max_depth=8`, `min_samples_leaf=10`, `n_estimators=50`

## 2. Execution

- Command: `python -m arbos_trading.main`
- Data: BTCUSDT 15m, 90 days (8641 samples), features: 31
- Training limit: 4000 most recent samples
- Population: 5 random forest models (identical hyperparameters)
- Walk-forward validation: single split (Train 2209, Valid 672)

## 3. Results

```
Accuracy:           0.5517
Bets/month:         126.11
Sharpe:            -27.37
Max drawdown:       0.0199
Total return:      -0.0199
Latest signal:      -1 (NO_TRADE)
Signals generated: 2880 → ~126 bets/month after consensus
```

- Bet frequency target **ACHIEVED** (>90/month)
- Accuracy far below 65% (55.2%)
- Sharpe deeply negative → unprofitable strategy

## 4. Analysis & Reflection

**Effect of confidence relaxation:** Lowering MIN_CONFIDENCE from 0.75 to 0.65 increased bet frequency from 8.7 to 126 per month (14.5x). However, accuracy collapsed from ~100% (tiny sample) to 55% on a meaningful validation set, and Sharpe became hugely negative.

**Why accuracy dropped:** The model's predictive power is weak. Walk-forward validation shows only marginal accuracy (~random). The confidence threshold is not a reliable indicator of correctness; lower-confidence predictions are largely noise/incorrect.

**Why Sharpe negative:** The strategy is losing money on average. The consensus gate with low confidence admits many losing trades.

**Consensus gate effectiveness:** With all 5 models being identical RFs, `MIN_MODELS_AGREE=2` is automatically satisfied whenever any model produces a prediction. The consensus gate therefore offers no additional filtering beyond the confidence threshold. The ensemble is not diverse enough to provide true consensus.

**Population diversity:** All 5 models have identical architecture and hyperparameters. Evolutionary search explores only one point in the hyperparameter space due to restricted ranges. This limits the potential for ensemble diversity to improve signal quality.

**Fitness vs. OOS performance:** Evolution produced models with fitness ~0.77, but out-of-sample accuracy is only 55%. This indicates overfitting or that the fitness metric (likely training accuracy) does not correlate with walk-forward performance.

## 5. Path Forward (Proposal for Step 15)

We must improve predictive accuracy while maintaining sufficient bet frequency. Given we currently have 126 bets/month, we can afford to tighten selection to improve quality.

**Proposed changes:**

1. **Raise MIN_CONFIDENCE** to 0.70 (or 0.72) to be more selective. Expected: bets/month may drop to ~90-100 range, but accuracy and Sharpe should improve if lower-confidence bets are indeed lower-quality.
2. **Add model diversity:** Change `MODEL_TYPES` to include both `random_forest` and `simple_nn`. This will create heterogeneous predictions; consensus across different model families should be a stronger signal. With population size 5, we can have e.g., 3 RFs + 2 NNs, or 2 each plus 1 extra. The exact composition will be determined by evolutionary selection; but having both types available will increase diversity.
3. Keep `MIN_MODELS_AGREE=2` (still requires 2/5 agreement, now across mixed types).
4. Keep horizons limited to 15m for now to maintain speed; we will expand later if needed.
5. Consider later: incorporate longer horizons (1h, 4h) as contextual features, as originally envisioned.
6. Consider later: fitness based on validation Sharpe rather than training accuracy.

**Expected outcome:** Stricter confidence and mixed-model consensus should yield fewer but higher-quality bets. We aim to bring accuracy back above 65% and Sharpe positive, while keeping bets/month above 90.

## 6. Metrics Summary

| Metric | Target | Step 13 | Step 14 |
|--------|--------|---------|---------|
| Accuracy | >=0.65 | 1.000 (8 trades) | 0.552 (672 valid) |
| Bets/month | >90 | 8.7 | 126.1 |
| Sharpe | >0 | 3.536 | -27.37 |
| Max DD | controlled | ~0 | 1.99% |
| Net return | positive | tiny | -1.99% |

**Status:** Step 14 achieved bet frequency but failed accuracy and Sharpe. Need corrective action in Step 15.
