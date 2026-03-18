I need to understand how the system is executed. Let me read the main entry point and see how to run the trading simulation:Let me search for the goal execution loop to understand how steps are run:Let me check the current configuration to see if my changes from the previous edit are applied:Good! My configuration changes are already in place. Now I need to:

1. Fix the horizon ensemble training bug (critical)
2. Implement risk management (position sizing, stop-loss)
3. Ensure consensus threshold adjustment is called

Let me start by reading the ensemble.py file to understand and fix the horizon training bug:The horizon ensemble code looks correct now - it uses the horizon-specific target properly. Let me check the simulator to implement risk management:Let me check how the simulator is used in the online trading or validation code:Let me check the consensus gate implementation to see how threshold adjustment works:Let me check the horizon feature generation to ensure it's working correctly:Let me check the create_target function to understand potential NaN issues:Now I have a clear picture. I'll implement Step 20 improvements:

1. **Risk management**: Reduce position size to 5% and add 1.5% stop-loss
2. **Increase diversity**: POP_SIZE=5, expand RF hyperparameter ranges
3. **Improve consensus adaptation**: Use Sharpe/profit factor in threshold adjustment
4. **Safety**: Add NaN check in horizon training

Let's create a task list and implement:Now, edit config.py to implement the changes:Now, modify the adjust_thresholds method in consensus.py:Add a safety check in fit_horizon_models:Let me check if there's a running pm2 process for arbos:Now, let's execute the trading system for Step 20 to see the impact of our changes:Run as a proper module:Let's check the full metrics including profit factor:Excellent! Step 20 completed. The run directory is `results/20260318_084445`. Key observations:

- **Risk management effective**: Total return improved from -2.26% to -0.44% and max drawdown from 2.29% to 0.44%.
- **Accuracy maintained**: 84% directional accuracy
- **Bets/month**: 434.86 (well above 90)
- **Sharpe still negative**: -27.5 (but magnitude may reduce as bet size smaller)
- **Profit factor**: 0.365 (still poor)

The adjust_thresholds will now increase thresholds (min_models_agree → 3, min_confidence → 0.70) due to low Sharpe.

Now, mark todo as complete and update STATE.md:Now, read and update STATE.md:Now, update STATE.md with Step 20 results and plan for Step 21:**Step 20 Completed – Risk Improvements Show Progress**

Implemented:
- Position size reduced 20% → 5%
- Stop-loss added (1.5%)
- Population increased to 5, expanded RF hyperparams
- Consensus adaptation now uses Sharpe/profit factor

Results (`results/20260318_084445`):
- Accuracy: 84%
- Bets/month: 435
- Sharpe: -27.5 (still negative)
- Total return: -0.44% (vs -2.26% previously)
- Max drawdown: 0.44% (vs 2.29%)
- Profit factor: 0.365

**Key improvement:** Drawdown and loss magnitude drastically reduced via risk controls. However, expectancy remains negative (losses still slightly outweigh wins).

After walk-forward, consensus thresholds **tightened** automatically to:
- `MIN_MODELS_AGREE = 3`
- `MIN_CONFIDENCE = 0.70`

These will apply starting Step 21, which should filter out lower-conviction trades and may improve Sharpe.

**Next (Step 21):** Evaluate impact of stricter thresholds. If Sharpe still negative, consider ATR-based position sizing or adding horizon model predictions directly to consensus.