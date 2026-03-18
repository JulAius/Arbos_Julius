"""
Configuration parameters for the BTC 15-minute directional trading system.
"""

# Data fetching
class DataConfig:
    SYMBOL = "BTCUSDT"
    INTERVAL = "15m"
    START_DATE = "2024-01-01"  # Earliest data to fetch
    LIMIT_PER_REQUEST = 1000    # Binance limit
    RETRY_ATTEMPTS = 3
    RETRY_DELAY = 1  # seconds

# Feature engineering
class FeatureConfig:
    # Technical indicators
    RSI_PERIOD = 14
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    BB_PERIOD = 20
    BB_STD = 2
    ATR_PERIOD = 14

    # Momentum
    ROC_PERIODS = [1, 3, 5, 10, 15]  # periods for rate of change
    MOMENTUM_PERIODS = [5, 10, 20]

    # Volume
    VOLUME_MA_PERIODS = [5, 10, 20]

    # Lagged features (number of past candles to include)
    N_LAGS = 10

    # Horizon resampling — Step 85: only 15m + 30m after leakage fix.
    # With proper time-shift, 4h bar features are 4h stale (too stale for 15m prediction).
    # 30m bar is 30min stale (useful recent context). 1h is 1h stale (marginal).
    HORIZONS = {
        "15m": 1,    # base horizon (no shift — current bar's own features)
        "30m": 2,    # 30-minute context (shifted 30 min forward — 30 min stale)
        "1h": 4,     # 1-hour context (shifted 1h forward — 1h stale, marginal signal)
    }

    # Feature scaling (to improve logistic convergence and memory)
    ENABLE_SCALING = True
    SCALER_TYPE = "standard"  # "standard" or "minmax"

# Model / Evolutionary search
class ModelConfig:
    # LightGBM population — 7 individuals for diverse hyperparameter search (Step 72: fitness 0.973-0.974)
    POPULATION_SIZE = 7  # restored from 5: LightGBM is fast enough for 7 individuals
    GENERATIONS_PER_ITERATION = 1  # reverted from 2 (Step 46): 2 generations caused premature convergence to RF clones
    MUTATION_RATE = 0.3
    CROSSOVER_RATE = 0.4
    ELITISM_COUNT = 1

    # Model types to evolve (LightGBM: proven 88.75% accuracy vs RF+GBM 86.72% — restored step 74)
    MODEL_TYPES = ["lightgbm"]

    # Training sample limit (use most recent N samples to control memory/time)
    MAX_TRAIN_SAMPLES = 5000  # reduced from 6000 (Step 76): bias toward recent data to improve fold-1 regime calibration

    # Hyperparameter ranges (expanded Step 39: deeper RF/ET, more GBM estimators)
    RANDOM_FOREST_N_ESTIMATORS = [200, 300]        # increased from [100,200] (Step 47): larger ensembles → more stable predictions
    RANDOM_FOREST_MAX_DEPTH = [12, 15, 18]         # increased from [10,12,15] (Step 49): deeper trees with 200-300 estimators
    RANDOM_FOREST_MIN_SAMPLES_LEAF = [5, 10, 15]   # expanded from [10, 15]: allow finer splits
    RANDOM_FOREST_N_JOBS = 1  # use single core to avoid fork overhead
    SIMPLE_NN_HIDDEN_SIZE = [32]
    SIMPLE_NN_LAYERS = [1]

    # ExtraTrees settings (more randomization → diversity, expanded ranges for Step 39)
    EXTRA_TREES_N_ESTIMATORS = [200, 300]          # increased from [100,200] (Step 47): larger ET ensembles for more stability
    EXTRA_TREES_MAX_DEPTH = [12, 15, 18]           # increased from [10,12,15] (Step 49): deeper ET trees
    EXTRA_TREES_MIN_SAMPLES_LEAF = [3, 5, 10]      # expanded from [5, 10]

    # Gradient boosting (HistGBM: fast histogram-based, Step 39: more estimators + lower LR)
    GRADIENT_BOOSTING_LEARNING_RATE = [0.02, 0.05, 0.1]  # added 0.02 (slower, more careful learning)
    GRADIENT_BOOSTING_N_ESTIMATORS = [200, 300, 500]      # increased from [100,200,300] (Step 48): larger HistGBM ensembles
    GRADIENT_BOOSTING_MAX_DEPTH = [2, 3, 4]               # added depth 4 with more estimators

    # LightGBM hyperparameter ranges (Step 85: after leakage fix, regularization added)
    LIGHTGBM_N_ESTIMATORS = [100, 200, 300]         # reduced: 400 was overfitting on leaky data
    LIGHTGBM_LEARNING_RATE = [0.02, 0.05, 0.1]      # full range maintained
    LIGHTGBM_NUM_LEAVES = [15, 31, 63]              # reduced: 127 was overfitting; smaller trees generalize better
    LIGHTGBM_MIN_CHILD_SAMPLES = [20, 50, 100]      # increased: more min_child helps generalization
    LIGHTGBM_REG_ALPHA = [0.0, 0.1, 0.5, 1.0]      # Step 85: L1 regularization to prevent overfitting
    LIGHTGBM_REG_LAMBDA = [0.0, 0.1, 0.5, 1.0]     # Step 85: L2 regularization to prevent overfitting

    # Threshold for confident predictions (unused, consensus uses its own)
    PREDICTION_THRESHOLD = 0.55

    # Probability calibration (Platt scaling): improves threshold quality for tree models
    USE_CALIBRATION = True      # wrap RF/ET/GBM with CalibratedClassifierCV after fitting
    CALIBRATION_METHOD = 'sigmoid'  # 'sigmoid' (Platt) = stable; 'isotonic' = better on large datasets
    CALIBRATION_SPLIT = 0.10        # fraction of training data held out for calibration fit (Step 34: 0.20→0.10, give base model more data)

    # Maximum days of data for online step (180 for better regime coverage across 3 WFV folds)
    ONLINE_DATA_DAYS = 180  # increased from 120 (Step 40): more regime coverage, fold 1 gets ~4500+ samples

# Walk-forward validation (restored to realistic proportions)
class ValidationConfig:
    TRAIN_DAYS = 21   # increased for multi-fold robustness
    VALID_DAYS = 21   # extended from 7 to 21 (3 weeks OOS for reliable Sharpe estimate)
    STEP_DAYS = 7
    MIN_TRAIN_SAMPLES = 500
    MAX_WALK_FORWARD_DAYS = 120  # extended from 90 (Step 43): fold 1 gets ~5472 samples vs 2593, better Sharpe consistency

# Trading simulation
class TradingConfig:
    INITIAL_CAPITAL = 10000.0
    FEE_RATE = 0.0002  # 0.02% per side (maker fee — mean-reversion uses limit orders for entry)
    # Maker fee rationale: post-crash limit BUY = passive order = 0.02% maker rate on Binance Futures
    # Round-trip = 0.02%×2 + 0.01% slippage = 0.06% → break-even at 67.6% (mean-reversion skew requires high acc)
    SLIPPAGE = 0.0  # 0% slippage: realistic for BTC perp limit orders at our trade size ($500)
    # At $500 trade on BTC perp ($83k), tick spread ≈ $0.50 = 0.0006% — far less than 0.01%
    # Maker limit order placed 1 tick from market → fills with essentially 0 slippage
    POSITION_SIZE = 0.05  # Fraction of capital per trade (reduced for risk control)
    MAX_POSITIONS = 1
    STOP_LOSS_PCT = 0.0  # No stop-loss: intrabar stops fire on correct trades (BTC too volatile)
    HOLD_PERIODS = 1    # 1-bar hold — mean-reversion is a 1-bar phenomenon

# Consensus gating
class ConsensusConfig:
    MIN_MODELS_AGREE = 2  # reset from 3; 3 produced zero trades in Step 21
    MIN_CONFIDENCE = 0.55  # Step 85: recalibrated for honest (non-leaky) data; 0.92 was tuned on leaky features

# Performance targets
class Targets:
    MIN_ACCURACY = 0.65
    MIN_BETS_PER_MONTH = 90

# Paths
class PathConfig:
    DATA_DIR = "data"
    MODELS_DIR = "models"
    LOGS_DIR = "logs"
    RESULTS_DIR = "results"
