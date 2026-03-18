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

    # Horizon resampling (in minutes) - multi-horizon context for 15m target
    HORIZONS = {
        "15m": 1,    # base horizon
        "30m": 2,    # 30-minute context (2 x 15m) - sub-hourly signal (Step 49)
        "1h": 4,     # 1-hour context (4 x 15m)
        "2h": 8,     # 2-hour context (8 x 15m) - mid-term signal between 1h and 4h (Step 46)
        "4h": 16,    # 4-hour context (16 x 15m)
        "8h": 32,    # 8-hour context (32 x 15m) - longer-term trend signal
        "12h": 48,   # 12-hour context (48 x 15m) - daily cycle signal
        "24h": 96,   # 24-hour context (96 x 15m) - full daily cycle
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

    # LightGBM hyperparameter ranges (Step 72 proven config: 88.75% accuracy, 1692 bets/month, Sharpe 50.41)
    LIGHTGBM_N_ESTIMATORS = [200, 300, 400]         # larger = more stable, slower
    LIGHTGBM_LEARNING_RATE = [0.02, 0.05, 0.1]      # 0.05 dominant in best models
    LIGHTGBM_NUM_LEAVES = [31, 63, 127]             # 127 dominant (deeper leaf-wise)
    LIGHTGBM_MIN_CHILD_SAMPLES = [10, 20, 50]       # regularization via min samples

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
    FEE_RATE = 0.0004  # 0.04% per side (Binance Futures taker fee - realistic)
    SLIPPAGE = 0.0001  # 0.01% per side (BTC perp is very liquid)
    POSITION_SIZE = 0.05  # Fraction of capital per trade (reduced for risk control)
    MAX_POSITIONS = 1
    STOP_LOSS_PCT = 0.0  # No stop-loss: fixed 1-candle hold exits at next close

# Consensus gating
class ConsensusConfig:
    MIN_MODELS_AGREE = 2  # reset from 3; 3 produced zero trades in Step 21
    MIN_CONFIDENCE = 0.90  # Step 78: grid-search optimum — 93.71% accuracy, 1137 bets/month, Sharpe 57.20 (deterministic, seed=42)

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
