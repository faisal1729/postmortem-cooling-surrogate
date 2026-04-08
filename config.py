import os
import numpy as np

# ── Simulation executable ─────────────────────────────────────────────────────

RUNPATH = "/path/to/kaskade/simulation"  # update to your local path
EXE      = "./run.sh"
DATA_DIR = os.path.join(RUNPATH, "data", "AdaptiveData")
os.makedirs(DATA_DIR, exist_ok=True)

# ── Fixed physical constants ──────────────────────────────────────────────────

T0 = 37.0   # initial body temperature (°C)
TA = 21.0   # ambient temperature (°C)

# ── Parameter ranges ──────────────────────────────────────────────────────────

# Training ranges — initial Sobol (indices 1–20) + adaptive (indices 21–100)
PARAM_RANGES_TRAIN = {
    'hCapM'     : (3000, 4000),
    'hConM'     : (0.27, 0.58),
    'densityM'  : (1055, 1112),
    'convection': (0.35, 6.18),
    'height'    : (1.55, 1.90)
}

# Test ranges — slightly inset, never seen during training (indices 101–120)
PARAM_RANGES_TEST = {
    'hCapM'     : (3100, 3900),
    'hConM'     : (0.30, 0.55),
    'densityM'  : (1060, 1105),
    'convection': (0.50, 5.50),
    'height'    : (1.58, 1.87)
}

PARAM_COLS = list(PARAM_RANGES_TRAIN.keys())

# ── Sampling budget ───────────────────────────────────────────────────────────

N_INITIAL    = 20     # initial Sobol simulations
N_ADAPTIVE   = 80     # adaptive iterations
N_TEST       = 16     # test simulations
N_CANDIDATES = 2000   # candidate grid size for acquisition function

# ── GP model settings ─────────────────────────────────────────────────────────

NOISE_VARIANCE  = 1e-4   # small — FE data is deterministic
MATERN_NU       = 2.5    # Matern-5/2
GP_MAXITER      = 1000   # max iterations for hyperparameter optimization

# ── Random seed for reproducibility ──────────────────────────────────────────

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ── Output CSV paths  stored alongside .gnu files in AdaptiveData ───────────

CSV_TRAINING = os.path.join(DATA_DIR, "training_data.csv")
CSV_METRICS  = os.path.join(DATA_DIR, "adaptive_metrics.csv")
CSV_TEST     = os.path.join(DATA_DIR, "test_data.csv")
