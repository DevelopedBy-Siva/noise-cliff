from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "raw"
RESULTS_DIR = BASE_DIR / "results"

DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 43, 44]

NOISE_LEVELS = [0.0, 0.1, 0.2, 0.4]

NOISE_TYPES = ["label", "text", "structural"]

LOGREG_CONFIG = {
    "max_iter": 1000,
    "C": 1.0,
    "solver": "lbfgs",
    "max_features": 50000,
    "ngram_range": (1, 2),
}

DISTILBERT_CONFIG = {
    "model_name": "distilbert-base-uncased",
    "max_length": 128,
    "batch_size": 32,
    "epochs": 3,
    "learning_rate": 2e-5,
    "warmup_ratio": 0.1,
}

QUANTITY_VS_QUALITY = {
    "noisy_size": 50000,
    "noisy_noise_level": 0.3,
    "clean_size": 20000,
}

CLEANING_CONFIG = {
    "confidence_threshold": 0.85,
    "loss_percentile": 80,
    "min_token_length": 3,
}
