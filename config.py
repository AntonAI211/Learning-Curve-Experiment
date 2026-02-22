"""
Zentrale Konfiguration für das Lernkurven-Experiment.
Alle Hyperparameter an einem Ort.
"""

import torch

# ── Device ──────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# ── Dataset ─────────────────────────────────────────────────────────
DATASET_ROOT = "./data"
NUM_CLASSES = 10
IMAGE_SIZE = 32
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

# Datenaufteilung aus den 50.000 CIFAR-10 Trainingsbildern:
# 48.000 als Training-Pool (davon je Subset der Größe n)
# 2.000 als fester Validierungssatz (für Early Stopping / LR-Scheduling)
# Die originalen 10.000 Testbilder bleiben unverändert
TRAIN_SIZE = 48000
VAL_SIZE = 2000

# ── Trainingsgrößen (logarithmisch verteilt) ────────────────────────
TRAINING_SIZES = [100, 250, 500, 1000, 2500, 5000, 10000, 25000, 48000]

# Anzahl Partitionen pro Trainingsgröße
NUM_PARTITIONS = 5  # k=5, außer für n=48000 → k=1

# ── Modellwahl ──────────────────────────────────────────────────────
# "SimpleCNN" oder "ResNet18"
MODEL_NAME = "ResNet18"

# ── Optimizer ───────────────────────────────────────────────────────
# "Adam" oder "SGD"
OPTIMIZER_NAME = "Adam"

# Adam
ADAM_LR = 0.001

# SGD
SGD_LR = 0.1
SGD_MOMENTUM = 0.9
SGD_WEIGHT_DECAY = 5e-4

# ── Training ────────────────────────────────────────────────────────
BATCH_SIZE = 32  # Reduziert von 128 → 64 → 32 für maximale Speichersicherheit
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10

# Learning Rate Scheduler (ReduceLROnPlateau)
LR_SCHEDULER_FACTOR = 0.5
LR_SCHEDULER_PATIENCE = 5

# Dropout
DROPOUT_CONV = 0.25
DROPOUT_FC = 0.5

# ── Reproduzierbarkeit ──────────────────────────────────────────────
BASE_SEED = 42

def get_seed(n: int, k: int) -> int:
    """Deterministischer Seed für jedes (n, k)-Paar."""
    return n * 100 + k

# ── Power-Law-Fit ───────────────────────────────────────────────────
# Mindest-Trainingsgröße für den Fit (kleine n brechen das Power-Law)
FIT_MIN_N = 250

# Startwerte für curve_fit: A_inf, eta, gamma
FIT_P0 = [0.9, -10.0, -0.5]
# Bounds: (lower, upper) für (A_inf, eta, gamma)
FIT_BOUNDS = ([0.0, -1e6, -1.0], [0.97, 0.0, 0.0])

# ── Pfade ───────────────────────────────────────────────────────────
RESULTS_DIR = "./results"
PLOTS_DIR = "./plots"
INDIVIDUAL_RESULTS_FILE = "individual_runs.json"
AGGREGATED_RESULTS_FILE = "aggregated_results.csv"
