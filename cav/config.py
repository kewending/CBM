from pathlib import Path

# Model Params
BERT_MODEL_NAME = "bert-base-uncased"
PROBE_LAYER = 9          # default; overridden per-concept if layer sweep finds better
LAYER_SWEEP = [7,8,9,10,11]
SVM_C = 0.01             # low regularisation = stable direction with small data
MAX_SEQ_LEN = 512
BATCH_SIZE = 4

# Validation Params
MIN_CV_ACCURACY = 0.70   # probes below this threshold are flagged as unreliable
CV_FOLDS = 5

# Data Paths
TRAIN_CSV = Path("outputs/par_only/train.csv")
TEST_CSV = Path("outputs/par_only/test.csv")

# Output Paths
EMBED_CACHE_DIR = Path("outputs/cav_llm/embed_cache")
PROBE_OUT_DIR = Path("outputs/cav_llm/probes")
BOTTLENECK_OUT = Path("outputs/cav_llm/bottleneck_matrix.npy")
BOTTLENECK_NAMES = Path("outputs/cav_llm/bottleneck_names.json")
VALIDATION_REPORT = Path("outputs/cav_llm/validation_report.md")

# Ensure output directories exist
EMBED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
PROBE_OUT_DIR.mkdir(parents=True, exist_ok=True)
BOTTLENECK_OUT.parent.mkdir(parents=True, exist_ok=True)
