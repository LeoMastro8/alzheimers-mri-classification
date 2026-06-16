from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "adni4" / "data"
SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
SPLIT_CSV = PROJECT_ROOT / "data" / "splits" / "splitting_1775900489.csv"
SPLIT_LAB_CSV = PROJECT_ROOT / "data" / "splits" / "split_labels.csv"
META_CSV = PROJECT_ROOT / "data" / "raw" / "adni4" / "meta" / "data_num.csv"
PREPROCESSED_DIR = PROJECT_ROOT / "data" / "preprocessed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODALITIES = ["T1", "T2", "FLAIR"]
