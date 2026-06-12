from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_RAW_DIR = BASE_DIR / "data" / "raw"

FEATURES_CSV_PATH = DATA_RAW_DIR / "elliptic_txs_features.csv"
CLASSES_CSV_PATH = DATA_RAW_DIR / "elliptic_txs_classes.csv"
EDGELIST_CSV_PATH = DATA_RAW_DIR / "elliptic_txs_edgelist.csv"

TIME_SPLITS = {
    "train": (1, 34),
    "val": (35, 42),
    "test": (43, 49),
}
