from pathlib import Path

BASE_DIR           = Path(__file__).resolve().parent.parent.parent
DATA_RAW_DIR       = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
SAVED_MODELS_DIR   = BASE_DIR / "saved_models"

FEATURES_CSV_PATH        = DATA_RAW_DIR / "elliptic_txs_features.csv"
CLASSES_CSV_PATH         = DATA_RAW_DIR / "elliptic_txs_classes.csv"
EDGELIST_CSV_PATH        = DATA_RAW_DIR / "elliptic_txs_edgelist.csv"

GRAPH_FEATURES_CSV_PATH  = DATA_PROCESSED_DIR / "elliptic_txs_graph_features.csv"
PYG_DATA_PATH            = DATA_PROCESSED_DIR / "elliptic_pyg_data.pt"

TIME_SPLITS = {
    "train": (1, 34),
    "val":   (35, 42),
    "test":  (43, 49),
}
