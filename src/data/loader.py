import pandas as pd

from src.config import FEATURES_CSV_PATH, CLASSES_CSV_PATH

N_LOCAL_FEATURES = 93
N_AGG_FEATURES = 72


# Header of feature.csv
def build_feature_columns() -> list[str]:
    cols = ["txId", "time_step"]
    cols += [f"local_{i}" for i in range(N_LOCAL_FEATURES)]
    cols += [f"agg_{i}" for i in range(N_AGG_FEATURES)]
    return cols


# Read features.csv and classes.csv
def read_elliptic_csv(features_path: str, classes_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_cols = build_feature_columns()

    df_features = pd.read_csv(
        features_path,
        header=None,
        names=feature_cols)

    df_classes = pd.read_csv(classes_path)

    df_features["txId"] = df_features["txId"].astype("int64")
    df_classes["txId"] = df_classes["txId"].astype("int64")
    df_classes["class"] = df_classes["class"].astype(str)
    return df_features, df_classes


# Load and prepare tabular data
def load_and_prep_tabular_data(
    features_path=FEATURES_CSV_PATH,
    classes_path=CLASSES_CSV_PATH,
) -> pd.DataFrame:
    df_features, df_classes = read_elliptic_csv(str(features_path), str(classes_path))

    # Merge features and classes by txId
    df = df_features.merge(df_classes, on="txId", how="inner")

    # Remove unknown before training
    n_before = len(df)
    df = df[df["class"] != "unknown"].copy()

    # Remap labels: 1 -> 0, 2 -> 1
    df["y"] = df["class"].map({"1": 1, "2": 0})

    return df

if __name__ == "__main__":
    df = load_and_prep_tabular_data()
