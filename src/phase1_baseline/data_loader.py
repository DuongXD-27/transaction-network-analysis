import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from utils.config import FEATURES_CSV_PATH, CLASSES_CSV_PATH

import pandas as pd


N_LOCAL_FEATURES = 93
N_AGG_FEATURES = 72

# Add feature columns to feature.csv
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

    print("\n=== Merge diagnostics (before merge) ===")
    print(
        "features:",
        f"rows={len(df_features):,}",
        f"unique_txId={df_features['txId'].nunique():,}",
        f"dup_txId_rows={df_features.duplicated('txId').sum():,}",
    )
    print(
        "classes:",
        f"rows={len(df_classes):,}",
        f"unique_txId={df_classes['txId'].nunique():,}",
        f"dup_txId_rows={df_classes.duplicated('txId').sum():,}",
    )

    # Merge features and classes by txId
    df = df_features.merge(df_classes, on="txId", how="inner")

    print("\n=== Merge diagnostics (after merge) ===")
    print(
        "merged:",
        f"rows={len(df):,}",
        f"unique_txId={df['txId'].nunique():,}",
    )
    merged_unique = df["txId"].nunique()
    features_unique = df_features["txId"].nunique()
    classes_unique = df_classes["txId"].nunique()

    print(
        "lost_unique_txId_from_features_missing_in_classes=",
        f"{features_unique - merged_unique:,}",
    )
    print(
        "lost_unique_txId_from_classes_missing_in_features=",
        f"{classes_unique - merged_unique:,}",
    )

    # Remove unknown before training
    n_before = len(df)
    df = df[df["class"] != "unknown"].copy()
    print("\n=== After removing unknown ===")
    print(
        f"removed: {n_before - len(df):,} rows ({(n_before - len(df)) / n_before * 100:.1f}%)"
    )
    print(f"remaining: {len(df):,} rows")

    # Remap labels: 1 -> 0, 2 -> 1
    df["y"] = df["class"].map({"1": 0, "2": 1})

    print("label distribution:\n" + str(df["y"].value_counts()))

    if df["y"].isna().any():
        raise ValueError("Label mapping failed: still has NaN in y.")
    if not df["y"].isin([0, 1]).all():
        raise ValueError("y must contain only 0/1.")

    return df

if __name__ == "__main__":
    df = load_and_prep_tabular_data()
    print("df shape:", df.shape)
    print("time_step min/max:", df["time_step"].min(), df["time_step"].max())
    print("label distribution:")
    print(df["y"].value_counts())