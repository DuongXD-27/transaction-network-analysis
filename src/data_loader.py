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
    features_path: str = "data/raw/elliptic_txs_features.csv",
    classes_path: str = "data/raw/elliptic_txs_classes.csv",
    include_time_step: bool = True,) -> tuple[pd.DataFrame, pd.Series]:
    df_features, df_classes = read_elliptic_csv(features_path, classes_path)

    # Merge features and classes by txId
    df = df_features.merge(df_classes, on="txId", how="inner")

    # Remove unknown before training
    df = df[df["class"] != "unknown"].copy()

    # Remap labels: 1 -> 0, 2 -> 1
    df["y"] = df["class"].map({"1": 0, "2": 1})

    # Choose feature columns for X
    x_cols = [c for c in df_features.columns if c != "txId"]
    if not include_time_step:
        x_cols = [c for c in x_cols if c != "time_step"]

    X = df[x_cols].copy()
    y = df["y"].astype("int8")

    if y.isna().any():
        raise ValueError("Label mapping failed: still has NaN in y.")
    if not y.isin([0, 1]).all():
        raise ValueError("y must contain only 0/1.")

    return X, y

if __name__ == "__main__":
    X, y = load_and_prep_tabular_data()
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Label distribution:")
    print(y.value_counts())