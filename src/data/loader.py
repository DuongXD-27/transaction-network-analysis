import os
import pandas as pd
import torch
import numpy as np
from torch_geometric.data import Data
from src.utils.config import FEATURES_CSV_PATH, CLASSES_CSV_PATH, EDGELIST_CSV_PATH, TIME_SPLITS, DATA_PROCESSED_DIR

N_LOCAL_FEATURES = 93
N_AGG_FEATURES = 72


def build_feature_columns() -> list[str]:
    cols = ["txId", "time_step"]
    cols += [f"local_{i}" for i in range(N_LOCAL_FEATURES)]
    cols += [f"agg_{i}" for i in range(N_AGG_FEATURES)]
    return cols


def read_elliptic_csv(features_path: str, classes_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_cols = build_feature_columns()
    df_features = pd.read_csv(features_path, header=None, names=feature_cols)
    df_classes = pd.read_csv(classes_path)

    df_features["txId"] = df_features["txId"].astype("int64")
    df_classes["txId"] = df_classes["txId"].astype("int64")
    df_classes["class"] = df_classes["class"].astype(str)

    return df_features, df_classes


def load_and_prep_tabular_data(
    features_path=FEATURES_CSV_PATH,
    classes_path=CLASSES_CSV_PATH,
) -> pd.DataFrame:
    df_features, df_classes = read_elliptic_csv(str(features_path), str(classes_path))

    df = df_features.merge(df_classes, on="txId", how="inner")
    df = df[df["class"] != "unknown"].copy()
    df["y"] = df["class"].map({"1": 1, "2": 0})
    return df


def load_graph_data(force_rebuild=False) -> Data:
    processed_data_path = DATA_PROCESSED_DIR / "pyg_data.pt"

    if processed_data_path.exists() and not force_rebuild:
        print(f"Loading cache graph data from {processed_data_path}...")
        return torch.load(processed_data_path)

    print("Building PyG Data object from scratch")

    df_features, df_classes = read_elliptic_csv(
        str(FEATURES_CSV_PATH), str(CLASSES_CSV_PATH)
    )

    all_tx_ids = df_features["txId"].values
    tx_to_idx = {tx: idx for idx, tx in enumerate(all_tx_ids)}

    matrix_cols = [c for c in df_features.columns if c not in ["txId", "time_step"]]
    x = torch.tensor(df_features[matrix_cols].values, dtype=torch.float)

    df_edges = pd.read_csv(EDGELIST_CSV_PATH)
    df_edges.columns = ["txId1", "txId2"]
    mask_valid = df_edges["txId1"].isin(tx_to_idx) & df_edges["txId2"].isin(tx_to_idx)
    df_edges = df_edges[mask_valid]

    src = df_edges["txId1"].map(tx_to_idx).values
    dst = df_edges["txId2"].map(tx_to_idx).values
    edge_index = torch.tensor(np.array([src, dst]), dtype=torch.long)

    y = torch.full((len(all_tx_ids),), fill_value=-1, dtype=torch.long)
    label_map = {"1": 1, "2": 0}
    for _, row in df_classes.iterrows():
        tx_id = row["txId"]
        cls = row["class"]
        if tx_id in tx_to_idx and cls in label_map:
            y[tx_to_idx[tx_id]] = label_map[cls]

    time_step = torch.tensor(df_features["time_step"].values, dtype=torch.long)
    train_start, train_end = TIME_SPLITS["train"]
    val_start, val_end = TIME_SPLITS["val"]
    test_start, test_end = TIME_SPLITS["test"]

    train_mask = (time_step >= train_start) & (time_step <= train_end) & (y != -1)
    val_mask   = (time_step >= val_start)   & (time_step <= val_end)   & (y != -1)
    test_mask  = (time_step >= test_start)  & (time_step <= test_end)  & (y != -1)

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )

    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(data, processed_data_path)
    print(f"Graph data saved to {processed_data_path}")

    return data


if __name__ == "__main__":
    data = load_graph_data()