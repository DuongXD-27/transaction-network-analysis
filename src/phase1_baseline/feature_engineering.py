import sys
from pathlib import Path
from typing import Iterable

import pandas as pd
import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def build_tx_time_step_map(df: pd.DataFrame) -> pd.Series:
    mapping = df[["txId", "time_step"]].drop_duplicates("txId")
    mapping = mapping.set_index("txId")["time_step"]
    return mapping


def compute_graph_features_for_split(
    edges: pd.DataFrame,
    tx_time_step_map: pd.Series,
    target_tx_ids: Iterable[int],
    cutoff_time_step: int,
) -> pd.DataFrame:
    
    edges_filtered = edges.copy()
    edges_filtered["src_time"] = edges_filtered["txId1"].map(tx_time_step_map)
    edges_filtered["dst_time"] = edges_filtered["txId2"].map(tx_time_step_map)

    edges_filtered = edges_filtered[
        (edges_filtered["src_time"] <= cutoff_time_step)
        & (edges_filtered["dst_time"] <= cutoff_time_step)
    ].dropna(subset=["src_time", "dst_time"])

    in_deg = edges_filtered.groupby("txId2").size().rename("in_degree")
    out_deg = edges_filtered.groupby("txId1").size().rename("out_degree")

    G = nx.from_pandas_edgelist(
        edges_filtered, 
        source="txId1", 
        target="txId2", 
        create_using=nx.DiGraph()
    )

    pr_dict = nx.pagerank(G)
    cc_dict = nx.clustering(G)

    target_df = pd.DataFrame({"txId": list(target_tx_ids)})

    target_df["in_degree"] = target_df["txId"].map(in_deg).fillna(0).astype("int64")
    target_df["out_degree"] = target_df["txId"].map(out_deg).fillna(0).astype("int64")
    
    target_df["pagerank"] = target_df["txId"].map(pr_dict).fillna(0.0)
    target_df["clustering_coefficient"] = target_df["txId"].map(cc_dict).fillna(0.0)

    return target_df