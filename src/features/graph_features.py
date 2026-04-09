import time

import networkx as nx
import pandas as pd

from src.config import (
    EDGELIST_CSV_PATH,
    FEATURES_CSV_PATH,
    GRAPH_FEATURES_CSV_PATH,
)
from src.data.loader import load_and_prep_tabular_data


# Map txId to time_step
def build_tx_time_step_map(df: pd.DataFrame) -> pd.Series:
    mapping = df[["txId", "time_step"]].drop_duplicates("txId")
    mapping = mapping.set_index("txId")["time_step"]
    return mapping


def compute_graph_features_temporal(
    edges: pd.DataFrame,
    tx_time_step_map: pd.Series,
    target_df: pd.DataFrame,
) -> pd.DataFrame:

    all_results = []
    unique_timesteps = sorted(target_df["time_step"].unique())
    
    # Map src_time and dst_time to edges dataframe
    edges_mapped = edges.copy()
    edges_mapped["src_time"] = edges_mapped["txId1"].map(tx_time_step_map)
    edges_mapped["dst_time"] = edges_mapped["txId2"].map(tx_time_step_map)
    
    for t in unique_timesteps:
        # Filter edges with cutoff = t
        edges_t = edges_mapped[
            (edges_mapped["src_time"] == t) & 
            (edges_mapped["dst_time"] == t)
        ]
        
        # Calculate in_degree, out_degree
        in_deg = edges_t.groupby("txId2").size().rename("in_degree")
        out_deg = edges_t.groupby("txId1").size().rename("out_degree")
        
        # Build graph 
        G = nx.from_pandas_edgelist(
            edges_t, 
            source="txId1", 
            target="txId2", 
            create_using=nx.DiGraph()
        )

        # Calculate pagerank, clustering
        pr_dict = nx.pagerank(G, alpha=0.85, max_iter=100, tol=1e-6)
        cc_dict = nx.clustering(G.to_undirected())
        
        # Get target nodes at time_step == t
        target_ids = target_df[target_df["time_step"] == t]["txId"]
        
        # Map features to target_ids
        batch_df = pd.DataFrame({"txId": target_ids})
        batch_df["in_degree"] = batch_df["txId"].map(in_deg).fillna(0).astype("int64")
        batch_df["out_degree"] = batch_df["txId"].map(out_deg).fillna(0).astype("int64")
        batch_df["pagerank"] = batch_df["txId"].map(pr_dict).fillna(0.0)
        batch_df["clustering_coefficient"] = batch_df["txId"].map(cc_dict).fillna(0.0)
        
        all_results.append(batch_df)
    
    return pd.concat(all_results, ignore_index=True)

def main():
    start_time = time.time()

    # 1. Load df without unknown
    df_labeled = load_and_prep_tabular_data()

    # 2. Load edgelist
    edges_df = pd.read_csv(EDGELIST_CSV_PATH)

    # 3. Build tx_time_step_map including unknowns
    df_features_all = pd.read_csv(FEATURES_CSV_PATH, header=None, usecols=[0, 1], names=["txId", "time_step"])
    tx_time_step_map = build_tx_time_step_map(df_features_all)

    # 4. Compute temporal graph features once for the entire dataset
    all_graph_features = compute_graph_features_temporal(
        edges=edges_df,
        tx_time_step_map=tx_time_step_map,
        target_df=df_labeled
    )

    # 5. Save
    GRAPH_FEATURES_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    all_graph_features.to_csv(GRAPH_FEATURES_CSV_PATH, index=False)


if __name__ == "__main__":
    main()
