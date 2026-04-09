import joblib

import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)
from xgboost import XGBClassifier

from src.config import GRAPH_FEATURES_CSV_PATH, SAVED_MODELS_DIR, TIME_SPLITS
from src.data.loader import load_and_prep_tabular_data


def run_one_experiment(
    train_df,
    val_df,
    test_df,
    feature_cols,
    exp_name: str,
):
    X_train, y_train = train_df[feature_cols], train_df["y"]
    X_val, y_val = val_df[feature_cols], val_df["y"]
    X_test, y_test = test_df[feature_cols], test_df["y"]

    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    scale_pos_weight = n_neg / n_pos

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
        early_stopping_rounds=20,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False,
    )

    y_pred = model.predict(X_test)

    recall_illicit = recall_score(y_test, y_pred, pos_label=1)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n=== {exp_name} ===")
    print(f"n_features: {len(feature_cols)}")
    print(f"Recall (illicit=1): {recall_illicit:.4f}")
    print(f"F1-macro: {f1_macro:.4f}")
    print("Confusion matrix:")
    print(cm)
    print("Classification report:")
    print(classification_report(y_test, y_pred, digits=4))

    # Save model artifact
    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = exp_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("&", "and")
    model_path = SAVED_MODELS_DIR / f"xgboost_{safe_name}.pkl"
    joblib.dump(model, model_path)
    print(f"Model saved → {model_path}")

    return {
        "experiment": exp_name,
        "n_features": len(feature_cols),
        "recall_illicit": recall_illicit,
        "f1_macro": f1_macro,
        "model_path": str(model_path),
    }


def main() -> None:
    # 1. Load baseline tabular data
    df = load_and_prep_tabular_data()

    # 2. Load and merge graph features
    print("\nLoading graph features...")
    graph_df = pd.read_csv(GRAPH_FEATURES_CSV_PATH)
    df = df.merge(graph_df, on="txId", how="left")

    # 3. Time-based split
    train_start, train_end = TIME_SPLITS["train"]
    val_start, val_end = TIME_SPLITS["val"]
    test_start, test_end = TIME_SPLITS["test"]

    train_df = df[(df["time_step"] >= train_start) & (df["time_step"] <= train_end)].copy()
    val_df = df[(df["time_step"] >= val_start) & (df["time_step"] <= val_end)].copy()
    test_df = df[(df["time_step"] >= test_start) & (df["time_step"] <= test_end)].copy()

    # 4. Define Feature Sets
    exclude_cols = ["txId", "class", "y", "time_step", "in_degree", "out_degree", "pagerank", "clustering_coefficient"]
    base_feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    graph_feature_cols = ["in_degree", "out_degree", "pagerank", "clustering_coefficient"]

    # Ver A: Local + Agg
    feature_cols_A = base_feature_cols

    # Ver B: Local + Agg + time_step
    feature_cols_B = base_feature_cols + ["time_step"]

    # Ver C: Local + Agg + Graph + time_step
    feature_cols_C = base_feature_cols + graph_feature_cols + ["time_step"]

    # 5. Run Experiments
    result_A = run_one_experiment(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        feature_cols=feature_cols_A,
        exp_name="Version A (without time_step)",
    )

    result_B = run_one_experiment(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        feature_cols=feature_cols_B,
        exp_name="Version B (with time_step)",
    )

    result_C = run_one_experiment(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        feature_cols=feature_cols_C,
        exp_name="Version C (with Graph Features & time_step)",
    )

    print("\n=== Ablation Summary ===")
    print(result_A)
    print(result_B)
    print(result_C)


if __name__ == "__main__":
    main()
