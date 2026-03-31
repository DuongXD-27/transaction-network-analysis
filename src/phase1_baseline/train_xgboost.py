import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config import TIME_SPLITS

from phase1_baseline.data_loader import load_and_prep_tabular_data

from phase1_baseline.feature_engineering import (
    build_tx_time_step_map,
    compute_graph_features_for_split,
)

from sklearn.metrics import (
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from xgboost import XGBClassifier

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
    if n_pos == 0:
        raise ValueError(f"[{exp_name}] No positive samples in train set!")
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

    return {
        "experiment": exp_name,
        "n_features": len(feature_cols),
        "recall_illicit": recall_illicit,
        "f1_macro": f1_macro,
    }

def main() -> None:
    df = load_and_prep_tabular_data()

    train_start, train_end = TIME_SPLITS["train"]
    val_start, val_end = TIME_SPLITS["val"]
    test_start, test_end = TIME_SPLITS["test"]

    train_df = df[(df["time_step"] >= train_start) & (df["time_step"] <= train_end)].copy()
    val_df = df[(df["time_step"] >= val_start) & (df["time_step"] <= val_end)].copy()
    test_df = df[(df["time_step"] >= test_start) & (df["time_step"] <= test_end)].copy()

    base_feature_cols = [c for c in df.columns if c not in ["txId", "class", "y"]]

    # Version A: X do not include time_step
    feature_cols_A = [c for c in base_feature_cols if c != "time_step"]

    # Version B: X include time_step
    feature_cols_B = base_feature_cols

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

    print("\n=== Ablation Summary ===")
    print(result_A)
    print(result_B)

if __name__ == "__main__":
    main()