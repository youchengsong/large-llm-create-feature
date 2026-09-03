"""Cross-model sensitivity: evaluate No-FE and KnowFeat with XGBoost, LightGBM, CatBoost on all 12 datasets."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from knowfeat.data.split import get_or_make_split

DATASETS = {
    "credit_g": {"csv": "data/public/credit_g.csv", "target": "class",
                 "feat": "outputs/credit_g/knowfeat_features.csv"},
    "diabetes": {"csv": "data/public/diabetes.csv", "target": "class",
                 "feat": "outputs/diabetes/knowfeat_features.csv"},
    "blood_transfusion": {"csv": "data/public/blood_transfusion.csv", "target": "Class",
                          "feat": "outputs/blood_transfusion/knowfeat_features.csv"},
    "heart": {"csv": "data/public/heart.csv", "target": "HeartDisease",
              "feat": "outputs/heart/knowfeat_features.csv"},
    "bank_marketing": {"csv": "data/public/bank_marketing.csv", "target": "y",
                       "feat": "outputs/bank_marketing/knowfeat_features.csv"},
    "breast_w": {"csv": "data/public/breast_w.csv", "target": "Class",
                 "feat": "outputs/breast_w/knowfeat_features.csv"},
    "adult": {"csv": "data/public/adult.csv", "target": "class",
              "feat": "outputs/adult/knowfeat_features.csv"},
    "wine_quality": {"csv": "data/public/wine_quality.csv", "target": "class",
                     "feat": "outputs/wine_quality/knowfeat_features.csv"},
    "default_credit": {"csv": "data/public/default_credit.csv", "target": "target",
                       "feat": "outputs/default_credit_posthoc/knowfeat_features.csv"},
    "churn": {"csv": "data/public/churn.csv", "target": "target",
              "feat": "outputs/churn_posthoc/knowfeat_features.csv"},
    "kc2": {"csv": "data/public/kc2.csv", "target": "target",
            "feat": "outputs/kc2_posthoc/knowfeat_features.csv"},
    "jm1": {"csv": "data/public/jm1.csv", "target": "target",
            "feat": "outputs/jm1_posthoc/knowfeat_features.csv"},
}

MODEL_SEEDS = [42, 123, 456, 789, 2024]


def prepare_data(ds_name, cfg):
    df = pd.read_csv(PROJECT_ROOT / cfg["csv"])
    target = cfg["target"]
    y = df[target].copy()
    if y.dtype == "object" or not pd.api.types.is_numeric_dtype(y):
        from sklearn.preprocessing import LabelEncoder
        y = pd.Series(LabelEncoder().fit_transform(y.astype(str)))

    X_nfe = df.drop(columns=[target]).copy()
    for col in X_nfe.columns:
        if not pd.api.types.is_numeric_dtype(X_nfe[col]):
            X_nfe[col] = X_nfe[col].astype("category").cat.codes
    X_nfe = X_nfe.fillna(-999)

    feat_df = pd.read_csv(PROJECT_ROOT / cfg["feat"])
    X_kf = X_nfe.copy()
    for col in feat_df.columns:
        if col not in X_kf.columns:
            X_kf[col] = feat_df[col].values

    X_kf = X_kf.fillna(-999).replace([np.inf, -np.inf], -999)

    split_cache = PROJECT_ROOT / "outputs" / "splits" / f"{ds_name}_split.json"
    split = get_or_make_split(len(y), labels=y, cache_path=split_cache)

    return (X_nfe.iloc[split.dev_idx], X_nfe.iloc[split.test_idx],
            X_kf.iloc[split.dev_idx], X_kf.iloc[split.test_idx],
            y.iloc[split.dev_idx], y.iloc[split.test_idx])


def evaluate_model(model_cls, X_dev, y_dev, X_test, y_test, seeds, **model_kwargs):
    from sklearn.metrics import roc_auc_score
    scores = []
    for seed in seeds:
        m = model_cls(random_state=seed, **model_kwargs)
        m.fit(X_dev, y_dev)
        prob = m.predict_proba(X_test)[:, 1]
        scores.append(roc_auc_score(y_test, prob))
    return np.mean(scores)


def main():
    import xgboost as xgb
    import lightgbm as lgb
    from catboost import CatBoostClassifier

    models = {
        "XGBoost": (xgb.XGBClassifier, {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.1,
                                          "verbosity": 0, "eval_metric": "logloss"}),
        "LightGBM": (lgb.LGBMClassifier, {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.1,
                                            "verbosity": -1}),
        "CatBoost": (CatBoostClassifier, {"iterations": 200, "depth": 6, "learning_rate": 0.1,
                                           "verbose": 0}),
    }

    results = {m: {"deltas": [], "datasets": []} for m in models}

    for ds_name, cfg in DATASETS.items():
        print(f"  {ds_name}...", end=" ", flush=True)
        try:
            X_nfe_dev, X_nfe_test, X_kf_dev, X_kf_test, y_dev, y_test = prepare_data(ds_name, cfg)

            for model_name, (cls, kwargs) in models.items():
                nfe_auc = evaluate_model(cls, X_nfe_dev, y_dev, X_nfe_test, y_test, MODEL_SEEDS, **kwargs)
                kf_auc = evaluate_model(cls, X_kf_dev, y_dev, X_kf_test, y_test, MODEL_SEEDS, **kwargs)
                delta = kf_auc - nfe_auc
                results[model_name]["deltas"].append(delta)
                results[model_name]["datasets"].append(ds_name)

            print("OK")
        except Exception as e:
            print(f"FAILED: {e}")
            for m in models:
                results[m]["deltas"].append(None)
                results[m]["datasets"].append(ds_name)

    print(f"\n{'='*60}")
    print("Cross-Model Sensitivity (KnowFeat - No-FE)")
    print(f"{'='*60}")
    print(f"{'Model':<12} {'Wins':<8} {'Mean Δ':<10} {'Median Δ':<10} {'p_Holm':<8}")
    print("-" * 48)

    p_values = []
    for model_name, data in results.items():
        deltas = [d for d in data["deltas"] if d is not None]
        wins = sum(1 for d in deltas if d > 0)
        total = len(deltas)
        mean_d = np.mean(deltas)
        median_d = np.median(deltas)
        _, p = wilcoxon(deltas, alternative="greater")
        p_values.append((model_name, p))
        print(f"{model_name:<12} {wins}/{total}    {mean_d:+.4f}    {median_d:+.4f}    {p:.3f}")

    # Holm correction
    sorted_p = sorted(p_values, key=lambda x: x[1])
    n = len(sorted_p)
    print(f"\nHolm-corrected p-values:")
    for rank, (name, p) in enumerate(sorted_p):
        adj_p = min(p * (n - rank), 1.0)
        print(f"  {name}: {adj_p:.3f}")

    # Save detailed results
    out = PROJECT_ROOT / "outputs" / "cross_model_12datasets.json"
    json.dump(results, open(out, "w"), indent=2, default=str)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
