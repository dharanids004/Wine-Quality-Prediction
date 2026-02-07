from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    recall_score,
)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from features import FeatureConfig, build_features, clean_and_interpolate, make_target_from_failure


def load_config(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train_test_split_time(df: pd.DataFrame, time_col: str, id_col: str, test_size: float):
    df = df.sort_values([id_col, time_col])
    train_idx = []
    test_idx = []
    for _, g in df.groupby(id_col):
        n = g.shape[0]
        if n < 5:
            train_idx.extend(g.index)
            continue
        cut = int(np.floor((1 - test_size) * n))
        train_idx.extend(g.index[:cut])
        test_idx.extend(g.index[cut:])
    return df.loc[train_idx], df.loc[test_idx]


def build_baseline_model():
    return Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=False)),
            (
                "clf",
                LogisticRegression(
                    max_iter=200,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
            ),
        ]
    )


def build_xgb_model(params: Dict, scale_pos_weight: float):
    params = params.copy()
    params["scale_pos_weight"] = scale_pos_weight
    return XGBClassifier(**params)


def evaluate(y_true, y_pred, name: str) -> Dict:
    f1 = f1_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    return {
        "name": name,
        "f1": f1,
        "recall": rec,
        "report": report,
        "confusion_matrix": cm.tolist(),
    }


def write_classification_report(report: Dict, reports_dir: Path, prefix: str) -> None:
    report_path_json = reports_dir / f"{prefix}_classification_report.json"
    report_path_csv = reports_dir / f"{prefix}_classification_report.csv"

    with open(report_path_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    df = pd.DataFrame(report).T
    df.to_csv(report_path_csv)


def main():
    project_root = Path(__file__).resolve().parents[1]
    cfg = load_config(project_root / "src" / "config.yaml")

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    out_cfg = cfg["outputs"]

    data_path = project_root / data_cfg["path"]
    artifacts_dir = project_root / out_cfg["artifacts_dir"]
    reports_dir = project_root / out_cfg["reports_dir"]
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)

    feature_cfg = FeatureConfig(
        time_col=data_cfg["time_col"],
        id_col=data_cfg["id_col"],
        target_col=data_cfg["target_col"],
        failure_event_col=data_cfg["failure_event_col"],
        lead_time_hours=int(data_cfg["lead_time_hours"]),
        feature_cols=data_cfg.get("feature_cols") or None,
        drop_cols=data_cfg.get("drop_cols") or None,
    )

    df = clean_and_interpolate(df, feature_cfg.time_col, feature_cfg.id_col)

    # Create target if missing
    if feature_cfg.target_col not in df.columns:
        if feature_cfg.failure_event_col not in df.columns:
            raise ValueError(
                f"Missing target_col '{feature_cfg.target_col}' and failure_event_col '{feature_cfg.failure_event_col}'."
            )
        df = make_target_from_failure(
            df,
            feature_cfg.time_col,
            feature_cfg.id_col,
            feature_cfg.failure_event_col,
            feature_cfg.target_col,
            feature_cfg.lead_time_hours,
        )

    df, feature_cols = build_features(df, feature_cfg)

    # Drop rows with NaNs created by shifting/rolling
    df = df.dropna(subset=feature_cols + [feature_cfg.target_col])

    # Correlation matrix for numeric features
    corr = df[feature_cols].corr()
    corr.to_csv(reports_dir / "corr_matrix.csv")

    train_df, test_df = train_test_split_time(
        df, feature_cfg.time_col, feature_cfg.id_col, model_cfg["test_size"]
    )

    X_train = train_df[feature_cols]
    y_train = train_df[feature_cfg.target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[feature_cfg.target_col]

    # Baseline: Logistic Regression
    baseline = build_baseline_model()
    baseline.fit(X_train, y_train)
    base_pred = baseline.predict(X_test)
    baseline_metrics = evaluate(y_test, base_pred, "logreg_baseline")
    write_classification_report(
        baseline_metrics["report"], reports_dir, "baseline"
    )

    # Main model: XGBoost with imbalance handling
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    scale_pos_weight = float(neg / max(1, pos))

    xgb = build_xgb_model(model_cfg["xgb_params"], scale_pos_weight)

    if model_cfg.get("use_smote", True):
        smote = SMOTE(
            sampling_strategy="auto",
            k_neighbors=int(model_cfg.get("smote_k_neighbors", 5)),
            random_state=int(model_cfg.get("random_state", 42)),
        )
        model = ImbPipeline(steps=[("smote", smote), ("xgb", xgb)])
    else:
        model = xgb

    # Optional randomized search
    if model_cfg.get("random_search", {}).get("enabled", False):
        params = {
            "xgb__max_depth": [3, 4, 5, 6, 8],
            "xgb__learning_rate": [0.01, 0.03, 0.05, 0.1],
            "xgb__subsample": [0.7, 0.8, 0.9, 1.0],
            "xgb__colsample_bytree": [0.6, 0.8, 1.0],
            "xgb__min_child_weight": [1, 2, 5],
            "xgb__n_estimators": [200, 400, 600],
        }
        if not model_cfg.get("use_smote", True):
            # Adjust param keys if no SMOTE pipeline
            params = {k.replace("xgb__", ""): v for k, v in params.items()}
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=params,
            n_iter=int(model_cfg.get("random_search", {}).get("n_iter", 20)),
            scoring=model_cfg.get("random_search", {}).get("scoring", "f1"),
            cv=int(model_cfg.get("random_search", {}).get("cv", 3)),
            n_jobs=-1,
            verbose=1,
            random_state=int(model_cfg.get("random_state", 42)),
        )
        search.fit(X_train, y_train)
        model = search.best_estimator_
        best_params = search.best_params_
    else:
        model.fit(X_train, y_train)
        best_params = None

    pred = model.predict(X_test)
    xgb_metrics = evaluate(y_test, pred, "xgb")
    write_classification_report(xgb_metrics["report"], reports_dir, "xgb")

    metrics = {
        "baseline": baseline_metrics,
        "xgb": xgb_metrics,
        "best_params": best_params,
    }

    with open(reports_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Save artifacts
    joblib.dump(model, artifacts_dir / "model.joblib")
    joblib.dump(feature_cols, artifacts_dir / "feature_cols.joblib")
    joblib.dump(feature_cfg, artifacts_dir / "feature_cfg.joblib")

    # SHAP explainability (optional, can be heavy)
    try:
        import shap

        # Take a sample for speed
        sample = X_test.sample(min(500, X_test.shape[0]), random_state=42)
        if hasattr(model, "named_steps"):
            explainer = shap.TreeExplainer(model.named_steps["xgb"])
            shap_values = explainer.shap_values(sample)
        else:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(sample)

        shap.summary_plot(shap_values, sample, show=False)
        import matplotlib.pyplot as plt

        plt.tight_layout()
        plt.savefig(reports_dir / "shap_summary.png", dpi=150)
        plt.close()

        # Force plot for a single sample
        fp = shap.force_plot(
            explainer.expected_value,
            shap_values[0],
            sample.iloc[0],
            matplotlib=False,
        )
        shap.save_html(str(reports_dir / "shap_force.html"), fp)
    except Exception as exc:
        with open(reports_dir / "shap_error.txt", "w", encoding="utf-8") as f:
            f.write(str(exc))

    print("Training complete.")
    print(f"Baseline F1: {baseline_metrics['f1']:.4f} | Recall: {baseline_metrics['recall']:.4f}")
    print(f"XGB F1: {xgb_metrics['f1']:.4f} | Recall: {xgb_metrics['recall']:.4f}")


if __name__ == "__main__":
    main()
