import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


def generate_fraud_dataset(n_samples: int = 12000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    amount = rng.exponential(scale=28000, size=n_samples) + 500
    txn_hour = rng.integers(0, 24, size=n_samples)
    is_city_changed = rng.choice([0, 1], size=n_samples, p=[0.86, 0.14])
    account_balance = rng.uniform(25000, 1200000, size=n_samples)
    balance_drain_ratio = np.clip(amount / account_balance, 0.005, 1.0)
    velocity_1h = rng.poisson(lam=1.3, size=n_samples)

    risk_signal = (
        (amount > 150000).astype(float) * 0.42
        + ((txn_hour >= 0) & (txn_hour <= 5)).astype(float) * 0.24
        + (is_city_changed == 1).astype(float) * 0.28
        + (balance_drain_ratio > 0.70).astype(float) * 0.34
        + (velocity_1h >= 4).astype(float) * 0.25
        + rng.normal(0, 0.06, size=n_samples)
    )
    is_fraud = (risk_signal > 0.68).astype(int)

    return pd.DataFrame({
        "amount": amount,
        "txn_hour": txn_hour,
        "is_city_changed": is_city_changed,
        "balance_drain_ratio": balance_drain_ratio,
        "velocity_1h": velocity_1h,
        "is_fraud": is_fraud,
    })


def generate_loan_dataset(n_samples: int = 12000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    monthly_income = rng.integers(40000, 600000, size=n_samples)
    account_balance = rng.integers(10000, 2000000, size=n_samples)
    requested_amount = rng.integers(50000, 3500000, size=n_samples)
    existing_debt = rng.integers(0, 500000, size=n_samples)
    tenure_months = rng.choice([12, 24, 36, 48, 60], size=n_samples)
    avg_monthly_txns = rng.integers(2, 45, size=n_samples)

    monthly_emi = (requested_amount + existing_debt) / tenure_months
    dti_ratio = monthly_emi / monthly_income
    liquidity_ratio = account_balance / (requested_amount + 1)

    default_signal = (
        (dti_ratio > 0.42).astype(float) * 0.48
        + (liquidity_ratio < 0.15).astype(float) * 0.28
        + (existing_debt > monthly_income * 2.5).astype(float) * 0.26
        + (avg_monthly_txns < 5).astype(float) * 0.14
        + rng.normal(0, 0.06, size=n_samples)
    )
    is_default = (default_signal > 0.56).astype(int)

    return pd.DataFrame({
        "monthly_income": monthly_income,
        "account_balance": account_balance,
        "requested_amount": requested_amount,
        "existing_debt": existing_debt,
        "tenure_months": tenure_months,
        "avg_monthly_txns": avg_monthly_txns,
        "is_default": is_default,
    })


def evaluate_binary_classifier(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, feature_names: list) -> dict:
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, y_pred).tolist()
    clf = pipeline.named_steps["classifier"]
    importances = {
        feat: round(float(imp) * 100, 2)
        for feat, imp in zip(feature_names, clf.feature_importances_)
    }

    return {
        "accuracy": round(float(accuracy_score(y_test, y_pred)) * 100, 2),
        "precision": round(float(precision_score(y_test, y_pred)) * 100, 2),
        "recall": round(float(recall_score(y_test, y_pred)) * 100, 2),
        "f1_score": round(float(f1_score(y_test, y_pred)) * 100, 2),
        "roc_auc": round(float(roc_auc_score(y_test, y_prob)) * 100, 2),
        "confusion_matrix": {
            "tn": cm[0][0], "fp": cm[0][1],
            "fn": cm[1][0], "tp": cm[1][1],
        },
        "feature_importance": importances,
    }


def train_and_export_all():
    print("⚙️  [1/2] Training Enterprise Fraud Detection Pipeline (RandomForest + StandardScaler)...")
    df_fraud = generate_fraud_dataset()
    fraud_features = ["amount", "txn_hour", "is_city_changed", "balance_drain_ratio", "velocity_1h"]
    Xf, yf = df_fraud[fraud_features], df_fraud["is_fraud"]
    Xf_train, Xf_test, yf_train, yf_test = train_test_split(Xf, yf, test_size=0.2, stratify=yf, random_state=42)

    fraud_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_split=4, class_weight="balanced", random_state=42, n_jobs=-1
        )),
    ])
    cv_fraud = cross_val_score(fraud_pipeline, Xf_train, yf_train, cv=StratifiedKFold(5), scoring="roc_auc").mean()
    fraud_pipeline.fit(Xf_train, yf_train)
    fraud_metrics = evaluate_binary_classifier(fraud_pipeline, Xf_test, yf_test, fraud_features)
    fraud_metrics["cv_roc_auc"] = round(float(cv_fraud) * 100, 2)
    fraud_metrics["dataset_size"] = len(df_fraud)
    joblib.dump(fraud_pipeline, os.path.join(ARTIFACTS_DIR, "fraud_pipeline.pkl"))

    print("⚙️  [2/2] Training Smart Loan Risk Pipeline (GradientBoosting + StandardScaler)...")
    df_loan = generate_loan_dataset()
    loan_features = ["monthly_income", "account_balance", "requested_amount", "existing_debt", "tenure_months", "avg_monthly_txns"]
    Xl, yl = df_loan[loan_features], df_loan["is_default"]
    Xl_train, Xl_test, yl_train, yl_test = train_test_split(Xl, yl, test_size=0.2, stratify=yl, random_state=42)

    loan_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", GradientBoostingClassifier(
            n_estimators=180, learning_rate=0.08, max_depth=5, subsample=0.85, random_state=42
        )),
    ])
    cv_loan = cross_val_score(loan_pipeline, Xl_train, yl_train, cv=StratifiedKFold(5), scoring="roc_auc").mean()
    loan_pipeline.fit(Xl_train, yl_train)
    loan_metrics = evaluate_binary_classifier(loan_pipeline, Xl_test, yl_test, loan_features)
    loan_metrics["cv_roc_auc"] = round(float(cv_loan) * 100, 2)
    loan_metrics["dataset_size"] = len(df_loan)
    joblib.dump(loan_pipeline, os.path.join(ARTIFACTS_DIR, "loan_pipeline.pkl"))

    metrics_payload = {"fraud_model": fraud_metrics, "loan_model": loan_metrics}
    with open(os.path.join(ARTIFACTS_DIR, "model_metrics.json"), "w") as f:
        json.dump(metrics_payload, f, indent=4)

    print(f"✅ Fraud Model -> Accuracy: {fraud_metrics['accuracy']}% | ROC-AUC: {fraud_metrics['roc_auc']}%")
    print(f"✅ Loan Model  -> Accuracy: {loan_metrics['accuracy']}% | ROC-AUC: {loan_metrics['roc_auc']}%")
    print(f"📦 Artifacts saved inside: {ARTIFACTS_DIR}")


if __name__ == "__main__":
    train_and_export_all()