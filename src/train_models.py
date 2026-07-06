"""
train_models.py
----------------
Task 4: Machine-Learning Model Development.

Trains and compares:
  1. Logistic Regression
  2. Random Forest (ensemble)
  3. Gradient Boosting (ensemble)
with stratified train/test split, 5-fold cross-validation, hyperparameter
tuning (GridSearchCV) for Random Forest and Gradient Boosting, and full
evaluation (accuracy, precision, recall, F1, ROC-AUC, confusion matrix).

Saves the best model + the fitted preprocessing column list to models/.
"""
import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              roc_auc_score, confusion_matrix, classification_report)

BASE = os.path.join(os.path.dirname(__file__), "..")
MODELS_DIR = os.path.join(BASE, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

DROP_COLS = ["id_student", "final_result", "final_result_multiclass", "At_Risk"]


def load_data():
    df = pd.read_csv(os.path.join(BASE, "data", "processed", "student_features_encoded.csv"))
    y = df["At_Risk"]
    X = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    return X, y


def evaluate(name, model, X_test, y_test, scaler=None):
    Xt = scaler.transform(X_test) if scaler is not None else X_test
    y_pred = model.predict(Xt)
    y_proba = model.predict_proba(Xt)[:, 1]
    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    return metrics


def main():
    X, y = load_data()
    feature_cols = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    # ---- Model 1: Logistic Regression ----
    lr = LogisticRegression(max_iter=2000, class_weight="balanced")
    lr.fit(X_train_scaled, y_train)
    cv_scores = cross_val_score(lr, X_train_scaled, y_train, cv=cv, scoring="f1")
    m = evaluate("Logistic Regression", lr, X_test, y_test, scaler=scaler)
    m["cv_f1_mean"] = cv_scores.mean()
    results.append(m)

    # ---- Model 2: Random Forest (ensemble) -- hyperparameter tuned ----
    rf_grid = {
        "n_estimators": [150, 300],
        "max_depth": [6, 10, None],
        "min_samples_leaf": [1, 3],
    }
    rf = GridSearchCV(RandomForestClassifier(class_weight="balanced", random_state=42),
                       rf_grid, cv=cv, scoring="f1", n_jobs=-1)
    rf.fit(X_train, y_train)  # tree models don't need scaling
    best_rf = rf.best_estimator_
    m = evaluate("Random Forest (tuned, ensemble)", best_rf, X_test, y_test)
    m["cv_f1_mean"] = rf.best_score_
    m["best_params"] = rf.best_params_
    results.append(m)

    # ---- Model 3: Gradient Boosting (ensemble) -- hyperparameter tuned ----
    gb_grid = {
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1],
        "max_depth": [2, 3],
    }
    gb = GridSearchCV(GradientBoostingClassifier(random_state=42),
                       gb_grid, cv=cv, scoring="f1", n_jobs=-1)
    gb.fit(X_train, y_train)
    best_gb = gb.best_estimator_
    m = evaluate("Gradient Boosting (tuned, ensemble)", best_gb, X_test, y_test)
    m["cv_f1_mean"] = gb.best_score_
    m["best_params"] = gb.best_params_
    results.append(m)

    # ---- Compare & select final model ----
    # Recall on the at-risk class matters more than raw accuracy: missing an
    # at-risk student (false negative) has a higher real-world cost than a
    # false alarm, since a false alarm just means an advisor checks in on a
    # student who turns out to be fine. We therefore select using F1 as the
    # primary balance metric, but report recall prominently.
    results_sorted = sorted(results, key=lambda r: r["f1"], reverse=True)
    best = results_sorted[0]
    best_name = best["model"]
    best_model = {"Logistic Regression": lr,
                  "Random Forest (tuned, ensemble)": best_rf,
                  "Gradient Boosting (tuned, ensemble)": best_gb}[best_name]
    needs_scaler = best_name == "Logistic Regression"

    joblib.dump(best_model, os.path.join(MODELS_DIR, "risk_prediction_model.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(feature_cols, os.path.join(MODELS_DIR, "feature_columns.pkl"))
    with open(os.path.join(MODELS_DIR, "model_meta.json"), "w") as f:
        json.dump({"best_model": best_name, "needs_scaler": needs_scaler}, f, indent=2)

    with open(os.path.join(MODELS_DIR, "model_comparison.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\n=== MODEL COMPARISON ===")
    for r in results:
        print(f"{r['model']:38s} | acc={r['accuracy']:.3f} prec={r['precision']:.3f} "
              f"recall={r['recall']:.3f} f1={r['f1']:.3f} roc_auc={r['roc_auc']:.3f} "
              f"cv_f1={r['cv_f1_mean']:.3f}")

    print(f"\nSelected final model: {best_name}")
    print("Saved model + scaler + metadata to", MODELS_DIR)


if __name__ == "__main__":
    main()
