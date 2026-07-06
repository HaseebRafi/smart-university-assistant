"""
explainability.py
------------------
Module 6: Explainable Artificial Intelligence.

Global explanation: Random Forest feature importance (+ permutation importance
as a cross-check) across the whole test set.
Local explanation: for one student, shows which of their own feature values
pushed the prediction toward "at risk", using the trained tree ensemble's
per-feature contribution (a lightweight, dependency-free stand-in for SHAP —
swap in the `shap` library directly if it's installed in your environment).
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

BASE = os.path.join(os.path.dirname(__file__), "..")
MODELS_DIR = os.path.join(BASE, "models")
FIG_DIR = os.path.join(BASE, "report", "figures")
DROP_COLS = ["id_student", "final_result", "final_result_multiclass", "At_Risk"]


def load_everything():
    df = pd.read_csv(os.path.join(BASE, "data", "processed", "student_features_encoded.csv"))
    model = joblib.load(os.path.join(MODELS_DIR, "risk_prediction_model.pkl"))
    feature_cols = joblib.load(os.path.join(MODELS_DIR, "feature_columns.pkl"))
    y = df["At_Risk"]
    # reindex (not a direct column selection) so that if the current data is
    # missing a category the model was trained on (e.g. a course code or
    # imd_band bucket that doesn't appear in this run), we fill it with 0
    # instead of crashing. If you see many missing columns printed below,
    # it means the model is stale for this data -- rerun train_models.py.
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"[explainability] warning: {len(missing)} training columns not found in current "
              f"data (filled with 0): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    X = df.reindex(columns=feature_cols, fill_value=0)
    return df, X, y, model, feature_cols


def global_explanation(top_n: int = 10):
    df, X, y, model, feature_cols = load_everything()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # 1. Built-in impurity-based importance (fast, model-specific)
    if hasattr(model, "feature_importances_"):
        importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    else:  # e.g. logistic regression -> use absolute coefficients
        importances = pd.Series(np.abs(model.coef_[0]), index=feature_cols).sort_values(ascending=False)

    # 2. Permutation importance (model-agnostic cross-check on held-out data)
    perm = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
    perm_importances = pd.Series(perm.importances_mean, index=feature_cols).sort_values(ascending=False)

    top_features = importances.head(top_n)
    plt.figure(figsize=(7, 5))
    top_features[::-1].plot(kind="barh", color="#4C72B0")
    plt.title(f"Top {top_n} Global Feature Importances (Random Forest)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "09_global_feature_importance.png"), dpi=130)
    plt.close()

    print("=== GLOBAL EXPLANATION ===")
    print("\nTop features (model-based importance):")
    print(importances.head(top_n).round(4))
    print("\nTop features (permutation importance, held-out data):")
    print(perm_importances.head(top_n).round(4))

    result = {"model_importance": importances.head(top_n).to_dict(),
              "permutation_importance": perm_importances.head(top_n).to_dict()}
    with open(os.path.join(BASE, "report", "global_explanation.json"), "w") as f:
        json.dump(result, f, indent=2)
    return result


def local_explanation(id_student: int, top_n: int = 5) -> dict:
    df, X, y, model, feature_cols = load_everything()
    row_idx = df.index[df.id_student == id_student]
    if len(row_idx) == 0:
        raise ValueError(f"Student {id_student} not found")
    row_idx = row_idx[0]
    x = X.loc[row_idx]

    proba = model.predict_proba(X.loc[[row_idx]])[0][1]
    prediction = "At Risk" if proba >= 0.5 else "Not At Risk"

    # Lightweight local explanation: for tree ensembles, approximate each
    # feature's contribution as importance * how far the student's (scaled)
    # value sits from the training-set mean, in the "risky" direction.
    means = X.mean()
    stds = X.std().replace(0, 1)
    z_scores = (x - means) / stds

    if hasattr(model, "feature_importances_"):
        importances = pd.Series(model.feature_importances_, index=feature_cols)
    else:
        importances = pd.Series(np.abs(model.coef_[0]), index=feature_cols)

    contribution = (importances * z_scores).sort_values()
    # features pushing toward "at risk" (positive z on risk-increasing direction,
    # here we just take the most extreme contributions in either direction)
    top_contributors = contribution.reindex(contribution.abs().sort_values(ascending=False).index).head(top_n)

    readable = []
    for feat, val in top_contributors.items():
        direction = "higher than average" if z_scores[feat] > 0 else "lower than average"
        readable.append(f"{feat.replace('_', ' ')} is {direction} (value={x[feat]:.1f})")

    explanation_text = (
        f"Student {id_student} was classified as '{prediction}' with an at-risk "
        f"probability of {proba:.2f}. The main contributing factors were: " +
        "; ".join(readable) + "."
    )

    print(explanation_text)
    return {
        "id_student": id_student,
        "prediction": prediction,
        "risk_probability": round(float(proba), 3),
        "top_contributing_factors": readable,
        "explanation_text": explanation_text,
    }


if __name__ == "__main__":
    global_explanation()
    print()
    df, *_ = load_everything()
    sample_id = int(df.sort_values("avg_assessment_score").iloc[0]["id_student"])
    local_explanation(sample_id)
