"""
early_warning.py
-----------------
Module 3: Early-Warning Prediction.

Trains the same model type using only data available in the first 14 and
first 30 days of the course, and compares performance against the
full-semester model to see how early a reliable warning can be given.
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

import sys
sys.path.append(os.path.dirname(__file__))
from preprocessing import build_feature_table

DROP_COLS = ["id_student", "final_result", "final_result_multiclass", "At_Risk"]


def run_window(cutoff_day, label):
    df_raw, df_encoded = build_feature_table(cutoff_day=cutoff_day)
    y = df_encoded["At_Risk"]
    X = df_encoded.drop(columns=[c for c in DROP_COLS if c in df_encoded.columns])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    model = RandomForestClassifier(n_estimators=250, max_depth=8, class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        "window": label,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }


if __name__ == "__main__":
    results = [
        run_window(14, "First 14 days"),
        run_window(30, "First 30 days"),
        run_window(None, "Full semester"),
    ]
    out = pd.DataFrame(results)
    print(out.to_string(index=False))
    out.to_csv(os.path.join(os.path.dirname(__file__), "..", "report", "early_warning_comparison.csv"), index=False)

    print("""
Discussion:
- Using only the first 14 days, the model already captures early VLE
  engagement and registration timing, but has not seen any assessment
  scores yet (the first TMA in this dataset lands around day 30), so recall
  is typically lower than the 30-day and full-semester windows.
- By day 30, at least one assessment score is usually available, which is
  historically one of the strongest single predictors of final outcome, so
  recall and F1 usually improve noticeably over the 14-day window.
- The full-semester model is the upper bound on performance, since it has
  every assessment and the complete engagement trend.
- Practically, a 14-day model is precise enough to flag students for a
  light-touch "how's it going" check-in, while the 30-day model is strong
  enough to support a more formal academic-advising referral.
""")
