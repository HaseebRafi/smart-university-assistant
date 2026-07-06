"""
preprocessing.py
-----------------
Module 1 (Task 2): Data Preparation and Feature Engineering.

- Loads the raw OULAD-style CSVs
- Cleans missing / duplicated values
- Aggregates VLE clicks and assessment performance per student
- Merges everything into one student-level feature table
- Encodes categoricals, defines the At_Risk target
- Saves data/processed/student_features.csv
"""
import os
import pandas as pd
import numpy as np

BASE = os.path.join(os.path.dirname(__file__), "..")
RAW = os.path.join(BASE, "data", "raw")
PROCESSED = os.path.join(BASE, "data", "processed")
os.makedirs(PROCESSED, exist_ok=True)


def load_raw():
    student_info = pd.read_csv(os.path.join(RAW, "studentInfo.csv"))
    registration = pd.read_csv(os.path.join(RAW, "studentRegistration.csv"))
    assessments = pd.read_csv(os.path.join(RAW, "assessments.csv"))
    student_assessment = pd.read_csv(os.path.join(RAW, "studentAssessment.csv"))
    student_vle = pd.read_csv(os.path.join(RAW, "studentVle.csv"))
    return student_info, registration, assessments, student_assessment, student_vle


def clean_student_info(student_info: pd.DataFrame) -> pd.DataFrame:
    before = len(student_info)
    student_info = student_info.drop_duplicates(subset=["id_student"]).copy()
    dupes_removed = before - len(student_info)

    # Missing value handling: imd_band is missing-not-at-random for some students.
    # We fill with the mode per region, which keeps the distribution realistic
    # rather than silently dropping rows (data loss) or filling with a single global constant.
    student_info["imd_band"] = student_info.groupby("region")["imd_band"].transform(
        lambda s: s.fillna(s.mode().iloc[0] if not s.mode().empty else "Unknown")
    )
    print(f"[clean_student_info] removed {dupes_removed} duplicate student rows; "
          f"filled missing imd_band using region-mode imputation")
    return student_info


def engineer_vle_features(student_vle: pd.DataFrame, cutoff_day: int | None = None) -> pd.DataFrame:
    """Aggregate VLE interactions per student. If cutoff_day is set, only
    interactions up to that day are used -> this powers the early-warning model (Module 3)."""
    df = student_vle.copy()
    if cutoff_day is not None:
        df = df[df["date"] <= cutoff_day]

    agg = df.groupby("id_student").agg(
        total_clicks=("sum_click", "sum"),
        active_days=("date", "nunique"),
        resource_diversity=("id_site", "nunique"),
    ).reset_index()
    agg["avg_clicks_per_active_day"] = agg["total_clicks"] / agg["active_days"].replace(0, np.nan)
    agg["avg_clicks_per_active_day"] = agg["avg_clicks_per_active_day"].fillna(0)

    # early engagement: clicks in first 14 days vs total (only meaningful for full-window calls)
    early = df[df["date"] <= 14].groupby("id_student")["sum_click"].sum().rename("early_engagement")
    agg = agg.merge(early, on="id_student", how="left")
    agg["early_engagement"] = agg["early_engagement"].fillna(0)
    return agg


def engineer_assessment_features(student_assessment: pd.DataFrame, assessments: pd.DataFrame,
                                  cutoff_day: int | None = None) -> pd.DataFrame:
    df = student_assessment.merge(assessments, on="id_assessment", how="left")
    if cutoff_day is not None:
        df = df[df["date_submitted"] <= cutoff_day]

    if df.empty:
        return pd.DataFrame(columns=["id_student", "n_assessments_attempted", "avg_assessment_score",
                                      "min_assessment_score", "submission_delay", "assessment_trend"])

    agg = df.groupby("id_student").agg(
        n_assessments_attempted=("id_assessment", "nunique"),
        avg_assessment_score=("score", "mean"),
        min_assessment_score=("score", "min"),
        submission_delay=("date_submitted", lambda s: (s - df.loc[s.index, "date"]).mean()),
    ).reset_index()

    # performance trend: score of the last attempted assessment minus the first (chronologically)
    def trend(g):
        g = g.sort_values("date")
        if len(g) < 2:
            return 0.0
        return g["score"].iloc[-1] - g["score"].iloc[0]

    trend_series = df.groupby("id_student").apply(trend, include_groups=False)
    trend_series = trend_series.rename("assessment_trend").reset_index()
    agg = agg.merge(trend_series, on="id_student", how="left")
    agg["assessment_trend"] = agg["assessment_trend"].fillna(0)
    return agg


def build_feature_table(cutoff_day: int | None = None) -> pd.DataFrame:
    """cutoff_day=None -> full-semester features (Module 2).
    cutoff_day=14/30 -> early-warning features (Module 3)."""
    student_info, registration, assessments, student_assessment, student_vle = load_raw()
    student_info = clean_student_info(student_info)

    vle_feats = engineer_vle_features(student_vle, cutoff_day=cutoff_day)
    assess_feats = engineer_assessment_features(student_assessment, assessments, cutoff_day=cutoff_day)

    total_possible_assessments = assessments.groupby(["code_module", "code_presentation"])["id_assessment"] \
        .count().rename("n_assessments_in_module").reset_index()

    df = student_info.merge(vle_feats, on="id_student", how="left")
    df = df.merge(assess_feats, on="id_student", how="left")
    df = df.merge(registration[["id_student", "date_registration"]], on="id_student", how="left")
    df = df.merge(total_possible_assessments, on=["code_module", "code_presentation"], how="left")

    # fill students with zero recorded activity (they exist, they just didn't interact)
    fill_zero_cols = ["total_clicks", "active_days", "resource_diversity", "avg_clicks_per_active_day",
                       "early_engagement", "n_assessments_attempted", "avg_assessment_score",
                       "min_assessment_score", "submission_delay", "assessment_trend"]
    for c in fill_zero_cols:
        df[c] = df[c].fillna(0)

    df["pct_assessments_submitted"] = (df["n_assessments_attempted"] /
                                        df["n_assessments_in_module"].replace(0, np.nan)).fillna(0).clip(0, 1)
    df["registration_duration"] = -df["date_registration"]  # more negative = registered earlier

    # target definitions
    df["At_Risk"] = df["final_result"].isin(["Fail", "Withdrawn"]).astype(int)
    df["final_result_multiclass"] = df["final_result"]

    # encode categoricals
    cat_cols = ["gender", "region", "highest_education", "imd_band", "age_band", "disability",
                "code_module", "code_presentation"]
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    return df, df_encoded


if __name__ == "__main__":
    df_raw, df_encoded = build_feature_table(cutoff_day=None)
    out_path = os.path.join(PROCESSED, "student_features.csv")
    df_raw.to_csv(out_path, index=False)
    df_encoded.to_csv(os.path.join(PROCESSED, "student_features_encoded.csv"), index=False)

    print("\nFinal feature table shape:", df_raw.shape)
    print("Class balance (At_Risk):\n", df_raw["At_Risk"].value_counts(normalize=True).round(3))
    print("\nSaved:", out_path)
