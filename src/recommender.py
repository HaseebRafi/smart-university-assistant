"""
recommender.py
---------------
Module 5: Personalized Learning Resource Recommendation (hybrid).

Rule-based layer: uses risk level, assessment score, and engagement level to
pick a difficulty band and a "need" category.
Content-based layer: uses TF-IDF over resource topic+description to pick the
specific resource within that band that best matches the student's weak topic.
"""
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE = os.path.join(os.path.dirname(__file__), "..")
RESOURCES_PATH = os.path.join(BASE, "data", "learning_resources.csv")
FEATURES_PATH = os.path.join(BASE, "data", "processed", "student_features.csv")


class ResourceRecommender:
    def __init__(self, resources_path: str = RESOURCES_PATH, features_path: str = FEATURES_PATH):
        self.resources = pd.read_csv(resources_path)
        self.resources["text"] = (self.resources["topic"] + " " + self.resources["description"]).str.lower()
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(self.resources["text"])

        # Engagement rules are defined relative to the class, not a fixed magic
        # number, since "total clicks" scale differs across courses/datasets.
        ref = pd.read_csv(features_path)
        self.engagement_low_threshold = ref["total_clicks"].quantile(0.35)

    def _risk_level(self, row) -> str:
        if row["At_Risk"] == 1 and row["avg_assessment_score"] < 40:
            return "High"
        elif row["At_Risk"] == 1 or row["avg_assessment_score"] < 55:
            return "Medium"
        return "Low"

    def _identify_need(self, row) -> tuple[str, str]:
        """Returns (need_description, target_difficulty)."""
        if row["avg_assessment_score"] < 50:
            return "Low assessment performance", "Beginner"
        if row["avg_assessment_score"] >= 80:
            return "Advanced performance", "Advanced"
        if row["total_clicks"] < self.engagement_low_threshold:
            return "Low platform engagement", "Beginner"
        if row["assessment_trend"] < -5:
            return "Declining assessment performance", "Intermediate"
        return "Steady, on-track performance", "Intermediate"

    def recommend_for_student(self, student_row: pd.Series, topic_hint: str = "Classification", top_n: int = 1) -> dict:
        risk = self._risk_level(student_row)
        need, difficulty = self._identify_need(student_row)

        # content-based: find resources at the right difficulty whose topic/description
        # best match the hinted weak topic (in a full system, topic_hint would come from
        # per-topic score breakdowns rather than a single overall average).
        candidates = self.resources[self.resources.difficulty == difficulty]
        if candidates.empty:
            candidates = self.resources

        query_vec = self.vectorizer.transform([topic_hint.lower()])
        cand_matrix = self.matrix[candidates.index]
        sims = cosine_similarity(query_vec, cand_matrix)[0]
        candidates = candidates.assign(similarity=sims).sort_values("similarity", ascending=False)
        top = candidates.head(top_n)

        explanation = (
            f"The student's risk level is {risk} (avg. assessment score = "
            f"{student_row['avg_assessment_score']:.1f}, total VLE clicks = "
            f"{int(student_row['total_clicks'])}). Identified need: {need.lower()}, so a "
            f"{difficulty.lower()}-level resource was selected."
        )

        return {
            "id_student": int(student_row["id_student"]),
            "risk_level": risk,
            "identified_need": need,
            "recommended_resources": top[["resource_id", "title", "resource_type", "difficulty"]].to_dict("records"),
            "explanation": explanation,
        }


def run_test_cases():
    df = pd.read_csv(FEATURES_PATH)
    rec = ResourceRecommender()

    test_cases = [
        ("High-risk student with weak assessment scores", df.sort_values("avg_assessment_score").iloc[0]),
        ("Medium-risk student with low engagement", df[(df.At_Risk == 0) & (df.total_clicks < df.total_clicks.median())].iloc[0]),
        ("Low-risk student requiring advanced material", df.sort_values("avg_assessment_score", ascending=False).iloc[0]),
        ("Student with missed assessments", df.sort_values("pct_assessments_submitted").iloc[0]),
        ("Student with a weakness in a specific topic", df.sort_values("assessment_trend").iloc[0]),
    ]

    print("=== RECOMMENDATION TEST CASES ===\n")
    results = []
    for label, row in test_cases:
        out = rec.recommend_for_student(row)
        out["test_case"] = label
        results.append(out)
        print(f"[{label}]")
        print(f"  Student {out['id_student']} | Risk: {out['risk_level']} | Need: {out['identified_need']}")
        for r in out["recommended_resources"]:
            print(f"  -> {r['title']} ({r['resource_type']}, {r['difficulty']})")
        print(f"  Explanation: {out['explanation']}\n")

    pd.json_normalize(results).to_csv(os.path.join(BASE, "report", "recommendation_test_cases.csv"), index=False)


if __name__ == "__main__":
    run_test_cases()
