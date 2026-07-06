"""
eda.py
------
Task 3: Exploratory Data Analysis. Produces the 8 required visualizations
and saves them as PNGs in report/figures/.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt

BASE = os.path.join(os.path.dirname(__file__), "..")
FIG_DIR = os.path.join(BASE, "report", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

df = pd.read_csv(os.path.join(BASE, "data", "processed", "student_features.csv"))


def savefig(name):
    path = os.path.join(FIG_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()
    print("saved", path)


# 1. Distribution of final results
plt.figure(figsize=(6, 4))
df["final_result"].value_counts().plot(kind="bar", color="#4C72B0")
plt.title("Distribution of Final Results")
plt.ylabel("Number of students")
savefig("01_final_result_distribution.png")

# 2. Final result by gender
plt.figure(figsize=(6, 4))
pd.crosstab(df["gender"], df["final_result"], normalize="index").plot(kind="bar", stacked=True)
plt.title("Final Result by Gender")
plt.ylabel("Proportion")
savefig("02_final_result_by_gender.png")

# 3. Final result by education level
plt.figure(figsize=(7, 4))
pd.crosstab(df["highest_education"], df["final_result"], normalize="index").plot(kind="bar", stacked=True)
plt.title("Final Result by Highest Education")
plt.ylabel("Proportion")
plt.xticks(rotation=30, ha="right")
savefig("03_final_result_by_education.png")

# 4. Average assessment score by result
plt.figure(figsize=(6, 4))
df.groupby("final_result")["avg_assessment_score"].mean().plot(kind="bar", color="#55A868")
plt.title("Average Assessment Score by Final Result")
plt.ylabel("Average score")
savefig("04_avg_score_by_result.png")

# 5. VLE activity by result
plt.figure(figsize=(6, 4))
df.groupby("final_result")["total_clicks"].mean().plot(kind="bar", color="#C44E52")
plt.title("Average Total VLE Clicks by Final Result")
plt.ylabel("Average total clicks")
savefig("05_vle_activity_by_result.png")

# 6. Student engagement over time proxy (early vs total engagement by result)
plt.figure(figsize=(6, 4))
eng = df.groupby("final_result")[["early_engagement", "total_clicks"]].mean()
eng.plot(kind="bar")
plt.title("Early Engagement vs Total Engagement by Result")
plt.ylabel("Clicks")
savefig("06_engagement_over_time.png")

# 7. Correlation matrix (numeric features)
plt.figure(figsize=(8, 6))
numeric = df.select_dtypes("number").drop(columns=["id_student"], errors="ignore")
corr = numeric.corr()
im = plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(im)
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=7)
plt.yticks(range(len(corr.columns)), corr.columns, fontsize=7)
plt.title("Correlation Matrix of Numeric Features")
savefig("07_correlation_matrix.png")

# 8. Class distribution chart (At_Risk)
plt.figure(figsize=(5, 4))
df["At_Risk"].value_counts().rename({0: "Not At Risk", 1: "At Risk"}).plot(kind="pie", autopct="%1.1f%%")
plt.title("At-Risk Class Distribution")
plt.ylabel("")
savefig("08_class_distribution.png")

print("\nAll 8 required EDA visualizations generated in:", FIG_DIR)
