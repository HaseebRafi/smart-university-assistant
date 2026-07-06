"""
generate_synthetic_data.py
--------------------------
Creates synthetic data that mimics the STRUCTURE of the real OULAD dataset
(studentInfo, studentRegistration, studentAssessment, assessments, studentVle, vle, courses).

IMPORTANT FOR THE REAL SUBMISSION:
Download the real OULAD dataset from https://analyse.kmi.open.ac.uk/open_dataset
and drop the real CSVs into data/raw/ using the SAME file names used here.
Every other script (preprocessing, modeling, EDA, etc.) will then work unchanged
on the real data, because they only depend on the column names, not on this generator.

This synthetic version exists only so the pipeline can be built, tested, and
demonstrated right now without an internet connection.
"""
import numpy as np
import pandas as pd
import os

RNG = np.random.default_rng(42)  # fixed seed -> reproducible
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

N_STUDENTS = 2000
MODULES = ["AAA", "BBB", "CCC", "DDD"]
PRESENTATIONS = ["2013B", "2013J", "2014B", "2014J"]
GENDERS = ["M", "F"]
REGIONS = ["North Region", "South Region", "Scotland", "Wales", "London Region", "East Anglian Region"]
EDU = ["No Formal quals", "Lower Than A Level", "A Level or Equivalent", "HE Qualification", "Post Graduate Qualification"]
IMD = ["0-10%", "10-20%", "20-30%", "30-40%", "40-50%", "50-60%", "60-70%", "70-80%", "80-90%", "90-100%"]
AGE = ["0-35", "35-55", "55<="]

student_ids = np.arange(100000, 100000 + N_STUDENTS)

# ---------- studentInfo.csv ----------
rows = []
for sid in student_ids:
    module = RNG.choice(MODULES)
    presentation = RNG.choice(PRESENTATIONS)
    gender = RNG.choice(GENDERS)
    region = RNG.choice(REGIONS)
    education = RNG.choice(EDU, p=[0.08, 0.30, 0.35, 0.22, 0.05])
    imd = RNG.choice(IMD)
    age = RNG.choice(AGE, p=[0.75, 0.22, 0.03])
    prev_attempts = RNG.choice([0, 1, 2, 3], p=[0.75, 0.15, 0.07, 0.03])
    credits = RNG.choice([30, 60, 90, 120], p=[0.4, 0.35, 0.15, 0.10])
    disability = RNG.choice(["N", "Y"], p=[0.9, 0.1])

    # latent "ability/effort" score drives everything downstream (assessments, vle, outcome)
    ability = RNG.normal(0.55, 0.20) - 0.05 * prev_attempts + (0.05 if education in ["HE Qualification", "Post Graduate Qualification"] else 0)
    ability = np.clip(ability, 0.02, 0.98)

    p_withdraw = np.clip(0.28 - 0.30 * ability + 0.05 * prev_attempts, 0.02, 0.55)
    if RNG.random() < p_withdraw:
        final_result = "Withdrawn"
    else:
        score_band = ability + RNG.normal(0, 0.12)
        if score_band > 0.80:
            final_result = "Distinction"
        elif score_band > 0.45:
            final_result = "Pass"
        else:
            final_result = "Fail"

    rows.append([sid, module, presentation, gender, region, education, imd, age,
                 prev_attempts, credits, disability, final_result, ability])

student_info = pd.DataFrame(rows, columns=[
    "id_student", "code_module", "code_presentation", "gender", "region",
    "highest_education", "imd_band", "age_band", "num_of_prev_attempts",
    "studied_credits", "disability", "final_result", "_ability"
])

# inject a small amount of realistic messiness for the cleaning module to catch
dupe_rows = student_info.sample(15, random_state=1)
student_info = pd.concat([student_info, dupe_rows], ignore_index=True)
missing_idx = RNG.choice(student_info.index, size=40, replace=False)
student_info.loc[missing_idx, "imd_band"] = np.nan

student_info.drop(columns=["_ability"]).to_csv(os.path.join(RAW_DIR, "studentInfo.csv"), index=False)
ability_lookup = student_info.drop_duplicates("id_student").set_index("id_student")["_ability"]

# ---------- studentRegistration.csv ----------
reg_rows = []
for sid in student_ids:
    date_registration = int(RNG.integers(-60, 5))
    withdrawn = student_info.loc[student_info.id_student == sid, "final_result"].iloc[0] == "Withdrawn"
    date_unreg = int(RNG.integers(10, 240)) if withdrawn else np.nan
    reg_rows.append([sid, date_registration, date_unreg])
student_registration = pd.DataFrame(reg_rows, columns=["id_student", "date_registration", "date_unregistration"])
student_registration.to_csv(os.path.join(RAW_DIR, "studentRegistration.csv"), index=False)

# ---------- courses.csv ----------
courses = pd.DataFrame([(m, p, int(RNG.choice([234, 262, 269]))) for m in MODULES for p in PRESENTATIONS],
                        columns=["code_module", "code_presentation", "module_presentation_length"])
courses.to_csv(os.path.join(RAW_DIR, "courses.csv"), index=False)

# ---------- assessments.csv ----------
assess_rows = []
aid = 1000
for m in MODULES:
    for p in PRESENTATIONS:
        for i, (atype, date, weight) in enumerate([
            ("TMA", 30, 10), ("TMA", 60, 15), ("TMA", 100, 15),
            ("TMA", 140, 20), ("CMA", 170, 10), ("Exam", 230, 30)
        ]):
            assess_rows.append([m, p, aid, atype, date, weight])
            aid += 1
assessments = pd.DataFrame(assess_rows, columns=["code_module", "code_presentation", "id_assessment", "assessment_type", "date", "weight"])
assessments.to_csv(os.path.join(RAW_DIR, "assessments.csv"), index=False)

# ---------- studentAssessment.csv ----------
sa_rows = []
for _, srow in student_info.drop_duplicates("id_student").iterrows():
    sid = srow.id_student
    ability = ability_lookup[sid]
    module_assessments = assessments[(assessments.code_module == srow.code_module) &
                                      (assessments.code_presentation == srow.code_presentation)]
    n_take = RNG.integers(max(1, len(module_assessments) - 3), len(module_assessments) + 1)
    taken = module_assessments.sample(min(n_take, len(module_assessments)), random_state=int(sid) % (2**31))
    for _, arow in taken.iterrows():
        score = np.clip(RNG.normal(ability * 100, 12), 0, 100)
        submitted_date = arow.date + int(RNG.integers(-5, 15))
        is_banked = RNG.choice([0, 1], p=[0.95, 0.05])
        sa_rows.append([arow.id_assessment, sid, submitted_date, is_banked, round(score, 1)])
student_assessment = pd.DataFrame(sa_rows, columns=["id_assessment", "id_student", "date_submitted", "is_banked", "score"])
student_assessment.to_csv(os.path.join(RAW_DIR, "studentAssessment.csv"), index=False)

# ---------- vle.csv ----------
ACTIVITY_TYPES = ["resource", "oucontent", "url", "forumng", "quiz", "subpage", "homepage", "glossary"]
vle_rows = []
site_id = 500000
for m in MODULES:
    for p in PRESENTATIONS:
        for _ in range(40):
            vle_rows.append([site_id, m, p, RNG.choice(ACTIVITY_TYPES), int(RNG.integers(0, 20)), int(RNG.integers(20, 40))])
            site_id += 1
vle = pd.DataFrame(vle_rows, columns=["id_site", "code_module", "code_presentation", "activity_type", "week_from", "week_to"])
vle.to_csv(os.path.join(RAW_DIR, "vle.csv"), index=False)

# ---------- studentVle.csv ----------
sv_rows = []
for _, srow in student_info.drop_duplicates("id_student").iterrows():
    sid = srow.id_student
    ability = ability_lookup[sid]
    engagement = np.clip(ability + RNG.normal(0, 0.15), 0.05, 1.0)
    module_sites = vle[(vle.code_module == srow.code_module) & (vle.code_presentation == srow.code_presentation)]
    n_days_active = int(RNG.integers(5, 25) * (0.4 + engagement))
    for day in RNG.choice(range(0, 240), size=min(n_days_active, 200), replace=False):
        site = module_sites.sample(1, random_state=int(day) + int(sid) % 1000).iloc[0]
        clicks = int(max(1, RNG.poisson(2 + engagement * 6)))
        sv_rows.append([srow.code_module, srow.code_presentation, sid, site.id_site, int(day), clicks])
student_vle = pd.DataFrame(sv_rows, columns=["code_module", "code_presentation", "id_student", "id_site", "date", "sum_click"])
student_vle.to_csv(os.path.join(RAW_DIR, "studentVle.csv"), index=False)

print("Synthetic OULAD-style raw data written to:", os.path.abspath(RAW_DIR))
for f in os.listdir(RAW_DIR):
    print(" -", f)
