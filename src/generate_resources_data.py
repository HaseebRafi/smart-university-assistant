"""
generate_resources_data.py
---------------------------
Creates data/learning_resources.csv with 50+ resources spanning AI/ML topics
at Beginner/Intermediate/Advanced difficulty, used by the recommender (Module 5).
"""
import csv
import os

BASE = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(BASE, "data", "learning_resources.csv")

TOPICS = [
    "Classification", "Regression", "Decision Trees", "Model Evaluation", "Feature Engineering",
    "Data Cleaning", "Neural Networks", "Clustering", "Time Management", "Study Planning",
    "Exam Preparation", "Natural Language Processing", "Recommendation Systems", "Explainable AI",
    "Probability Basics", "Statistics for ML", "Python Programming", "SQL for Data Analysis",
    "Ensemble Methods", "Hyperparameter Tuning",
]
DIFFICULTIES = ["Beginner", "Intermediate", "Advanced"]
TYPES = ["Video", "Article", "Notebook", "Tutorial", "Quiz", "Book Chapter", "Practice Exercise", "Lecture Notes"]

TEMPLATES = {
    "Video": ("Introduction to {t}", "A short video walking through the core ideas behind {t_low}."),
    "Article": ("Understanding {t}", "A written explanation covering the fundamentals of {t_low}."),
    "Notebook": ("Hands-on {t} Notebook", "A Jupyter notebook with runnable code exploring {t_low}."),
    "Tutorial": ("Step-by-Step {t} Tutorial", "A guided tutorial that builds {t_low} skills from scratch."),
    "Quiz": ("{t} Self-Check Quiz", "A short quiz to test your understanding of {t_low}."),
    "Book Chapter": ("{t}: A Deeper Look", "An in-depth chapter-style reading on {t_low}."),
    "Practice Exercise": ("{t} Practice Set", "A set of practice problems to reinforce {t_low}."),
    "Lecture Notes": ("{t} Lecture Notes", "Concise lecture notes summarizing {t_low} for revision."),
}

rows = []
rid = 1
module_cycle = ["AAA", "BBB", "CCC", "DDD"]
for i, topic in enumerate(TOPICS):
    # generate ~2-3 resources per topic at varying difficulty/type to comfortably exceed 50
    combos = [
        (DIFFICULTIES[0], TYPES[i % len(TYPES)]),
        (DIFFICULTIES[1], TYPES[(i + 3) % len(TYPES)]),
        (DIFFICULTIES[2], TYPES[(i + 5) % len(TYPES)]),
    ]
    for diff, rtype in combos:
        title_t, desc_t = TEMPLATES[rtype]
        title = title_t.format(t=topic)
        desc = desc_t.format(t_low=topic.lower())
        module = module_cycle[i % len(module_cycle)]
        url = f"https://university-resources.example.edu/{topic.lower().replace(' ', '-')}-{rtype.lower().replace(' ', '-')}"
        rows.append([f"R{rid:03d}", module, topic, diff, rtype, title, desc, url])
        rid += 1

with open(OUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["resource_id", "module", "topic", "difficulty", "resource_type", "title", "description", "url"])
    writer.writerows(rows)

print(f"Wrote {len(rows)} learning resources to {OUT}")
