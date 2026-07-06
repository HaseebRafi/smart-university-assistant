# AI-Powered Smart University Assistant

A working prototype covering all required modules of the assignment:
data preparation, EDA, academic-risk prediction, early-warning prediction,
a university FAQ chatbot, a personalized learning-resource recommender,
explainable AI, and an integrated Streamlit dashboard.

## ⚠️ About the data (read this first)

This repo ships with **synthetic data that mimics OULAD's structure**
(`src/generate_synthetic_data.py`), because the real dataset requires an
internet download. **Before you submit**, replace the files in `data/raw/`
with the real OULAD CSVs from
https://analyse.kmi.open.ac.uk/open_dataset — the column names match, so
`preprocessing.py`, `train_models.py`, etc. will work unchanged. The
`university_faq.csv` and `learning_resources.csv` files are real, usable
content — but you should still edit them to match your own institution's
actual policies (the assignment requires this).

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run the pipeline (in this order)

```bash
# 1. Generate synthetic raw data (skip this once you have the real OULAD CSVs)
python src/generate_synthetic_data.py

# 2. Create the FAQ and learning-resource datasets (edit the source lists first!)
python src/generate_faq_data.py
python src/generate_resources_data.py

# 3. Clean, merge, and engineer features -> data/processed/student_features.csv
python src/preprocessing.py

# 4. Exploratory data analysis -> report/figures/*.png
python src/eda.py

# 5. Train & compare classification models -> models/risk_prediction_model.pkl
python src/train_models.py

# 6. Early-warning comparison (14-day / 30-day / full-semester)
python src/early_warning.py

# 7. Explainable AI (global + local explanations) -> report/global_explanation.json
python src/explainability.py

# 8. Try the chatbot + run its 30-question evaluation
python src/chatbot.py

# 9. Try the recommender + run its 5 test cases
python src/recommender.py

# 10. Launch the full application
streamlit run app/app.py
```

## Project structure

```
smart-university-assistant/
├── data/
│   ├── raw/                     # OULAD-style CSVs (synthetic, replace with real data)
│   ├── processed/               # cleaned + engineered feature table
│   ├── university_faq.csv       # 100 FAQ records across 10 categories
│   └── learning_resources.csv   # 60 learning resources
├── src/
│   ├── generate_synthetic_data.py
│   ├── generate_faq_data.py
│   ├── generate_resources_data.py
│   ├── preprocessing.py         # Module 1: cleaning + feature engineering
│   ├── eda.py                   # Module 1: 8 required visualizations
│   ├── train_models.py          # Module 2: risk prediction models
│   ├── early_warning.py         # Module 3: early-warning comparison
│   ├── chatbot.py                # Module 4: FAQ chatbot
│   ├── recommender.py            # Module 5: recommendation system
│   └── explainability.py         # Module 6: global + local explanations
├── app/
│   └── app.py                    # Module 7: Streamlit dashboard (6 pages)
├── models/                       # saved model, scaler, metadata
├── report/                       # figures, evaluation CSVs, JSON explanations
├── requirements.txt
└── README.md
```

## Where each assignment task lives

| Task | File(s) |
|---|---|
| Task 1: Problem/dataset understanding | this README + report you write |
| Task 2: Data prep & feature engineering | `src/preprocessing.py` |
| Task 3: EDA | `src/eda.py` → `report/figures/01-08*.png` |
| Task 4: ML models | `src/train_models.py`, `src/early_warning.py` |
| Task 5: FAQ chatbot | `src/chatbot.py`, `data/university_faq.csv` |
| Task 6: Recommender | `src/recommender.py`, `data/learning_resources.csv` |
| Task 7: Explainable AI | `src/explainability.py` → `report/figures/09*.png` |
| Task 8: Application | `app/app.py` |
| Task 9: Ethics & limitations | see below |

## Ethics, privacy & limitations (Task 9 starting point)

- **Privacy:** student IDs in OULAD are already anonymized; do not attempt to
  re-identify students, and store any real institutional data under access
  control, not in a public GitHub repo.
- **Informed consent:** if you deploy this against real, current students,
  they should know their VLE/assessment data feeds a risk model.
- **Algorithmic bias / fairness:** the model can pick up proxies for
  disadvantage (region, IMD band, prior attempts). Bonus Task 8
  (fairness analysis across demographic groups) is a good way to check this
  — compare recall/false-negative rate across gender, disability status, and
  IMD band, not just overall accuracy.
- **Misclassification risk:** recall on the at-risk class matters more than
  raw accuracy, since a missed at-risk student (false negative) has a higher
  cost than a false alarm — but false alarms aren't free either (advisor time,
  student anxiety at being flagged).
- **Human oversight:** predictions are decision-support only. An advisor,
  not the model, should make the final call on any intervention.
- **Chatbot limitations:** the TF-IDF chatbot in this prototype is the
  *basic* approach from the spec. As the evaluation shows, it correctly
  declines ambiguous paraphrases rather than guessing — safer, but it does
  mean genuine questions sometimes hit the fallback. Upgrading to the
  *intermediate* approach (Sentence Transformers / semantic embeddings) would
  close this gap and is a natural extension for extra marks.

## Notes on the synthetic data generator

The synthetic generator gives every student a hidden "ability" score that
drives their assessment results, VLE engagement, and final outcome — so the
relationships in the data (e.g. engagement correlates with passing) are
realistic enough to build and sanity-check every module end-to-end. It is
**not** a substitute for running the real OULAD data before submission.
