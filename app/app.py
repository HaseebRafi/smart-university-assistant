"""
app.py
------
Module 7: Application and Dashboard (Streamlit).

Run with:  streamlit run app/app.py    (from the project root)
"""
import os
import sys
import json
import joblib
import pandas as pd
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from chatbot import FAQChatbot
from recommender import ResourceRecommender
from explainability import explain_student, load_everything as load_model_everything

BASE = os.path.join(os.path.dirname(__file__), "..")

st.set_page_config(page_title="Smart University Assistant", layout="wide")

# ---- Light / dark palettes for the "Academic Ledger" theme ----
PALETTES = {
    "dark": {
        "bg": "#10192E", "surface": "#1B2740", "sidebar-bg": "#0C1424",
        "border": "#2A3A5C", "accent": "#C9A24B", "btn-text": "#10192E",
        "text": "#EDE6D6", "muted": "#8B96A8",
        "high": "#B3413A", "high-bg": "rgba(179,65,58,0.15)", "high-text": "#E8B4AE",
        "med": "#C9A24B", "med-bg": "rgba(201,162,75,0.15)", "med-text": "#E8D9AE",
        "low": "#6B9080", "low-bg": "rgba(107,144,128,0.15)", "low-text": "#BFD8CD",
    },
    "light": {
        "bg": "#FAF8F3", "surface": "#FFFFFF", "sidebar-bg": "#F1ECDD",
        "border": "#DED5BE", "accent": "#8A6A1F", "btn-text": "#FFFFFF",
        "text": "#1E2433", "muted": "#5B6472", "newtext": "#FFFFFF",
        "high": "#9C3A32", "high-bg": "rgba(156,58,50,0.08)", "high-text": "#7A2C26",
        "med": "#8A6A1F", "med-bg": "rgba(138,106,31,0.10)", "med-text": "#6B5217",
        "low": "#3F6656", "low-bg": "rgba(63,102,86,0.10)", "low-text": "#2F4D42",
    },
}

if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "dark"

# toggle lives at the very top of the sidebar so it's the first thing visible
_mode_label = st.sidebar.toggle(
    "☀️ Light mode", value=(st.session_state.theme_mode == "light"), key="_theme_toggle"
)
st.session_state.theme_mode = "light" if _mode_label else "dark"
_palette = PALETTES[st.session_state.theme_mode]

_root_vars = "; ".join(f"--{k}: {v}" for k, v in _palette.items())
st.markdown(f"<style>:root {{ {_root_vars}; }}</style>", unsafe_allow_html=True)

# ---- inject the custom "Academic Ledger" theme (app/style.css) ----
_css_path = os.path.join(os.path.dirname(__file__), "style.css")
if os.path.exists(_css_path):
    with open(_css_path) as _f:
        st.markdown(f"<style>{_f.read()}</style>", unsafe_allow_html=True)


def eyebrow(text: str):
    """Small tracked-out label above a section header, e.g. 'STUDENT RECORD'."""
    st.markdown(f'<div class="eyebrow">{text}</div>', unsafe_allow_html=True)


def risk_badge_html(level: str) -> str:
    cls = level.lower()
    return f'<span class="risk-badge {cls}"><span class="dot"></span>{level} risk</span>'


@st.cache_data
def load_features():
    return pd.read_csv(os.path.join(BASE, "data", "processed", "student_features.csv"))


@st.cache_resource
def load_chatbot():
    return FAQChatbot()


@st.cache_resource
def load_recommender():
    return ResourceRecommender()


@st.cache_resource
def get_model_bundle():
    """Loads the encoded feature table + model ONCE per app instance and keeps
    it in memory. This is what makes student selection fast: without this,
    every click would re-read the (potentially 30,000+ row) encoded CSV from
    disk and re-deserialize the model from scratch, which is what was causing
    the app to hang on Student Dashboard / Prediction / Recommendations."""
    return load_model_everything()


df = load_features()

# Initialize page session state for navigation
if "page" not in st.session_state:
    st.session_state.page = "Home"

st.sidebar.title("Smart University Assistant")
st.sidebar.markdown('<div class="sidebar-nav-title">Navigate</div>', unsafe_allow_html=True)

pages = [
    "Home", "Student Dashboard", "Prediction", "Learning Recommendations",
    "University Chatbot", "Analytics"
]

for p in pages:
    btn_type = "primary" if st.session_state.page == p else "secondary"
    if st.sidebar.button(p, type=btn_type, use_container_width=True):
        st.session_state.page = p
        st.rerun()

page = st.session_state.page

# ---------------------------------------------------------------- Home
if page == "Home":
    eyebrow("Project Overview")
    st.title("AI-Powered Smart University Assistant")
    st.markdown("""
    **Purpose:** identify academically at-risk students early, explain *why*
    they were flagged, recommend personalized learning resources, and answer
    common student questions through a chatbot — all in one integrated tool.
    """)
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.subheader("Dataset summary")
            st.write(f"- {df.shape[0]} students")
            st.write(f"- {df['code_module'].nunique() if 'code_module' in df else 'N/A'} course modules")
            st.write(f"- At-risk rate: {df['At_Risk'].mean():.1%}")
    with col2:
        with st.container(border=True):
            st.subheader("Student Info")
            st.write("- Student Name : Abdul Haseeb")
            st.write("- Student Portal ID : ACA942")
            st.write("- Course: Artificial Intelligence")

# ---------------------------------------------------------------- Student Dashboard
elif page == "Student Dashboard":
    eyebrow("Student Record")
    st.title("Student Dashboard")
    student_id = st.selectbox("Select a student", df["id_student"].unique())
    row = df[df.id_student == student_id].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg. assessment score", f"{row['avg_assessment_score']:.1f}")
    c2.metric("Total VLE clicks", int(row["total_clicks"]))
    c3.metric("Active days", int(row["active_days"]))
    c4.metric("Final result", row["final_result"])

    df_m, X_m, y_m, model_m, feature_cols_m = get_model_bundle()
    row_idx = df_m.index[df_m.id_student == student_id][0]
    proba = model_m.predict_proba(X_m.loc[[row_idx]])[0][1]
    st.metric("Predicted risk probability", f"{proba:.1%}")
    level = "High" if proba >= 0.66 else ("Medium" if proba >= 0.33 else "Low")
    st.markdown(risk_badge_html(level), unsafe_allow_html=True)

    exp = explain_student(int(student_id), df_m, X_m, model_m, feature_cols_m)
    st.subheader("Why this prediction?")
    st.markdown(f'<div class="answer-card">{exp["explanation_text"]}</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------- Prediction
elif page == "Prediction":
    eyebrow("Risk Assessment")
    st.title("Student Risk Prediction")
    df_m, X_m, y_m, model_m, feature_cols_m = get_model_bundle()
    mode = st.radio("Input mode", ["Select existing student", "Enter manually"])

    if mode == "Select existing student":
        student_id = st.selectbox("Student", df["id_student"].unique())
        if st.button("Predict risk"):
            exp = explain_student(int(student_id), df_m, X_m, model_m, feature_cols_m)
            level = "High" if exp["risk_probability"] >= 0.66 else ("Medium" if exp["risk_probability"] >= 0.33 else "Low")
            st.metric("Prediction", exp["prediction"])
            st.metric("Risk probability", f"{exp['risk_probability']:.1%}")
            st.markdown(risk_badge_html(level), unsafe_allow_html=True)
            st.markdown(f'<div class="answer-card">{exp["explanation_text"]}</div>', unsafe_allow_html=True)
    else:
        st.info("Manual entry maps your inputs onto the closest feature template for a live demo prediction.")
        avg_score = st.slider("Average assessment score", 0, 100, 55)
        total_clicks = st.slider("Total VLE clicks", 0, 3000, 400)
        prev_attempts = st.slider("Number of previous attempts", 0, 3, 0)
        if st.button("Predict (manual)"):
            approx = df.iloc[(df["avg_assessment_score"] - avg_score).abs().argsort()[:1]]
            exp = explain_student(int(approx.iloc[0]["id_student"]), df_m, X_m, model_m, feature_cols_m)
            level = "High" if exp["risk_probability"] >= 0.66 else ("Medium" if exp["risk_probability"] >= 0.33 else "Low")
            st.write("Closest matching profile in the dataset was used for this demo prediction:")
            st.metric("Prediction", exp["prediction"])
            st.metric("Risk probability", f"{exp['risk_probability']:.1%}")
            st.markdown(risk_badge_html(level), unsafe_allow_html=True)
            st.markdown(f'<div class="answer-card">{exp["explanation_text"]}</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------- Learning Recommendations
elif page == "Learning Recommendations":
    eyebrow("Personalized Guidance")
    st.title("Personalized Learning Recommendations")
    student_id = st.selectbox("Select a student", df["id_student"].unique(), key="rec_student")
    row = df[df.id_student == student_id].iloc[0]
    topic_hint = st.text_input("Weak topic (optional hint)", "Classification")

    rec = load_recommender()
    result = rec.recommend_for_student(row, topic_hint=topic_hint, top_n=3)

    st.markdown(risk_badge_html(result["risk_level"]), unsafe_allow_html=True)
    st.write(f"**Identified need:** {result['identified_need']}")
    st.write("**Recommended resources:**")
    st.table(pd.DataFrame(result["recommended_resources"]))
    st.markdown(f'<div class="answer-card">{result["explanation"]}</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------- University Chatbot
elif page == "University Chatbot":
    # Initialize session state for chatbot history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        
    bot = load_chatbot()
    
    # Header row with title and clear button
    col_title, col_clear = st.columns([6, 1.2])
    with col_title:
        st.markdown("""
        <div class="premium-header">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" class="premium-chat-icon" style="color: #C9A24B; vertical-align: middle; display: inline-block;">
                <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
            </svg>
            <h1 class="premium-page-title" style="display: inline-block; margin: 0; padding-bottom: 0; border: none; margin-left: 10px; vertical-align: middle;">University FAQ Assistant</h1>
        </div>
        """, unsafe_allow_html=True)
    with col_clear:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
            
    st.markdown("<div class='premium-divider'></div>", unsafe_allow_html=True)
    
    # If chat history is empty, show the welcome and suggestions
    if not st.session_state.chat_history:
        st.markdown("""
        <div class="premium-welcome">
            <h1 class="premium-gradient-text">Hello, Student</h1>
            <h2 class="premium-subtitle">How can I help you today?</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Suggestion cards
        suggestions = [
            ("Admissions", "What documents are required for admission?"),
            ("Fee instalments", "Can I pay my tuition fee in instalments?"),
            ("Attendance requirement", "What is the minimum attendance requirement?"),
            ("Internship updates", "Where can I find internship opportunities?"),
        ]
        
        cols = st.columns(4)
        for idx, (label, question_text) in enumerate(suggestions):
            with cols[idx]:
                card_content = f"**{label}**\n\n{question_text}"
                if st.button(card_content, key=f"sug_{idx}", use_container_width=True):
                    # Add user message
                    st.session_state.chat_history.append({"role": "user", "content": question_text})
                    # Get response
                    res = bot.ask(question_text)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": res["answer"],
                        "confidence": res["confidence"],
                        "category": res["category"],
                        "source": res["source"]
                    })
                    st.rerun()
    else:
        # Display chat message history
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                with st.chat_message("user"):
                    st.write(msg["content"])
            else:
                with st.chat_message("assistant", avatar="✨"):
                    st.write(msg["content"])
                    
                    # Add sources and metadata in a beautiful subcard
                    conf = msg.get("confidence", 0.0)
                    category = msg.get("category", "")
                    source = msg.get("source", "")
                    
                    if category or source:
                        st.markdown(f"""
                        <div class="premium-meta-card">
                            <span class="premium-meta-badge">Confidence: {conf:.2f}</span>
                            <span class="premium-meta-badge">Category: {category}</span>
                            <span class="premium-meta-badge">Source: {source}</span>
                        </div>
                        """, unsafe_allow_html=True)
                        
    # Chat input to get user query
    if prompt := st.chat_input("Ask a university-related question..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        res = bot.ask(prompt)
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": res["answer"],
            "confidence": res["confidence"],
            "category": res["category"],
            "source": res["source"]
        })
        st.rerun()

# ---------------------------------------------------------------- Analytics
elif page == "Analytics":
    st.title("Analytics")
    st.subheader("Student result distribution")
    st.bar_chart(df["final_result"].value_counts())

    comp_path = os.path.join(BASE, "models", "model_comparison.json")
    if os.path.exists(comp_path):
        with open(comp_path) as f:
            comparison = json.load(f)
        st.subheader("Model comparison")
        st.dataframe(pd.DataFrame(comparison)[["model", "accuracy", "precision", "recall", "f1", "roc_auc"]])

    fig_path = os.path.join(BASE, "report", "figures", "09_global_feature_importance.png")
    if os.path.exists(fig_path):
        st.subheader("Global feature importance")
        st.image(fig_path)

    st.subheader("Engagement trend (avg. clicks/day by result)")
    st.bar_chart(df.groupby("final_result")["avg_clicks_per_active_day"].mean())
