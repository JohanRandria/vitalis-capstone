import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pickle
import shap
import os
import warnings
warnings.filterwarnings('ignore')

from supabase import create_client, Client

SUPABASE_URL = "https://xvdwmizwwfyhqjesaezk.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inh2ZHdtaXp3d2Z5aHFqZXNhZXprIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzMxMjgwNDUsImV4cCI6MjA4ODcwNDA0NX0.oJ6ULgzwIZZDCiR4a9dPfbJ3GZCPUYM9E6ysf_4Kwds"

@st.cache_resource
def get_supabase() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_KEY)

def save_prediction(data: dict):
    try:
        sb = get_supabase()
        sb.table("predictions").insert(data).execute()
        return True
    except Exception as e:
        st.error(f"Database error: {e}")
        return False

def load_all_predictions():
    try:
        sb = get_supabase()
        res = sb.table("predictions").select("*").order("created_at", desc=True).execute()
        return pd.DataFrame(res.data) if res.data else pd.DataFrame()
    except Exception as e:
        st.error(f"Could not load predictions: {e}")
        return pd.DataFrame()

def load_model_results():
    try:
        sb = get_supabase()
        res = sb.table("model_results").select("*").execute()
        return pd.DataFrame(res.data) if res.data else pd.DataFrame()
    except:
        return pd.DataFrame()

def load_shap_importance():
    try:
        sb = get_supabase()
        res = sb.table("shap_importance").select("*").order("rank").execute()
        return pd.DataFrame(res.data) if res.data else pd.DataFrame()
    except:
        return pd.DataFrame()

def load_dataset_stats():
    try:
        sb = get_supabase()
        res = sb.table("dataset_stats").select("*").limit(1).execute()
        return res.data[0] if res.data else {}
    except:
        return {}

st.set_page_config(
    page_title="Vitalis",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

*, html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

.stApp { background: #f5f5f7; }
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

section[data-testid="stSidebar"] {
    width: 220px !important;
    min-width: 220px !important;
    max-width: 220px !important;
    background: #1d1d1f !important;
    border-right: none !important;
    transform: translateX(0) !important;
}
section[data-testid="stSidebar"] > div:first-child {
    width: 220px !important;
    background: #1d1d1f !important;
}
button[data-testid="baseButton-headerNoPadding"],
button[data-testid="baseButton-header"],
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="stSidebarCollapseButton"],
button[aria-label="Collapse sidebar"],
button[aria-label="collapse sidebar"] {
    display: none !important;
    visibility: hidden !important;
    opacity: 0 !important;
    pointer-events: none !important;
    width: 0 !important;
    height: 0 !important;
}
section[data-testid="stSidebar"] section { padding: 0 !important; }
section[data-testid="stSidebar"] * { color: #8e8e93 !important; }
section[data-testid="stSidebar"] .stRadio > div { gap: 0 !important; }
section[data-testid="stSidebar"] .stRadio label {
    font-size: 14px !important;
    font-weight: 400 !important;
    color: #8e8e93 !important;
    padding: 10px 16px !important;
    border-radius: 8px !important;
    margin: 1px 8px !important;
    display: flex !important;
    align-items: center !important;
    cursor: pointer !important;
    transition: all 0.15s !important;
    border: none !important;
}
section[data-testid="stSidebar"] .stRadio label:hover {
    color: #ffffff !important;
    background: rgba(255,255,255,0.08) !important;
}
section[data-testid="stSidebar"] .stRadio div[data-baseweb="radio"] > div:first-child {
    display: none !important;
}
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label > div:first-child {
    display: none !important;
}

.block-container {
    padding-top: 2rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    padding-bottom: 2rem !important;
    max-width: 100% !important;
}

.stButton > button {
    background: #1d1d1f !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px 28px !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
    width: 100% !important;
    letter-spacing: -0.01em !important;
}
.stButton > button:hover { background: #3a3a3c !important; }

.stSelectbox > div > div,
.stNumberInput > div > div > input,
.stTextInput > div > div > input {
    border-radius: 10px !important;
    border: 1.5px solid #e5e5ea !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
    background: #ffffff !important;
}

.stTabs [data-baseweb="tab-list"] {
    background: #e8e8e8 !important;
    border-radius: 12px !important;
    padding: 4px !important;
    gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 9px !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    padding: 8px 20px !important;
    color: #6e6e73 !important;
}
.stTabs [aria-selected="true"] {
    background: #ffffff !important;
    color: #1d1d1f !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.1) !important;
}

.vt-section {
    font-size: 12px;
    font-weight: 600;
    color: #aeaeb2;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 12px;
    margin-top: 20px;
}
.vt-divider {
    height: 1px;
    background: #e8e8e8;
    margin: 24px 0;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(base, 'models', 'xgboost.pkl'), 'rb') as f:
        model = pickle.load(f)
    with open(os.path.join(base, 'models', 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    with open(os.path.join(base, 'models', 'shap_explainer.pkl'), 'rb') as f:
        explainer = pickle.load(f)
    with open(os.path.join(base, 'data', 'processed', 'feature_columns.pkl'), 'rb') as f:
        feature_cols = pickle.load(f)
    return model, scaler, explainer, feature_cols

@st.cache_data
def load_data():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return pd.read_csv(os.path.join(base, 'data', 'raw', 'vitalis_dataset.csv'))

model, scaler, explainer, feature_cols = load_models()
df = load_data()

EFFORT_RPE = {
    "Easy — I could keep going for hours": 4,
    "Moderate — comfortable but working": 6,
    "Hard — challenging, breathing heavy": 7,
    "Very Hard — pushing my limits": 8,
    "Maximum — all out effort": 10,
}

sport_model_map = {
    "Football":      "sport_Football",
    "Basketball":    "sport_Basketball",
    "Rugby":         "sport_Football",
    "MMA":           "sport_Football",
    "Running":       "sport_Running",
    "Cycling":       "sport_Running",
    "Swimming":      "sport_Running",
    "Weightlifting": "sport_Gym",
    "Sprinting":     "sport_Gym",
    "Gymnastics":    "sport_Gym",
}

sport_profiles = {
    "Football":      {"icon":"⚽","category":"Mixed — Aerobic + Anaerobic","muscle":"Fast-twitch + Slow-twitch","risks":"Hamstring tears · ACL injuries · Ankle sprains"},
    "Basketball":    {"icon":"🏀","category":"Mixed — Aerobic + Anaerobic","muscle":"Fast-twitch dominant","risks":"Ankle sprains · Knee injuries · Achilles tears"},
    "Rugby":         {"icon":"🏉","category":"Mixed — Contact Sport","muscle":"Fast-twitch + Slow-twitch","risks":"Concussion · Muscle contusions · Shoulder injuries"},
    "MMA":           {"icon":"🥊","category":"Mixed — Extreme Demand","muscle":"Fast-twitch dominant","risks":"Joint injuries · Muscle tears · Overtraining syndrome"},
    "Running":       {"icon":"🏃","category":"Aerobic — Endurance","muscle":"Slow-twitch dominant","risks":"Stress fractures · IT band syndrome · Shin splints"},
    "Cycling":       {"icon":"🚴","category":"Aerobic — Endurance","muscle":"Slow-twitch dominant","risks":"Knee overuse · Lower back strain · Saddle injuries"},
    "Swimming":      {"icon":"🏊","category":"Aerobic — Full Body","muscle":"Slow-twitch dominant","risks":"Shoulder impingement · Lower back pain · Knee strain"},
    "Weightlifting": {"icon":"🏋️","category":"Anaerobic — Maximal Strength","muscle":"Fast-twitch Type II dominant","risks":"Muscle tears · Tendon ruptures · Spinal compression"},
    "Sprinting":     {"icon":"💨","category":"Anaerobic — Explosive Power","muscle":"Fast-twitch Type IIx dominant","risks":"Hamstring tears · Hip flexor strains · Achilles injuries"},
    "Gymnastics":    {"icon":"🤸","category":"Anaerobic — Power + Flexibility","muscle":"Fast-twitch + Connective tissue","risks":"Growth plate stress · Wrist injuries · Ankle sprains"},
}

def get_sport_explanation(sport, acwr, fatigue_level, stress_level, sleep_hours, intensity, session_duration):
    explanations = {
        "Football":      f"Football combines explosive anaerobic sprints with sustained aerobic endurance, placing simultaneous demand on fast-twitch and slow-twitch muscle fibres. Rapid direction changes, physical contact and repeated sprint bouts cause lactic acid accumulation and neuromuscular fatigue. Your ACWR of {acwr} is {'particularly dangerous — hamstring and ACL risk spikes with workload surges' if acwr > 1.3 else 'within manageable limits, but monitor fatigue closely'}.",
        "Basketball":    f"Basketball demands explosive jumping, rapid cutting and sustained court coverage. Fast-twitch fibres are heavily recruited for acceleration and vertical force. Landing mechanics under fatigue are a primary injury mechanism. Your fatigue level of {fatigue_level}/10 {'significantly increases landing injury risk' if fatigue_level >= 7 else 'is manageable but monitor during high-intensity play'}.",
        "Rugby":         f"Rugby combines maximal aerobic output with explosive anaerobic contact events. Physical collision adds trauma-based injury risk on top of workload-related risk. Your ACWR of {acwr} combined with {'high' if fatigue_level >= 7 else 'moderate'} fatigue creates elevated vulnerability to both contact and non-contact injuries.",
        "MMA":           f"MMA places the highest combined physiological demand of any sport — strength, power, aerobic endurance and skill training compete for recovery resources. Your stress level of {stress_level}/10 and fatigue of {fatigue_level}/10 are critical — {'overtraining risk is very high at these levels' if stress_level >= 7 or fatigue_level >= 7 else 'currently manageable but recovery must be prioritised'}.",
        "Running":       f"Distance running relies primarily on slow-twitch oxidative fibres. Injury risk is driven by repetitive stress and overuse rather than acute trauma. Sudden volume increases cause cumulative microtrauma in bone and connective tissue. Your ACWR of {acwr} is the single most important indicator — {'reduce mileage immediately to prevent stress fracture risk' if acwr > 1.3 else 'workload progression is within safe limits'}.",
        "Cycling":       f"Cycling generates high repetitive joint loading through pedal cycles. Overuse injuries dominate — particularly patellofemoral pain and IT band issues. Your sleep of {sleep_hours} hours per night {'is below the recovery threshold needed for aerobic adaptation' if sleep_hours < 7 else 'is adequate to support your training load'}.",
        "Swimming":      f"Swimming generates high repetitive shoulder loading through stroke mechanics. Rotator cuff impingement is the most common injury, driven by volume and technique breakdown under fatigue. Your training across {session_duration} minutes per session {'is very high — shoulder fatigue risk is elevated' if intensity >= 8 else 'is appropriate if technique is maintained throughout'}.",
        "Weightlifting": f"Weightlifting depends on fast-twitch Type II fibres for maximal force production. These fibres are highly susceptible to acute tears when loaded without adequate recovery. Your fatigue level of {fatigue_level}/10 {'significantly increases acute tear risk — avoid maximal lifts until recovered' if fatigue_level >= 7 else 'is manageable for moderate loading but avoid true maximal attempts'}.",
        "Sprinting":     f"Sprinting recruits the fastest Type IIx muscle fibres at near-maximal velocity, generating enormous tensile force through the hamstring complex. Your ACWR of {acwr} combined with fatigue of {fatigue_level}/10 creates {'a very high risk environment for hamstring injury — reduce sprint volume immediately' if acwr > 1.3 or fatigue_level >= 7 else 'a manageable risk profile — maintain warm-up protocols and avoid cold-start sprinting'}.",
        "Gymnastics":    f"Gymnastics combines explosive power with extreme range of motion, placing high stress on fast-twitch fibres and connective tissue. Your stress level of {stress_level}/10 is particularly relevant — {'psychological stress significantly impairs neuromuscular coordination, increasing fall and injury risk' if stress_level >= 6 else 'psychological readiness is adequate for training'}.",
    }
    return explanations.get(sport, "")

def risk_info(prob):
    p = prob * 100
    if p >= 65: return "Very High", "#ff3b30", "#fff0ee"
    if p >= 45: return "High",      "#ff6b00", "#fff4ee"
    if p >= 25: return "Medium",    "#f59e0b", "#fffbea"
    return               "Low",     "#22c55e", "#f0fdf4"

def pt():
    return dict(
        template="plotly_white",
        font=dict(family="Inter, sans-serif", size=12, color="#1d1d1f"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=40, b=40, l=40, r=20),
    )

def section_label(text):
    st.markdown(f'<div class="vt-section">{text}</div>', unsafe_allow_html=True)

def divider():
    st.markdown('<div class="vt-divider"></div>', unsafe_allow_html=True)


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:28px 20px 16px;">
        <div style="font-size:20px;font-weight:800;color:#ffffff;letter-spacing:-0.03em;">⚡ Vitalis</div>
        <div style="font-size:11px;color:#6e6e73;margin-top:3px;letter-spacing:0.02em;text-transform:uppercase;">Injury Risk System</div>
    </div>
    <div style="height:1px;background:#3a3a3c;margin:0 20px 8px;"></div>
    """, unsafe_allow_html=True)

    page = st.radio("", ["Home", "Risk Assessment", "My Results", "Analytics", "About"],
                    label_visibility="collapsed")

    st.markdown("""
    <div style="height:1px;background:#3a3a3c;margin:8px 20px 16px;"></div>
    <div style="padding:0 20px;">
        <div style="font-size:11px;font-weight:700;color:#ffffff;margin-bottom:6px;">XGBoost</div>
        <div style="font-size:11px;color:#6e6e73;line-height:2.0;">
            Accuracy 93.0%<br>
            ROC-AUC 96.6%<br>
            1,000 records
        </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "Home":
    st.markdown("""
    <div style="background:#ffffff;padding:48px 48px 40px;margin:-32px -32px 0 -32px;">
        <div style="max-width:640px;">
            <div style="font-size:12px;font-weight:700;color:#0071e3;text-transform:uppercase;letter-spacing:0.12em;margin-bottom:16px;">
                BDA6 Capstone · Polytechnics Mauritius
            </div>
            <div style="font-size:48px;font-weight:800;color:#1d1d1f;letter-spacing:-0.04em;line-height:1.05;margin-bottom:16px;">
                Predict your injury risk<br>before it happens.
            </div>
            <div style="font-size:17px;color:#6e6e73;line-height:1.7;">
                Answer a few simple questions about how you train, sleep and recover.
                Vitalis uses machine learning to tell you exactly how likely you are
                to get injured — and what to do about it.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    stats = load_dataset_stats()
    c1, c2, c3, c4 = st.columns(4)
    for col, (val, label, color) in zip([c1,c2,c3,c4], [
        ("93%", "Model accuracy", "#1d1d1f"),
        ("96.6%", "ROC-AUC", "#0071e3"),
        (f"{stats.get('total_records', 1000):,}", "Athletes analysed", "#1d1d1f"),
        (f"{stats.get('num_features', 19)}", "Risk factors", "#1d1d1f"),
    ]):
        with col:
            st.markdown(f"""
            <div style="background:#ffffff;border-radius:14px;padding:20px 22px;box-shadow:0 1px 8px rgba(0,0,0,0.07);">
                <div style="font-size:28px;font-weight:800;color:{color};letter-spacing:-0.03em;line-height:1;">{val}</div>
                <div style="font-size:11px;font-weight:600;color:#aeaeb2;text-transform:uppercase;letter-spacing:0.07em;margin-top:6px;">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    for col, (icon, title, desc) in zip([c1,c2,c3], [
        ("⚡", "Real-time prediction", "Fill in your details and get your personal injury risk score instantly — no sports science knowledge needed."),
        ("🔍", "Explains why", "SHAP analysis shows you exactly which factors are driving your risk, not just a number."),
        ("📋", "Tells you what to do", "Every result comes with specific, actionable recommendations based on your data."),
    ]):
        with col:
            st.markdown(f"""
            <div style="background:#ffffff;border-radius:16px;padding:24px;box-shadow:0 1px 8px rgba(0,0,0,0.07);">
                <div style="font-size:24px;margin-bottom:12px;">{icon}</div>
                <div style="font-size:15px;font-weight:700;color:#1d1d1f;margin-bottom:8px;">{title}</div>
                <div style="font-size:13px;color:#6e6e73;line-height:1.7;">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    section_label("Key findings from our research")

    c1, c2, c3 = st.columns(3)
    for col, (color, stat, desc) in zip([c1,c2,c3], [
        ("#ff3b30", "100%", "injury rate when ACWR exceeds 1.5 — the single most dangerous training pattern"),
        ("#ff9500", "41.1%", "injury rate for athletes sleeping under 6 hours vs 10.3% for those sleeping 8+"),
        ("#ff3b30", "47.8%", "injury rate for athletes with previous injuries vs only 6.7% for those without"),
    ]):
        with col:
            st.markdown(f"""
            <div style="background:#ffffff;border-radius:16px;padding:24px;box-shadow:0 1px 8px rgba(0,0,0,0.07);border-top:3px solid {color};">
                <div style="font-size:36px;font-weight:800;color:{color};letter-spacing:-0.03em;line-height:1;margin-bottom:8px;">{stat}</div>
                <div style="font-size:13px;color:#6e6e73;line-height:1.6;">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    section_label("How it works")

    c1, c2, c3, c4 = st.columns(4)
    for col, (num, t, d) in zip([c1,c2,c3,c4], [
        ("01", "Tell us about yourself", "Age, sport, height, weight — simple questions anyone can answer"),
        ("02", "Describe your training", "How often, how long, how hard — in plain language"),
        ("03", "Get your score", "XGBoost gives you a personalised injury risk percentage"),
        ("04", "Take action", "See exactly what is raising your risk and what to change"),
    ]):
        with col:
            st.markdown(f"""
            <div style="background:#ffffff;border-radius:14px;padding:20px;box-shadow:0 1px 8px rgba(0,0,0,0.07);">
                <div style="font-size:24px;font-weight:800;color:#0071e3;margin-bottom:8px;">{num}</div>
                <div style="font-size:14px;font-weight:700;color:#1d1d1f;margin-bottom:6px;">{t}</div>
                <div style="font-size:12px;color:#aeaeb2;line-height:1.6;">{d}</div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# RISK ASSESSMENT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Risk Assessment":
    st.markdown("""
    <div style="margin-bottom:8px;">
        <div style="font-size:32px;font-weight:800;color:#1d1d1f;letter-spacing:-0.03em;">Risk Assessment</div>
        <div style="font-size:15px;color:#6e6e73;margin-top:4px;">Answer these simple questions to get your personalised injury risk score.</div>
    </div>
    """, unsafe_allow_html=True)
    divider()

    col_form, col_gap, col_result = st.columns([5, 0.3, 5])

    with col_form:
        section_label("About you")
        c1, c2 = st.columns(2)
        with c1:
            name = st.text_input("Your name (optional)", placeholder="e.g. Johan")
            age = st.number_input("Age", 15, 60, 25)
            sex_input = st.selectbox("Sex", ["Male", "Female"])
        with c2:
            sport = st.selectbox("Sport", list(sport_model_map.keys()))
            experience = st.number_input("Years playing this sport", 0, 30, 3)
            previous_injury = st.selectbox("Have you had a sports injury before?", ["No", "Yes"])

        num_previous_injuries = 0
        if previous_injury == "Yes":
            num_previous_injuries = st.number_input("How many previous injuries?", 1, 20, 1)

        divider()
        section_label("Body measurements")
        c1, c2, c3 = st.columns(3)
        with c1:
            height_cm = st.number_input("Height (cm)", 140, 220, 175)
        with c2:
            weight_kg = st.number_input("Weight (kg)", 40, 180, 70)
        with c3:
            bmi = round(weight_kg / ((height_cm / 100) ** 2), 1)
            if bmi < 18.5:
                bmi_label, bmi_color = "Underweight", "#f59e0b"
            elif bmi < 25:
                bmi_label, bmi_color = "Healthy", "#22c55e"
            elif bmi < 30:
                bmi_label, bmi_color = "Overweight", "#f59e0b"
            else:
                bmi_label, bmi_color = "Obese", "#ff3b30"
            st.markdown(f"""
            <div style="background:#f5f5f7;border-radius:12px;padding:12px 16px;margin-top:28px;">
                <div style="font-size:11px;color:#aeaeb2;text-transform:uppercase;letter-spacing:0.07em;margin-bottom:4px;">Your BMI</div>
                <div style="font-size:24px;font-weight:800;color:{bmi_color};letter-spacing:-0.02em;">{bmi}</div>
                <div style="font-size:12px;color:{bmi_color};font-weight:600;">{bmi_label}</div>
            </div>""", unsafe_allow_html=True)

        divider()
        section_label("This week's training")
        c1, c2 = st.columns(2)
        with c1:
            sessions_this_week = st.number_input("How many times did you train this week?", 0, 14, 4)
            session_duration = st.number_input("Average session length (minutes)", 15, 240, 75)
        with c2:
            effort_this_week = st.selectbox("How hard did you push yourself this week?", list(EFFORT_RPE.keys()))
            training_days = sessions_this_week
            recovery_days = max(0, 7 - sessions_this_week)

        rpe_this_week = EFFORT_RPE[effort_this_week]
        acute_workload = sessions_this_week * session_duration * rpe_this_week

        divider()
        section_label("Your usual training (past month)")
        c1, c2 = st.columns(2)
        with c1:
            sessions_usual = st.number_input("How many times per week do you usually train?", 0, 14, 4)
            duration_usual = st.number_input("Usual session length (minutes)", 15, 240, 75)
        with c2:
            effort_usual = st.selectbox("How hard do you usually train?", list(EFFORT_RPE.keys()), index=1)

        rpe_usual = EFFORT_RPE[effort_usual]
        chronic_workload = sessions_usual * duration_usual * rpe_usual
        acwr = round(acute_workload / chronic_workload, 2) if chronic_workload > 0 else 1.0

        if acwr > 1.5:
            acwr_color, acwr_label, acwr_zone = "#ff3b30", "⚠️ Danger zone", "Danger"
            acwr_desc = "You trained much harder than usual this week. High injury risk."
        elif acwr > 1.3:
            acwr_color, acwr_label, acwr_zone = "#f59e0b", "⚠️ Caution zone", "Caution"
            acwr_desc = "Your training load is higher than normal. Be careful."
        else:
            acwr_color, acwr_label, acwr_zone = "#22c55e", "✅ Safe zone", "Safe"
            acwr_desc = "Your training load is consistent with your usual habits."

        st.markdown(f"""
        <div style="background:#ffffff;border-radius:12px;padding:16px 20px;margin:12px 0;box-shadow:0 1px 4px rgba(0,0,0,0.06);">
            <span style="font-size:13px;color:#6e6e73;">Training load ratio </span>
            <span style="font-size:22px;font-weight:800;color:{acwr_color};letter-spacing:-0.02em;"> {acwr}</span>
            <span style="font-size:13px;font-weight:600;color:{acwr_color};margin-left:8px;">{acwr_label}</span>
            <div style="font-size:12px;color:#6e6e73;margin-top:4px;">{acwr_desc}</div>
        </div>""", unsafe_allow_html=True)

        divider()
        section_label("How you feel")
        sleep_hours = st.slider("How many hours do you sleep per night on average?", 4.0, 10.0, 7.0, 0.5)
        fatigue_level = st.slider("How tired do you feel right now? (1 = fresh, 10 = exhausted)", 1, 10, 4)
        stress_level = st.slider("How stressed are you in general? (1 = relaxed, 10 = very stressed)", 1, 10, 4)

        divider()
        section_label("Training habits")
        c1, c2 = st.columns(2)
        with c1: warm_up = st.selectbox("Do you warm up before training?", ["Yes", "No"])
        with c2: stretching = st.selectbox("Do you stretch after training?", ["Yes", "No"])

        st.markdown("<br>", unsafe_allow_html=True)
        predict = st.button("⚡  Get my injury risk score")

    with col_result:
        if predict:
            sex_val = 1 if sex_input == "Male" else 0
            intensity = rpe_this_week

            input_dict = {
                'age': age, 'sex': sex_val, 'bmi': bmi,
                'experience_years': experience,
                'session_duration_mins': session_duration,
                'training_intensity_rpe': intensity,
                'training_days_per_week': int(training_days),
                'chronic_workload': float(chronic_workload),
                'acute_workload': float(acute_workload),
                'acwr': float(acwr),
                'sleep_hours': float(sleep_hours),
                'fatigue_level': int(fatigue_level),
                'stress_level': int(stress_level),
                'warm_up': 1 if warm_up == "Yes" else 0,
                'stretching': 1 if stretching == "Yes" else 0,
                'recovery_days': int(recovery_days),
                'previous_injury': 1 if previous_injury == "Yes" else 0,
                'num_previous_injuries': int(num_previous_injuries),
            }
            for col in feature_cols:
                if col.startswith("sport_"):
                    input_dict[col] = 1 if col == sport_model_map.get(sport) else 0

            input_df = pd.DataFrame([input_dict])[feature_cols]
            input_scaled = scaler.transform(input_df)
            prob = model.predict_proba(input_scaled)[0][1]
            level, color, bg = risk_info(prob)
            pct = int(prob * 100)

            shap_values = explainer(input_scaled)
            sv = shap_values.values[0] if hasattr(shap_values, 'values') else shap_values[0]
            if len(np.array(sv).shape) > 1:
                sv = np.array(sv)[:, 1]
            shap_df = pd.DataFrame({'feature': feature_cols, 'shap': sv})
            shap_df['abs'] = shap_df['shap'].abs()
            shap_df = shap_df.sort_values('abs', ascending=False).head(7)
            shap_df['label'] = shap_df['feature'].str.replace('_', ' ').str.title()

            recs = []
            if acwr > 1.5:
                recs.append(("🔴", "Reduce your training load now", "You trained much harder than usual this week. Take 2-3 lighter days before your next hard session."))
            elif acwr > 1.3:
                recs.append(("🟡", "Ease off for a few days", "Your training load is above your usual baseline. Reduce intensity for 3–5 days."))
            if sleep_hours < 7:
                recs.append(("🔴", "Get more sleep", "You are sleeping below the recommended threshold. Aim for 7–8 hours."))
            if fatigue_level >= 7:
                recs.append(("🟡", "Rest before training again", "You are reporting high fatigue. Take at least 1–2 full rest days."))
            if stress_level >= 7:
                recs.append(("🟡", "Manage your stress levels", "High stress significantly increases injury risk. Consider lighter training or extra recovery time."))
            if previous_injury == "Yes":
                recs.append(("🔴", "Protect your injury history", "Your previous injury significantly raises your risk. Prioritise injury prevention exercises."))
            if warm_up == "No":
                recs.append(("🟡", "Always warm up", "Starting training cold dramatically increases acute injury risk."))
            if not recs:
                recs.append(("✅", "You are in good shape", "Your training profile is well balanced. Keep maintaining your current habits."))

            top_feature = shap_df.iloc[0]['feature'].replace('_', ' ').title() if len(shap_df) > 0 else ""
            top_shap_val = float(shap_df.iloc[0]['shap']) if len(shap_df) > 0 else 0.0
            rec_texts = [f"{r[1]}: {r[2]}" for r in recs[:3]]
            while len(rec_texts) < 3:
                rec_texts.append("")

            # Save to Supabase
            save_prediction({
                "name": str(name) if name else "Anonymous",
                "age": int(age),
                "sex": int(sex_val),
                "sport": str(sport),
                "sport_category": str(sport_profiles.get(sport, {}).get("category", "")),
                "bmi": float(round(bmi, 1)),
                "experience_years": int(experience),
                "training_days_per_week": int(training_days),
                "session_duration_mins": int(session_duration),
                "training_intensity_rpe": int(intensity),
                "recovery_days": int(recovery_days),
                "acute_workload": int(acute_workload),
                "chronic_workload": int(chronic_workload),
                "acwr": float(round(acwr, 2)),
                "acwr_zone": str(acwr_zone),
                "sleep_hours": float(sleep_hours),
                "fatigue_level": int(fatigue_level),
                "stress_level": int(stress_level),
                "warm_up": bool(warm_up == "Yes"),
                "stretching": bool(stretching == "Yes"),
                "previous_injury": bool(previous_injury == "Yes"),
                "num_previous_injuries": int(num_previous_injuries),
                "risk_score": float(round(prob, 4)),
                "risk_level": str(level),
                "risk_percentage": int(pct),
                "shap_top_feature": str(top_feature),
                "shap_top_value": float(round(top_shap_val, 4)),
                "recommendation_1": str(rec_texts[0]),
                "recommendation_2": str(rec_texts[1]),
                "recommendation_3": str(rec_texts[2]),
            })

            # Score card
            st.markdown(f"""
            <div style="background:{bg};border-radius:24px;padding:40px 32px;text-align:center;margin-bottom:16px;border:1.5px solid {color}22;">
                <div style="font-size:12px;font-weight:600;color:{color};text-transform:uppercase;letter-spacing:0.12em;margin-bottom:8px;">Your Injury Risk Score</div>
                <div style="font-size:88px;font-weight:800;color:{color};letter-spacing:-0.05em;line-height:1;">{pct}<span style="font-size:36px;font-weight:400;color:{color}88;">%</span></div>
                <div style="margin-top:16px;">
                    <span style="background:{color};color:#fff;font-size:12px;font-weight:700;padding:6px 20px;border-radius:100px;letter-spacing:0.06em;text-transform:uppercase;">{level} Risk</span>
                </div>
                <div style="font-size:11px;color:{color}88;margin-top:10px;">Saved to database ✓</div>
            </div>""", unsafe_allow_html=True)

            # SHAP chart
            bar_colors = [color if v > 0 else "#22c55e" for v in shap_df['shap']]
            fig = go.Figure(go.Bar(
                x=shap_df['shap'], y=shap_df['label'],
                orientation='h', marker_color=bar_colors, marker_line_width=0,
                hovertemplate="%{y}: %{x:.3f}<extra></extra>"
            ))
            fig.update_layout(
                **pt(), height=280,
                title=dict(text="What is driving your risk", font_size=13, x=0),
                xaxis=dict(title="Impact on risk score", zeroline=True, zerolinecolor="#e5e5ea"),
                yaxis=dict(autorange="reversed"),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            # Recommendations
            section_label("What to do")
            for icon, title, desc in recs:
                st.markdown(f"""
                <div style="background:#ffffff;border-radius:14px;padding:14px 18px;margin-bottom:10px;box-shadow:0 1px 4px rgba(0,0,0,0.05);">
                    <div style="font-size:14px;font-weight:700;color:#1d1d1f;margin-bottom:4px;">{icon} {title}</div>
                    <div style="font-size:13px;color:#6e6e73;line-height:1.6;">{desc}</div>
                </div>""", unsafe_allow_html=True)

            # Sport profile
            profile = sport_profiles.get(sport)
            explanation = get_sport_explanation(sport, acwr, fatigue_level, stress_level, sleep_hours, intensity, session_duration)
            if profile:
                st.markdown(f"""
                <div style="background:#1d1d1f;border-radius:18px;padding:22px;margin-top:8px;">
                    <div style="font-size:12px;font-weight:700;color:#0071e3;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:12px;">
                        {profile['icon']} {sport} — Injury Profile
                    </div>
                    <div style="display:flex;gap:8px;margin-bottom:14px;flex-wrap:wrap;">
                        <span style="background:rgba(255,255,255,0.08);color:#ffffff;font-size:11px;font-weight:600;padding:3px 12px;border-radius:100px;">{profile['category']}</span>
                        <span style="background:rgba(255,255,255,0.08);color:#ffffff;font-size:11px;font-weight:600;padding:3px 12px;border-radius:100px;">{profile['muscle']}</span>
                    </div>
                    <div style="font-size:11px;font-weight:600;color:#ff6b6b;margin-bottom:10px;text-transform:uppercase;letter-spacing:0.06em;">
                        {profile['risks']}
                    </div>
                    <div style="font-size:13px;color:#aeaeb2;line-height:1.8;">{explanation}</div>
                </div>""", unsafe_allow_html=True)

        else:
            st.markdown("""
            <div style="background:#ffffff;border-radius:20px;padding:60px 32px;text-align:center;margin-top:20px;box-shadow:0 1px 8px rgba(0,0,0,0.06);">
                <div style="font-size:44px;margin-bottom:14px;">⚡</div>
                <div style="font-size:18px;font-weight:700;color:#1d1d1f;margin-bottom:8px;">Ready when you are</div>
                <div style="font-size:14px;color:#aeaeb2;line-height:1.7;">Fill in your details on the left<br>and click <strong style="color:#1d1d1f;">Get my injury risk score</strong></div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MY RESULTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "My Results":
    st.markdown("""
    <div style="margin-bottom:8px;">
        <div style="font-size:32px;font-weight:800;color:#1d1d1f;letter-spacing:-0.03em;">My Results</div>
        <div style="font-size:15px;color:#6e6e73;margin-top:4px;">All predictions stored in the Supabase database.</div>
    </div>
    """, unsafe_allow_html=True)
    divider()

    preds = load_all_predictions()

    if preds.empty:
        st.markdown("""
        <div style="background:#ffffff;border-radius:20px;padding:60px 32px;text-align:center;box-shadow:0 1px 8px rgba(0,0,0,0.06);">
            <div style="font-size:44px;margin-bottom:14px;">📊</div>
            <div style="font-size:18px;font-weight:700;color:#1d1d1f;margin-bottom:8px;">No predictions yet</div>
            <div style="font-size:15px;color:#aeaeb2;">Go to Risk Assessment to generate your first prediction.</div>
        </div>""", unsafe_allow_html=True)
    else:
        total = len(preds)
        avg_risk = int(preds['risk_percentage'].mean())
        high_risk = len(preds[preds['risk_percentage'] >= 45])
        low_risk = len(preds[preds['risk_percentage'] < 25])

        c1, c2, c3, c4 = st.columns(4)
        for col, (val, label, color) in zip([c1,c2,c3,c4], [
            (total, "Total Assessments", "#1d1d1f"),
            (f"{avg_risk}%", "Average Risk", "#f59e0b" if avg_risk >= 25 else "#22c55e"),
            (high_risk, "High Risk Cases", "#ff3b30"),
            (low_risk, "Low Risk Cases", "#22c55e"),
        ]):
            with col:
                st.markdown(f"""
                <div style="background:#ffffff;border-radius:14px;padding:20px 22px;box-shadow:0 1px 8px rgba(0,0,0,0.07);margin-bottom:16px;">
                    <div style="font-size:28px;font-weight:800;color:{color};letter-spacing:-0.03em;">{val}</div>
                    <div style="font-size:11px;font-weight:600;color:#aeaeb2;text-transform:uppercase;letter-spacing:0.07em;margin-top:6px;">{label}</div>
                </div>""", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure(go.Histogram(
                x=preds['risk_percentage'], nbinsx=20,
                marker_color="#0071e3", marker_line_width=0,
                hovertemplate="Risk %{x}%: %{y} people<extra></extra>"
            ))
            fig.update_layout(**pt(), title="Risk score distribution", height=260,
                             xaxis_title="Risk Score (%)", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        with c2:
            if 'sport' in preds.columns:
                sport_counts = preds['sport'].value_counts().head(6)
                fig = go.Figure(go.Bar(
                    x=sport_counts.values, y=sport_counts.index,
                    orientation='h', marker_color="#0071e3", marker_line_width=0,
                ))
                fig.update_layout(**pt(), title="Assessments by sport", height=260)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        section_label("All predictions")
        display_cols = ['created_at','name','sport','age','sex','acwr','sleep_hours','risk_percentage','risk_level','recommendation_1']
        available = [c for c in display_cols if c in preds.columns]
        st.dataframe(
            preds[available].head(50).rename(columns={
                'created_at':'Date','name':'Name','sport':'Sport','age':'Age','sex':'Sex',
                'acwr':'ACWR','sleep_hours':'Sleep',
                'risk_percentage':'Risk %','risk_level':'Level',
                'recommendation_1':'Top Recommendation'
            }),
            use_container_width=True, hide_index=True
        )


# ══════════════════════════════════════════════════════════════════════════════
# ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Analytics":
    st.markdown("""
    <div style="margin-bottom:8px;">
        <div style="font-size:32px;font-weight:800;color:#1d1d1f;letter-spacing:-0.03em;">Analytics</div>
        <div style="font-size:15px;color:#6e6e73;margin-top:4px;">Dataset and model performance — all data pulled from Supabase.</div>
    </div>
    """, unsafe_allow_html=True)
    divider()

    tab1, tab2, tab3 = st.tabs(["  Dataset  ", "  Model Performance  ", "  SHAP Analysis  "])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure(go.Pie(
                values=df['injury'].value_counts().values,
                labels=["Not Injured","Injured"], hole=0.65,
                marker_colors=["#22c55e","#ff3b30"], textinfo="percent",
                hovertemplate="%{label}: %{value}<extra></extra>"
            ))
            fig.update_layout(**pt(), title="Injury distribution", height=300)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        with c2:
            sport_inj = df.groupby('sport')['injury'].mean().sort_values() * 100
            fig = go.Figure(go.Bar(
                x=sport_inj.values, y=sport_inj.index,
                orientation='h', marker_color="#0071e3", marker_line_width=0,
                hovertemplate="%{y}: %{x:.1f}%<extra></extra>"
            ))
            fig.update_layout(**pt(), title="Injury rate by sport (%)", height=300)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            for inj, color, name in [(0,"#22c55e","Not Injured"),(1,"#ff3b30","Injured")]:
                fig.add_trace(go.Box(y=df[df['injury']==inj]['acwr'], name=name,
                                     marker_color=color, boxpoints="outliers", line_width=1.5))
            fig.update_layout(**pt(), title="ACWR by injury status", height=300)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        with c2:
            fig = go.Figure()
            for inj, color, name in [(0,"#22c55e","Not Injured"),(1,"#ff3b30","Injured")]:
                fig.add_trace(go.Box(y=df[df['injury']==inj]['sleep_hours'], name=name,
                                     marker_color=color, boxpoints="outliers", line_width=1.5))
            fig.update_layout(**pt(), title="Sleep hours by injury status", height=300)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with tab2:
        mdf = load_model_results()
        if mdf.empty:
            mdf = pd.DataFrame({
                'model_name': ['Logistic Regression','Random Forest','XGBoost'],
                'accuracy': [82.0, 91.5, 93.0],
                'precision_score': [66.67, 88.37, 90.91],
                'recall': [56.0, 76.0, 80.0],
                'f1_score': [60.87, 81.72, 85.11],
                'roc_auc': [87.76, 95.93, 96.56],
            })
        else:
            for col in ['accuracy','precision_score','recall','f1_score','roc_auc']:
                if mdf[col].max() <= 1.0:
                    mdf[col] = mdf[col] * 100

        fig = go.Figure()
        for i, row in mdf.iterrows():
            fig.add_trace(go.Bar(
                name=row['model_name'],
                x=['Accuracy','Precision','Recall','F1','ROC-AUC'],
                y=[row['accuracy'],row['precision_score'],row['recall'],row['f1_score'],row['roc_auc']],
                marker_color=["#d1d1d6","#8e8e93","#0071e3"][i % 3],
                marker_line_width=0,
                hovertemplate=row['model_name']+": %{y:.1f}%<extra></extra>"
            ))
        fig.update_layout(**pt(), barmode='group', height=360,
                         title="Model comparison — all metrics",
                         yaxis=dict(range=[50,100], title="Score (%)"),
                         legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        c1, c2, c3 = st.columns(3)
        for col, (_, row) in zip([c1,c2,c3], mdf.iterrows()):
            best = row['model_name'] == 'XGBoost'
            with col:
                st.markdown(f"""
                <div style="background:#ffffff;border-radius:16px;padding:22px;box-shadow:0 1px 8px rgba(0,0,0,0.07);
                    {'border:2px solid #0071e3;' if best else ''}">
                    <div style="font-size:11px;font-weight:700;color:{'#0071e3' if best else '#aeaeb2'};text-transform:uppercase;letter-spacing:0.06em;margin-bottom:10px;">
                        {row['model_name']} {'✦ Best' if best else ''}
                    </div>
                    <div style="font-size:32px;font-weight:800;color:#1d1d1f;letter-spacing:-0.03em;">{row['accuracy']:.1f}%</div>
                    <div style="font-size:11px;color:#aeaeb2;margin-bottom:14px;">Accuracy</div>
                    <div style="font-size:12px;color:#6e6e73;line-height:2.1;">
                        F1 Score: {row['f1_score']:.2f}%<br>
                        ROC-AUC: {row['roc_auc']:.2f}%<br>
                        Recall: {row['recall']:.1f}%
                    </div>
                </div>""", unsafe_allow_html=True)

    with tab3:
        shap_df_live = load_shap_importance()
        if not shap_df_live.empty:
            shap_df_live = shap_df_live.sort_values('mean_shap_value')
            shap_df_live['label'] = shap_df_live['feature'].str.replace('_', ' ').str.title()
            x_vals = shap_df_live['mean_shap_value']
            y_vals = shap_df_live['label']
        else:
            x_vals = [0.32, 0.48, 0.65, 0.77, 1.14, 1.27, 2.20, 2.54]
            y_vals = ['Recovery Days','Training Intensity','Stress Level','Fatigue Level',
                      'Num Previous Injuries','Sleep Hours','ACWR','Previous Injury']

        fig = go.Figure(go.Bar(
            x=x_vals, y=y_vals,
            orientation='h', marker_color="#0071e3", marker_line_width=0,
            hovertemplate="%{y}: %{x:.2f}<extra></extra>"
        ))
        fig.update_layout(**pt(), height=360,
                         title="Global feature importance — mean |SHAP value|",
                         xaxis_title="Mean |SHAP value|")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        st.markdown("""
        <div style="background:#ffffff;border-radius:14px;padding:18px 22px;box-shadow:0 1px 6px rgba(0,0,0,0.06);">
            <div style="font-size:13px;font-weight:700;color:#1d1d1f;margin-bottom:8px;">How to read this</div>
            <div style="font-size:13px;color:#6e6e73;line-height:1.8;">
                A higher SHAP value means that feature has a greater influence on the model prediction.
                Previous injury history and ACWR are the two most powerful predictors in Vitalis —
                consistent with peer-reviewed sports science literature.
            </div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "About":
    st.markdown("""
    <div style="margin-bottom:8px;">
        <div style="font-size:32px;font-weight:800;color:#1d1d1f;letter-spacing:-0.03em;">About Vitalis</div>
        <div style="font-size:15px;color:#6e6e73;margin-top:4px;">The science and technology behind the system.</div>
    </div>
    """, unsafe_allow_html=True)
    divider()

    c1, c2 = st.columns([3,2])
    with c1:
        st.markdown("""
        <div style="background:#ffffff;border-radius:16px;padding:26px;box-shadow:0 1px 8px rgba(0,0,0,0.07);margin-bottom:14px;">
            <div style="font-size:15px;font-weight:700;color:#1d1d1f;margin-bottom:12px;">What is Vitalis?</div>
            <div style="font-size:13px;color:#6e6e73;line-height:1.9;">
                Vitalis is a machine learning-based predictive injury risk system developed as a BDA6 Capstone
                Project at Polytechnics Mauritius. It analyses an athlete's training load, lifestyle and recovery
                data to predict the probability of sports injury before it happens.<br><br>
                The system is powered by XGBoost and uses SHAP to provide transparent, explainable,
                personalised predictions for athletes, coaches and fitness enthusiasts.
            </div>
        </div>
        <div style="background:#ffffff;border-radius:16px;padding:26px;box-shadow:0 1px 8px rgba(0,0,0,0.07);">
            <div style="font-size:15px;font-weight:700;color:#1d1d1f;margin-bottom:12px;">Scientific basis</div>
            <div style="font-size:13px;color:#6e6e73;line-height:2.2;">
                • Gabbett, T.J. (2016) — ACWR and the training-injury prevention paradox<br>
                • Milewski et al. (2014) — Sleep deprivation and sports injuries<br>
                • Ekstrand et al. (2011) — Hamstring injuries in professional football<br>
                • Lundberg & Lee (2017) — SHAP unified model interpretation<br>
                • Ma, Liu & Pei (2025) — ML for injury risk prediction, Scientific Reports
            </div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div style="background:#1d1d1f;border-radius:16px;padding:26px;margin-bottom:14px;">
            <div style="font-size:15px;font-weight:700;color:#ffffff;margin-bottom:14px;">Technical stack</div>
            <div style="font-size:13px;color:#aeaeb2;line-height:2.4;">
                🐍 Python 3.11<br>
                ⚡ XGBoost<br>
                🔍 SHAP<br>
                📊 scikit-learn<br>
                🌐 Streamlit<br>
                🗄️ Supabase (PostgreSQL)<br>
                📈 Plotly
            </div>
        </div>
        <div style="background:#ffffff;border-radius:16px;padding:26px;box-shadow:0 1px 8px rgba(0,0,0,0.07);">
            <div style="font-size:15px;font-weight:700;color:#1d1d1f;margin-bottom:12px;">Developer</div>
            <div style="font-size:13px;color:#6e6e73;line-height:2.2;">
                <strong style="color:#1d1d1f;">Ny Aina Johan Randriamanalina</strong><br>
                BDA6 — Big Data Analytics<br>
                Polytechnics Mauritius<br>
                ID: 2024_15184_6474<br>
                Supervisor: Mr. A. O. Peerally
            </div>
        </div>""", unsafe_allow_html=True)