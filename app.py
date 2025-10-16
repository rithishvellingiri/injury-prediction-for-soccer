import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load saved model and preprocessors
# -----------------------------
model = joblib.load("injury_model.pkl")        # trained XGBoost model
rfe = joblib.load("feature_selector.pkl")      # saved feature selector
scaler = joblib.load("scaler.pkl")             # saved StandardScaler

# -----------------------------
# Define numeric feature names
# -----------------------------
continuous_features = [
    'height_cm', 'weight_kg', 'work_rate_numeric', 'pace', 'physic',
    'age', 'cumulative_minutes_played', 'minutes_per_game_prev_seasons',
    'avg_days_injured_prev_seasons', 'cumulative_days_injured'
]

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(page_title="⚽ Soccer Injury Predictor", page_icon="⚽", layout="centered")

# -----------------------------
# Custom CSS styling
# -----------------------------
st.markdown("""
    <style>
    /* Main page background */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f1f5f9;
        font-family: 'Poppins', sans-serif;
    }

    /* Header */
    h1 {
        text-align: center;
        color: #38bdf8;
        font-weight: 800;
        font-size: 2.2rem;
        margin-bottom: 0.5em;
    }

    /* Description text */
    .description {
        text-align: center;
        color: #94a3b8;
        font-size: 1rem;
        margin-bottom: 2em;
    }

    /* Input container */
    .card {
        background-color: #1e293b;
        padding: 25px;
        border-radius: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        margin-bottom: 25px;
    }

    /* Buttons */
    div.stButton > button {
        background: linear-gradient(90deg, #2563eb, #1d4ed8);
        color: white;
        border: none;
        padding: 0.7em 2em;
        border-radius: 12px;
        font-weight: 600;
        transition: 0.3s;
        width: 100%;
    }

    div.stButton > button:hover {
        background: linear-gradient(90deg, #1e3a8a, #2563eb);
        transform: scale(1.03);
    }

    /* Success/Error boxes */
    .result-box {
        border-radius: 15px;
        padding: 15px 20px;
        font-size: 1rem;
        font-weight: 600;
    }

    .success-box {
        background-color: #064e3b;
        color: #6ee7b7;
        border-left: 5px solid #10b981;
    }

    .error-box {
        background-color: #7f1d1d;
        color: #fecaca;
        border-left: 5px solid #dc2626;
    }

    /* Footer / accuracy text */
    .footer {
        text-align: center;
        color: #94a3b8;
        margin-top: 20px;
        font-size: 0.9rem;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# Title and description
# -----------------------------
st.markdown("<h1>⚽ Soccer Player Injury Risk Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p class='description'>Predict the likelihood of a player sustaining a major injury using advanced machine learning insights.</p>", unsafe_allow_html=True)

# -----------------------------
# Input section (cards)
# -----------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    height_cm = st.number_input("Height (cm)", min_value=150, max_value=210, value=180)
    weight_kg = st.number_input("Weight (kg)", min_value=50, max_value=120, value=75)
    work_rate_numeric = st.selectbox("Work Rate (Low=1, Medium=1.5, High=2, High/Medium=3.5)", [1, 1.5, 2, 2.5, 3, 3.5], index=2)
    pace = st.number_input("Pace", min_value=20, max_value=100, value=70)
    physic = st.number_input("Physic", min_value=20, max_value=100, value=68)

with col2:
    position_map = {"Goalkeeper": 0, "Defender": 1, "Forward": 2, "Midfielder": 3}
    position_label = st.selectbox("Position", options=list(position_map.keys()))
    position_numeric = position_map[position_label]

    age = st.number_input("Age", min_value=16, max_value=45, value=25)
    cumulative_minutes_played = st.number_input("Cumulative Minutes Played", min_value=0, max_value=50000, value=5000)
    minutes_per_game_prev_seasons = st.number_input("Minutes per Game (Prev Seasons)", min_value=0, max_value=100, value=80)
    avg_days_injured_prev_seasons = st.number_input("Avg Days Injured (Prev Seasons)", min_value=0, max_value=200, value=10)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='card'>", unsafe_allow_html=True)
sig_injury_label = st.selectbox("Significant Injury in Previous Season?", options=["No", "Yes"])
significant_injury_prev_season = 1 if sig_injury_label == "Yes" else 0
cumulative_days_injured = st.number_input("Cumulative Days Injured", min_value=0, max_value=1000, value=50)
st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Prepare data
# -----------------------------
new_data = pd.DataFrame([{
    "height_cm": height_cm,
    "weight_kg": weight_kg,
    "work_rate_numeric": work_rate_numeric,
    "pace": pace,
    "physic": physic,
    "position_numeric": position_numeric,
    "age": age,
    "cumulative_minutes_played": cumulative_minutes_played,
    "minutes_per_game_prev_seasons": minutes_per_game_prev_seasons,
    "avg_days_injured_prev_seasons": avg_days_injured_prev_seasons,
    "significant_injury_prev_season": significant_injury_prev_season,
    "cumulative_days_injured": cumulative_days_injured
}])

# -----------------------------
# Prediction
# -----------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
if st.button("🔍 Predict Injury Risk"):
    try:
        new_data[continuous_features] = scaler.transform(new_data[continuous_features])
        new_data_rfe = rfe.transform(new_data)
        prediction = model.predict(new_data_rfe)[0]
        probability = model.predict_proba(new_data_rfe)[0][1]

        if prediction == 1:
            st.markdown(f"<div class='result-box error-box'>⚠️ Player is at <b>HIGH RISK</b> of major injury.<br>Probability: {probability:.2%}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='result-box success-box'>✅ Player is at <b>LOW RISK</b> of major injury.<br>Probability: {probability:.2%}</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error during prediction: {e}")
st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Footer
# -----------------------------
model_accuracy = 0.73
st.markdown(f"<div class='footer'>📊 <b>Model Accuracy:</b> {model_accuracy*100:.2f}%<br>Note: Accuracy from training phase.</div>", unsafe_allow_html=True)
