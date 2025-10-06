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
# Define numeric feature names (for scaling)
# -----------------------------
continuous_features = [
    'height_cm', 'weight_kg', 'work_rate_numeric', 'pace', 'physic',
    'age', 'cumulative_minutes_played', 'minutes_per_game_prev_seasons',
    'avg_days_injured_prev_seasons', 'cumulative_days_injured'
]

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Soccer Player Injury Predictor ⚽", layout="centered")
st.title("⚽ Soccer Player Injury Risk Prediction")
st.markdown("Enter player stats below to predict the risk of **major injury** using the trained machine learning model.")

# -----------------------------
# Input fields
# -----------------------------
height_cm = st.number_input("Height (cm)", min_value=150, max_value=210, value=180)
weight_kg = st.number_input("Weight (kg)", min_value=50, max_value=120, value=75)
work_rate_numeric = st.selectbox(
    "Work Rate (e.g., Low=1, Medium=1.5, High=2, High/Medium=3.5)",
    options=[1, 1.5, 2, 2.5, 3, 3.5], index=2
)
pace = st.number_input("Pace", min_value=20, max_value=100, value=70)
physic = st.number_input("Physic", min_value=20, max_value=100, value=68)

# Position input (map position to numeric)
position_map = {"Goalkeeper": 0, "Defender": 1, "Forward": 2, "Midfielder": 3}
position_label = st.selectbox("Position", options=list(position_map.keys()))
position_numeric = position_map[position_label]

age = st.number_input("Age", min_value=16, max_value=45, value=25)
cumulative_minutes_played = st.number_input("Cumulative Minutes Played", min_value=0, max_value=50000, value=5000)
minutes_per_game_prev_seasons = st.number_input("Minutes per Game (Prev Seasons)", min_value=0, max_value=100, value=80)
avg_days_injured_prev_seasons = st.number_input("Avg Days Injured (Prev Seasons)", min_value=0, max_value=200, value=10)

# Significant injury in previous season (binary)
sig_injury_label = st.selectbox("Significant Injury in Previous Season?", options=["No", "Yes"])
significant_injury_prev_season = 1 if sig_injury_label == "Yes" else 0

cumulative_days_injured = st.number_input("Cumulative Days Injured", min_value=0, max_value=1000, value=50)

# -----------------------------
# Create DataFrame for prediction
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
# Predict button
# -----------------------------
if st.button("🔍 Predict Injury Risk"):
    try:
        # Scale numeric features
        new_data[continuous_features] = scaler.transform(new_data[continuous_features])

        # Apply RFE feature selection
        new_data_rfe = rfe.transform(new_data)

        # Predict using trained model
        prediction = model.predict(new_data_rfe)[0]
        probability = model.predict_proba(new_data_rfe)[0][1]

        # Display result
        if prediction == 1:
            st.error(f"⚠️ Player is at **HIGH RISK** of major injury.\n\n**Probability:** {probability:.2%}")
        else:
            st.success(f"✅ Player is at **LOW RISK** of major injury.\n\n**Probability:** {probability:.2%}")

    except Exception as e:
        st.error(f"Error during prediction: {e}")

# -----------------------------
# Display model accuracy
# -----------------------------
model_accuracy = 0.73  # replace this with your actual model accuracy
st.markdown(f"---\n### 📊 Model Accuracy: **{model_accuracy*100:.2f}%**")
st.caption("Note: Accuracy shown is from the model training phase.")
