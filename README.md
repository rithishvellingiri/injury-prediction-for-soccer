Soccer Player Injury Prediction
Project Overview :

This project predicts whether a professional soccer player is at risk of a major injury during a season using machine learning.
The model was trained using FIFA player statistics, match history, and injury records, and saved as .pkl files.
A Streamlit web application provides an interactive interface for users to input player details and get injury risk predictions.

Project Structure :
Soccer_Injury_Prediction/
│
├── app.py                          # Streamlit app for predictions
├── injury_model.pkl                # Trained XGBoost model
├── feature_selector.pkl            # RFE feature selector
├── scaler.pkl                      # StandardScaler for preprocessing
├── soccer_injury_model.py          # Model training and evaluation code
├── df_injury_player_data_features_v1.csv  # Processed dataset
├── README.md                       # Project documentation
└── requirements.txt                # Python dependencies

Technologies Used :

Python 3.10+

Streamlit (UI)

XGBoost (Machine Learning)

scikit-learn (Feature scaling and preprocessing)

Pandas / NumPy

Joblib (Model saving/loading)

Installation & Setup :
1️Clone the Repository :
git clone https://github.com/your-username/soccer-injury-prediction.git
cd soccer-injury-prediction

2️Create a Virtual Environment :
python -m venv venv
source venv/bin/activate     # On macOS/Linux
venv\Scripts\activate        # On Windows

3️Install Dependencies :

If you have a requirements.txt file:

pip install -r requirements.txt


If not, install manually:

pip install streamlit scikit-learn xgboost joblib pandas numpy

Model Files :

The following .pkl files are required and must be in the same directory as app.py:

injury_model.pkl → Trained XGBoost classifier

feature_selector.pkl → Recursive Feature Eliminator (RFE) used for feature selection

scaler.pkl → StandardScaler for preprocessing numeric features

These files were created and saved during model training in soccer_injury_model.py using:

joblib.dump(final_xgb_clf, "injury_model.pkl")
joblib.dump(rfe, "feature_selector.pkl")
joblib.dump(scaler, "scaler.pkl")

How to Run the Application :
1️⃣ Ensure .pkl model files are in your project folder

Place the following files in the same directory as app.py:

injury_model.pkl
feature_selector.pkl
scaler.pkl

2️⃣ Run the Streamlit App
streamlit run app.py

3️⃣ Open in Browser

After running, Streamlit will automatically open:

http://localhost:8501

Using the App :

Input player details like height, weight, pace, age, cumulative minutes, etc.

Click the “Predict Injury Risk” button.

The app will display:

Predicted Injury Risk: High or Low

Model Accuracy (e.g., 73%)

📊 Model Information

Model Type: XGBoost Classifier

Accuracy: ~73%

Target: target_major_injury (1 = major injury, 0 = no major injury)

Key Predictive Features:

Age

Pace

Physic

Significant injury in previous season

Average days injured in previous seasons

Cumulative minutes played

Interpretation of Predictions :
Output	Meaning	Example Probability
Low Risk	Player has a low chance of major injury	0–30%
Medium Risk	Player has moderate chance of injury	30–60%
High Risk	Player is likely to face major injury	60–100%

📈 Future Improvements :

Include live FIFA player data via API

Add visualization for risk trends

Deploy the app on Streamlit Cloud, Render, or Heroku

Author :

Rithish Kumar V
📧 Contact: [rithishvellingiri@gmail.com]

License :

This project is licensed under the MIT License – feel free to use and modify it for educational purposes.
