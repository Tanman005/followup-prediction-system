import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd

from src.preprocessing import load_data, preprocess_data

st.title("🏥 Follow-Up Appointment Prediction System")

# Load and preprocess data
df = load_data("data/appointments.csv")
df = preprocess_data(df)

# Train model (Random Forest)
from sklearn.ensemble import RandomForestClassifier

df_model = df.drop(['ScheduledDay', 'AppointmentDay', 'Neighbourhood'], axis=1)

X = df_model.drop('FollowUp', axis=1)
y = df_model['FollowUp']

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

st.subheader("Enter Patient Details:")

# Inputs
gender = st.selectbox("Gender", ["Female", "Male"])
age = st.slider("Age", 0, 100, 25)
hypertension = st.selectbox("Hypertension", [0, 1])
diabetes = st.selectbox("Diabetes", [0, 1])
alcoholism = st.selectbox("Alcoholism", [0, 1])
sms = st.selectbox("SMS Received", [0, 1])
waiting_days = st.slider("Waiting Days", 0, 30, 5)

# Convert inputs
gender_val = 1 if gender == "Male" else 0

input_data = pd.DataFrame([{
    'Gender': gender_val,
    'Age': age,
    'Scholarship': 0,
    'Hipertension': hypertension,
    'Diabetes': diabetes,
    'Alcoholism': alcoholism,
    'Handcap': 0,
    'SMS_received': sms,
    'WaitingDays': waiting_days
}])

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("✅ Patient is likely to NEED follow-up")
    else:
        st.error("❌ Patient is NOT likely to need follow-up")