# forest_fire_app.py
# -------------------------------------------------------
# AI-Driven Forest Fire Prediction using Streamlit & Sklearn
# -------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

# 1. Load Dataset
@st.cache_data
def load_data():
    return pd.read_csv("forestfires.csv")

data = load_data()

# 2. Preprocess Data
# Using dataset columns: temp = temperature, RH = humidity, wind = wind speed
X = data[['temp', 'RH', 'wind']]
# Create fire risk label based on area burnt (threshold can be changed)
y = np.where(data['area'] > 0, 1, 0)  # 1 = High risk, 0 = Low risk

# 3. Train Model
@st.cache_resource
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Save model for reuse
    joblib.dump(model, "fire_risk_model.pkl")
    
    return model, accuracy

if os.path.exists("fire_risk_model.pkl"):
    model = joblib.load("fire_risk_model.pkl")
    # Rough accuracy from retraining if not stored
    _, accuracy = train_model(X, y)
else:
    model, accuracy = train_model(X, y)

# 4. Streamlit UI
st.title("🌲🔥 AI-Driven Forest Fire Prediction")
st.write(f"**Model Accuracy:** {accuracy*100:.2f}%")

# Sidebar inputs
st.sidebar.header("Input Environmental Parameters")
temp = st.sidebar.slider("Temperature (°C)", 0, 40, 20, key="temp_slider")
humidity = st.sidebar.slider("Humidity (%)", 0, 100, 50, key="humidity_slider")
wind = st.sidebar.slider("Wind Speed (km/h)", 0, 50, 10, key="wind_slider")

# Prediction
if st.sidebar.button("Predict Fire Risk", key="predict_btn"):
    features = np.array([[temp, humidity, wind]])
    risk = model.predict(features)[0]
    prob = model.predict_proba(features)[0][1]

    if risk == 1:
        st.error(f"🔥 High Fire Risk! Probability: {prob*100:.2f}%")
    else:
        st.success(f"✅ Low Fire Risk. Probability: {prob*100:.2f}%")

# 5. Show feature importance
st.subheader("Feature Importance")
importances = model.feature_importances_
st.bar_chart(pd.DataFrame({
    'Feature': ['Temperature', 'Humidity', 'Wind Speed'],
    'Importance': importances
}).set_index('Feature'))

# 6. Show dataset
if st.checkbox("Show Raw Data", key="show_data"):
    st.subheader("Forest Fire Dataset")
    st.dataframe(data.head(20))
