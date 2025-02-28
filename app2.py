# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 21:55:37 2025

@author: ameen
"""

import streamlit as st
import numpy as np
import joblib

# Load the trained models and scaler
scaler = joblib.load("scaler.pkl")
svr_model = joblib.load("svr_model.pkl")
rf_model = joblib.load("random_forest_model.pkl")
xgb_model = joblib.load("xgboost_model.pkl")

# Streamlit UI
st.title("Time Prediction App ⏳")
st.write("Enter input values to predict time using different models.")

# Input fields
feature_names = ["Inlet_Concentration", "Bed_Height", "Flow_Rate", "Outlet_Concentration"]
inputs = {feature: st.number_input(f"{feature}:", value=0.0, format="%.2f") for feature in feature_names}

# Convert inputs to a NumPy array
features = np.array([[inputs[feature] for feature in feature_names]])

# Scale inputs
features_scaled = scaler.transform(features)

# **Make Predictions before selecting the best model**
if st.button("Predict Time"):
    pred_svr = svr_model.predict(features_scaled)[0]  # SVR prediction
    pred_rf = rf_model.predict(features_scaled)[0]  # Random Forest prediction
    pred_xgb = xgb_model.predict(features_scaled)[0]  # XGBoost prediction

    # **Display all predictions**
    st.write(f"**SVR Prediction:** {pred_svr:.2f} minutes")
    st.write(f"**Random Forest Prediction:** {pred_rf:.2f} minutes")
    st.write(f"**XGBoost Prediction:** {pred_xgb:.2f} minutes")

    # **Select the Best Model with the Minimum Prediction**
    predictions = {"XGBoost": pred_xgb, "Random Forest": pred_rf, "SVR": pred_svr}
    best_model = min(predictions, key=predictions.get)  # Get model with the lowest predicted value
    st.success(f"🔹 Best Model: {best_model} with {predictions[best_model]:.2f} minutes")


