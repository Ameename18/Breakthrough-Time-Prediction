# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 11:48:50 2025

@author: ameen
"""

import streamlit as st
import numpy as np
import joblib

# Load the trained model and scaler
model_filename = "xgboost_model.pkl"  # Ensure this matches your trained model file
scaler_filename = "scaler.pkl"

model = joblib.load(model_filename)
scaler = joblib.load(scaler_filename)

# Streamlit App UI
st.title("Time Prediction App ⏳")
st.write("Enter the input values to get time predictions.")

# Define input features
feature_names = ["Inlet_Concentration", "Bed_Height", "Flow_Rate", "Outlet_Concentration"]
inputs = {}

for feature in feature_names:
    inputs[feature] = st.number_input(f"{feature}:", value=0.0, format="%.2f")

# Convert inputs to a NumPy array
features = np.array([[inputs[feature] for feature in feature_names]])

# Scale the input features
features_scaled = scaler.transform(features)

# Predict when the button is clicked
if st.button("Predict Time"):
    prediction = model.predict(features_scaled)
    st.success(f"Predicted Time: {prediction[0]:.2f} minutes")


