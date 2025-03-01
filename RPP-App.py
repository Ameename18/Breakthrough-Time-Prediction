# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 20:42:50 2025

@author: ameen
"""

import numpy as np
import pandas as pd
import joblib
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
import os

# Define dataset path dynamically
file_path = "https://raw.githubusercontent.com/Ameename18/Breakthrough-Time-Prediction/main/DatsetRPP.xlsx"

# Load dataset
try:
    df = pd.read_excel(file_path, engine="openpyxl")  # Ensure correct engine for .xlsx files
    df.dropna(inplace=True)  # Handle missing values
except Exception as e:
    st.error(f"Failed to load dataset: {str(e)}")
    st.stop()

# Define features and target variable
features = ["Q (mL/min)", "Z (cm)", "C₀ (mg/L)", "Mass of Adsorbent (g)", "Mass Transfer Zone (cm)"]
target = "Breakthrough Time (min)"

# Split data
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models and hyperparameters
models = {
    "Gradient Boosting": GradientBoostingRegressor(),
    "Random Forest": RandomForestRegressor(),
    "XGBoost": XGBRegressor()
}

param_grid = {
    "Gradient Boosting": {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1]},
    "Random Forest": {"n_estimators": [100, 200]},
    "XGBoost": {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1]}
}

best_models = {}

# Train or Load Models
for name, model in models.items():
    model_path = f"{name.replace(' ', '_').lower()}_model.pkl"
    if os.path.exists(model_path):
        best_models[name] = joblib.load(model_path)
    else:
        try:
            grid_search = GridSearchCV(model, param_grid[name], scoring="r2", cv=3, n_jobs=-1, error_score="raise")
            grid_search.fit(X_train, y_train)
            best_models[name] = grid_search.best_estimator_
            joblib.dump(best_models[name], model_path)
        except Exception as e:
            st.warning(f"Model training for {name} failed: {str(e)}")

# Streamlit UI
st.title("🌟 Breakthrough Time Prediction App")
st.write("Enter the feature values below to predict breakthrough time.")

# Sidebar Input
st.sidebar.header("Enter Feature Values")
user_data = {}

for feature in features:
    user_data[feature] = st.sidebar.number_input(f"{feature}", min_value=0.0, format="%.2f")

# Predict Button
if st.sidebar.button("Predict"):
    user_input = pd.DataFrame([user_data])

    if not best_models:
        st.error("No trained models available for prediction.")
    else:
        predictions = {name: model.predict(user_input)[0] for name, model in best_models.items()}
        
        st.write("## 🔍 Predictions:")
        for name, value in predictions.items():
            st.success(f"**{name}:** {value:.2f} min")


