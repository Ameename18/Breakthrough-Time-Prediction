# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 00:06:37 2025

@author: ameen
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor

# Load dataset
file_path = "Minor Project-Data Set.xlsx"  # Change to your file path
df = pd.read_excel(r"C:\Users\ameen\OneDrive\Documents\Minor Project-Data Set.xlsx", sheet_name="Sheet1")

# Check for missing values
df.dropna(inplace=True)

# Define features and target
X = df.drop(columns=["Time_h"])  # Adjust target column name if needed
y = df["Time_h"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check test size
if len(y_test) < 2:
    raise ValueError("Test set must have at least two samples for R² calculation.")

# Train XGBoost model
xgb_model = XGBRegressor(learning_rate=0.05, max_depth=3, min_child_weight=1, 
                         n_estimators=100, subsample=0.7, random_state=42)
xgb_model.fit(X_train, y_train)

# Predictions
y_pred = xgb_model.predict(X_test)

# Evaluation
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Optimized XGBoost R² Score: {r2:.4f}")
print(f"Optimized XGBoost RMSE: {rmse:.4f}")
