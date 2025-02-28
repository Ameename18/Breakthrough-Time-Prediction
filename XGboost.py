# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 23:55:02 2025

@author: ameen
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
file_path = "Minor Project-Data Set.xlsx"  # Change to your file path
df = pd.read_excel(r"C:\Users\ameen\OneDrive\Documents\Minor Project-Data Set.xlsx", sheet_name="Sheet1")

# Define features and target variable
target_column = "Time_h"  # Correct column name
X = df.drop(columns=[target_column])  # Drop the target column from features
y = df[target_column]  # Set the target variable


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling (XGBoost works well without scaling, but sometimes it helps)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# XGBoost Model with Hyperparameter Tuning
param_grid = {
    "learning_rate": [0.05, 0.1, 0.2],
    "max_depth": [3, 5, 7],
    "n_estimators": [100, 200, 300],
    "subsample": [0.7, 0.8, 1.0],
    "min_child_weight": [1, 2, 3]
}

xgb = XGBRegressor(objective="reg:squarederror", random_state=42)
grid_search = GridSearchCV(xgb, param_grid, cv=5, scoring="r2", n_jobs=-1, verbose=1)
grid_search.fit(X_train_scaled, y_train)

# Best model
best_xgb = grid_search.best_estimator_
y_pred = best_xgb.predict(X_test_scaled)

# Evaluation
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Optimized XGBoost R² Score: {r2:.4f}")
print(f"Optimized XGBoost RMSE: {rmse:.4f}")
print(f"Best Parameters: {grid_search.best_params_}")
