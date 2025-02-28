# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 20:32:21 2025

@author: ameen
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
file_path = "C:/Users/ameen/OneDrive/Documents/Minor Project-Data Set.xlsx"
df = pd.read_excel(file_path)

# Log transformation for target variables
df["Breakthrough Time (log)"] = np.log1p(df["Breakthrough Time (min)"])
df["Exhaustion Time (log)"] = np.log1p(df["Exhaustion Time (min)"])

# Drop original target columns
df = df.drop(["Breakthrough Time (min)", "Exhaustion Time (min)"], axis=1)

# Convert categorical 'Material Type' to numerical
df = pd.get_dummies(df, columns=['Material Type'], drop_first=True)

# Remove outliers using IQR method
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

# Define features and target variables
X = df.drop(["Breakthrough Time (log)", "Exhaustion Time (log)"], axis=1)
y_breakthrough = df["Breakthrough Time (log)"]
y_exhaustion = df["Exhaustion Time (log)"]

# Scale features using MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split data for both target variables
X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X_scaled, y_breakthrough, test_size=0.2, random_state=42)
X_train_e, X_test_e, y_train_e, y_test_e = train_test_split(X_scaled, y_exhaustion, test_size=0.2, random_state=42)

# --------------------- Model Training and Evaluation ---------------------

def train_and_evaluate(model, model_name):
    model.fit(X_train_b, y_train_b)
    y_pred_b = model.predict(X_test_b)
    model.fit(X_train_e, y_train_e)
    y_pred_e = model.predict(X_test_e)
    
    r2_b = r2_score(y_test_b, y_pred_b)
    r2_e = r2_score(y_test_e, y_pred_e)
    rmse_b = np.sqrt(mean_squared_error(y_test_b, y_pred_b))
    rmse_e = np.sqrt(mean_squared_error(y_test_e, y_pred_e))

    print(f"🔹 {model_name} Model Performance:")
    print(f"R² Score (Breakthrough Time): {r2_b:.4f}")
    print(f"R² Score (Exhaustion Time): {r2_e:.4f}")
    print(f"RMSE (Breakthrough Time): {rmse_b:.4f}")
    print(f"RMSE (Exhaustion Time): {rmse_e:.4f}")
    print("-" * 60)

# --------------------- Hyperparameter Tuning ---------------------

# Random Forest Tuning
rf_params = {'n_estimators': [50, 100, 200], 'max_depth': [None, 5, 10]}
rf_grid = GridSearchCV(RandomForestRegressor(), rf_params, cv=3, scoring='r2')
rf_grid.fit(X_train_b, y_train_b)
best_rf = rf_grid.best_estimator_

# XGBoost Tuning
xgb_params = {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 10], 'learning_rate': [0.01, 0.1, 0.2]}
xgb_grid = GridSearchCV(xgb.XGBRegressor(), xgb_params, cv=3, scoring='r2')
xgb_grid.fit(X_train_b, y_train_b)
best_xgb = xgb_grid.best_estimator_

# --------------------- Train Models ---------------------
train_and_evaluate(GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5), "Gradient Boosting")
train_and_evaluate(best_rf, "Random Forest")
train_and_evaluate(best_xgb, "XGBoost")



