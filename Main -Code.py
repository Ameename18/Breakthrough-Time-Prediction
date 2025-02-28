# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 20:32:21 2025

@author: ameen
"""

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error

# Load dataset
file_path = "C:/Users/ameen/OneDrive/Documents/Minor Project-Data Set.xlsx"
df = pd.read_excel(file_path)

# Drop NaN values
df = df.dropna()
if df.empty:
    raise ValueError("Dataset is empty after dropping NaN values!")

# Encode categorical variable
df["Material Type"] = df["Material Type"].map({"RPP": 0, "GCS": 1})

# Separate datasets for RPP & GCS
df_rpp = df[df["Material Type"] == 0]
df_gcs = df[df["Material Type"] == 1]

# Define function to train models
def train_and_evaluate(df, material_name):
    print(f"\n🔹 **{material_name} Model Performance:**\n")
    
    # Define features & target
    X = df.drop(columns=["Breakthrough Time (min)", "Exhaustion Time (min)", "Material Type"])
    y_breakthrough = df["Breakthrough Time (min)"]
    y_exhaustion = df["Exhaustion Time (min)"]
    
    # Ensure enough samples
    if X.shape[0] < 2:
        raise ValueError(f"Not enough samples in {material_name} dataset for train-test split!")

    # Split data
    X_train, X_test, y_train_bt, y_test_bt = train_test_split(X, y_breakthrough, test_size=0.2, random_state=42)
    X_train, X_test, y_train_ex, y_test_ex = train_test_split(X, y_exhaustion, test_size=0.2, random_state=42)
    
    # Models to train
    models = {
        "Gradient Boosting Regressor": GradientBoostingRegressor(),
        "Random Forest Regressor": RandomForestRegressor(),
        "XGBoost Regressor": xgb.XGBRegressor()
    }

    # Train & evaluate each model
    for model_name, model in models.items():
        model.fit(X_train, y_train_bt)
        y_pred_bt = model.predict(X_test)

        model.fit(X_train, y_train_ex)
        y_pred_ex = model.predict(X_test)

        # Compute metrics
        r2_bt = r2_score(y_test_bt, y_pred_bt)
        r2_ex = r2_score(y_test_ex, y_pred_ex)
        rmse_bt = np.sqrt(mean_squared_error(y_test_bt, y_pred_bt))
        rmse_ex = np.sqrt(mean_squared_error(y_test_ex, y_pred_ex))

        # Print performance
        print(f"🔹 {model_name} Performance:")
        print(f"R² Score (Breakthrough Time): {r2_bt:.4f}")
        print(f"R² Score (Exhaustion Time): {r2_ex:.4f}")
        print(f"RMSE (Breakthrough Time): {rmse_bt:.4f}")
        print(f"RMSE (Exhaustion Time): {rmse_ex:.4f}")
        print("-" * 60)

# Train models for RPP & GCS separately
train_and_evaluate(df_rpp, "RPP")
train_and_evaluate(df_gcs, "GCS")








