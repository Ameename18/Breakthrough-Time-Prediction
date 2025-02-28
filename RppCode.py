# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 19:31:07 2025

@author: ameen
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error

# Load dataset
file_path = "C:/Users/ameen/OneDrive/Documents/Minor Project/DatsetRPP.xlsx"
df = pd.read_excel(file_path)

# Drop NaN values
df = df.dropna()
if df.empty:
    raise ValueError("Dataset is empty after dropping NaN values!")

# Define function to train models
def train_and_evaluate(df):
    print("\n🔹 **Model Training & Evaluation:**\n")
    
    # Define features & target
    X = df.drop(columns=["Breakthrough Time (min)"])
    y_breakthrough = df["Breakthrough Time (min)"]
    
    # Ensure enough samples
    if X.shape[0] < 2:
        raise ValueError("Not enough samples in dataset for train-test split!")
    
    # Split data
    X_train, X_test, y_train_bt, y_test_bt = train_test_split(X, y_breakthrough, test_size=0.2, random_state=42)
    
    # Models to train
    models = {
        "Gradient Boosting Regressor": GradientBoostingRegressor(),
        "Random Forest Regressor": RandomForestRegressor(),
        "XGBoost Regressor": xgb.XGBRegressor()
    }
    
    trained_models = {}
    
    # Train & evaluate each model
    for model_name, model in models.items():
        model.fit(X_train, y_train_bt)
        y_pred_bt = model.predict(X_test)
        
        # Compute metrics
        r2_bt = r2_score(y_test_bt, y_pred_bt)
        rmse_bt = np.sqrt(mean_squared_error(y_test_bt, y_pred_bt))
        
        # Print performance
        print(f"🔹 {model_name} Performance:")
        print(f"R² Score (Breakthrough Time): {r2_bt:.4f}")
        print(f"RMSE (Breakthrough Time): {rmse_bt:.4f}")
        print("-" * 60)
        
        trained_models[model_name] = model
    
    return trained_models

# Train models
trained_models = train_and_evaluate(df)

# Extract feature names from dataset
feature_names = [col for col in df.columns if col not in ["Breakthrough Time (min)"]]
print("\nFeatures used for training:", feature_names)

# Take user input for prediction
print("\n🔹 **Enter Feature Values for Prediction:**")
user_values = []
for col in feature_names:
    value = float(input(f"Enter {col}: "))
    user_values.append(value)

# Create user input DataFrame with correct feature names
user_input = pd.DataFrame([user_values], columns=feature_names)
print("\nUser input received:", user_input)

# Use trained models for prediction
for model_name, model in trained_models.items():
    breakthrough_pred = model.predict(user_input)[0]
    print(f"\n🔹 {model_name} Predictions:")
    print(f"Predicted Breakthrough Time: {breakthrough_pred:.2f} min")







