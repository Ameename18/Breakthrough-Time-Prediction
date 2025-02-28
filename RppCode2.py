# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 19:31:07 2025

@author: ameen
"""
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
import os

# Load dataset
file_path = "C:/Users/ameen/OneDrive/Documents/Minor Project/DatsetRPP.xlsx"
df = pd.read_excel(file_path)

# Handle missing values
df.dropna(inplace=True)

# Define features and target variables
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

for name, model in models.items():
    model_path = f"{name.replace(' ', '_').lower()}_model.pkl"
    if os.path.exists(model_path):
        best_models[name] = joblib.load(model_path)
        print(f"🔹 Loaded pre-trained {name} model.")
    else:
        try:
            grid_search = GridSearchCV(model, param_grid[name], scoring="r2", cv=3, n_jobs=-1, error_score="raise")
            grid_search.fit(X_train, y_train)
            best_models[name] = grid_search.best_estimator_
            joblib.dump(best_models[name], model_path)
            print(f"🔹 Trained and saved {name} model.")
        except Exception as e:
            print(f"⚠️ Error training {name}: {e}")
            continue
    
    y_pred = best_models[name].predict(X_test)
    print(f"\n🔹 {name} Performance:")
    print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
    print("-" * 50)

# User input prediction
print("\n🔹 **Enter Feature Values for Prediction:**")
user_data = {}
for feature in features:
    while True:
        try:
            user_data[feature] = float(input(f"Enter {feature}: "))
            break
        except ValueError:
            print(f"⚠️ Invalid input for {feature}. Please enter a numeric value.")

user_input = pd.DataFrame([user_data])

# Load best model and make prediction
for name, model in best_models.items():
    breakthrough_pred = model.predict(user_input)[0]
    print(f"\n🔹 {name} Predictions:")
    print(f"Predicted Breakthrough Time: {breakthrough_pred:.2f} min")










