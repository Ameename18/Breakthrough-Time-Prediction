# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 16:42:44 2025

@author: ameen
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load dataset
file_path = r"C:\Users\ameen\OneDrive\Documents\Minor Project\DataSET1.xlsx"
df = pd.read_excel(file_path, sheet_name="Sheet1")

# Drop NaN values
df.dropna(inplace=True)

# Feature-target split
X = df.drop(columns=["Time_h"])  # Features
y = df["Time_h"]  # Target

# Standardization (for all models to keep consistency)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler
joblib.dump(scaler, "scaler.pkl")

# K-Fold Cross Validation
kf = KFold(n_splits=3, shuffle=True, random_state=42)

# Models and parameters
models = {
    "SVR": SVR(C=10, kernel='linear', degree=2, gamma='scale'),
    "Random Forest": RandomForestRegressor(n_estimators=50, max_depth=None, min_samples_split=2, min_samples_leaf=1),
    "XGBoost": XGBRegressor(learning_rate=0.05, n_estimators=100, max_depth=3, subsample=0.7)
}

# Training and evaluation
for name, model in models.items():
    scores = cross_val_score(model, X_scaled, y, cv=kf, scoring="r2")
    rmse_scores = np.sqrt(-cross_val_score(model, X_scaled, y, cv=kf, scoring="neg_mean_squared_error"))

    print(f"\n📌 {name} Model Performance:")
    print(f"Mean R² Score: {np.nanmean(scores):.4f}")  # Ignore NaNs safely
    print(f"Mean RMSE: {np.mean(rmse_scores):.4f}")

    # Save trained models
    model.fit(X_scaled, y)
    joblib.dump(model, f"{name.lower().replace(' ', '_')}_model.pkl")

print("✅ Models and scaler saved successfully!")

# Heatmap for feature correlation
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()
