# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 12:27:51 2025

@author: ameen
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
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

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardization (only for SVR)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler (required for SVR predictions)
joblib.dump(scaler, "scaler.pkl")

# Hyperparameter tuning
svr_params = {'C': [1, 10], 'kernel': ['linear'], 'degree': [2], 'gamma': ['scale']}
rf_params = {'n_estimators': [50], 'max_depth': [None], 'min_samples_split': [2], 'min_samples_leaf': [1]}
xgb_params = {'learning_rate': [0.05], 'n_estimators': [100], 'max_depth': [3], 'subsample': [0.7]}

# Model initialization with GridSearchCV
svr = GridSearchCV(SVR(), svr_params, cv=3)
rf = GridSearchCV(RandomForestRegressor(), rf_params, cv=3)
xgb = GridSearchCV(XGBRegressor(), xgb_params, cv=3)

# Training models
print("Training SVR...")
svr.fit(X_train_scaled, y_train)  # Uses scaled input
print("Training Random Forest...")
rf.fit(X_train, y_train)  # Uses raw input
print("Training XGBoost...")
xgb.fit(X_train, y_train)  # Uses raw input

# Save trained models
joblib.dump(svr.best_estimator_, "svr_model.pkl")
joblib.dump(rf.best_estimator_, "random_forest_model.pkl")
joblib.dump(xgb.best_estimator_, "xgboost_model.pkl")

print("✅ Models and scaler saved successfully!")

# Predictions
y_pred_svr = svr.best_estimator_.predict(X_test_scaled)
y_pred_rf = rf.best_estimator_.predict(X_test)
y_pred_xgb = xgb.best_estimator_.predict(X_test)

# Evaluation
models = {
    'SVR': (y_test, y_pred_svr),
    'Random Forest': (y_test, y_pred_rf),
    'XGBoost': (y_test, y_pred_xgb)
}

for name, (y_true, y_pred) in models.items():
    print(f"\n📌 {name} Model Performance:")
    print(f"R² Score: {r2_score(y_true, y_pred):.4f}")
    print(f"RMSE: {mean_squared_error(y_true, y_pred, squared=False):.4f}")

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, (name, (y_true, y_pred)) in zip(axes, models.items()):
    ax.scatter(y_true, y_pred, label="Actual vs Predicted", color='blue')
    ax.plot(y_true, y_true, 'r--', label="Ideal Line")
    ax.set_title(f"{name}")
    ax.set_xlabel("Actual Time (h)")
    ax.set_ylabel("Predicted Time (h)")
    ax.legend()
plt.tight_layout()
plt.show()

# Heatmap for feature correlation
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

