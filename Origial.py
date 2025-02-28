# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 22:31:14 2025

@author: ameen
"""

# -- coding: utf-8 --
"""
Created on Tue Feb 11 00:10:24 2025

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

# Load dataset
file_path = "Minor Project-Data Set.xlsx"  # Change to your file path
df = pd.read_excel(r"C:\Users\ameen\OneDrive\Documents\Minor Project-Data Set.xlsx", sheet_name="Sheet1")

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

# Hyperparameter tuning
svr_params = {'C': [1, 10], 'kernel': ['linear'], 'degree': [2], 'gamma': ['scale']}
rf_params = {'n_estimators': [50], 'max_depth': [None], 'min_samples_split': [2], 'min_samples_leaf': [1]}
xgb_params = {'learning_rate': [0.05], 'n_estimators': [100], 'max_depth': [3], 'subsample': [0.7]}

# Model initialization
svr = GridSearchCV(SVR(), svr_params, cv=3)
rf = GridSearchCV(RandomForestRegressor(), rf_params, cv=3)
xgb = GridSearchCV(XGBRegressor(), xgb_params, cv=3)

# Training models
svr.fit(X_train_scaled, y_train)
rf.fit(X_train, y_train)
xgb.fit(X_train, y_train)

# Predictions
y_pred_svr = svr.predict(X_test_scaled)
y_pred_rf = rf.predict(X_test)
y_pred_xgb = xgb.predict(X_test)

# Evaluation
models = {'SVR': (y_test, y_pred_svr), 'Random Forest': (y_test, y_pred_rf), 'XGBoost': (y_test, y_pred_xgb)}
for name, (y_true, y_pred) in models.items():
    print(f"{name}:")
    print(f"R² Score: {r2_score(y_true, y_pred):.4f}")
    print(f"RMSE: {mean_squared_error(y_true, y_pred, squared=False):.4f}\n")

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