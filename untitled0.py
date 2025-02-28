# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 22:47:32 2025

@author: ameen
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor

# Sample data
data = {
    'Inlet_Concentration': [250, 250, 250, 250, 250, 250, 250, 250, 250, 250],
    'Bed_Height': [6, 6, 6, 10, 10, 10, 6, 6, 10, 10],
    'Flow_Rate': [1, 4, 7, 1, 4, 7, 1, 4, 1, 4],
    'Outlet_Concentration': [142, 150, 160, 142, 150, 160, 142, 150, 142, 150],
    'Time_h': [186.85, 180, 170, 186.85, 180, 170, 186.85, 180, 186.85, 180]
}

df = pd.DataFrame(data)

# Define features and target
X = df[['Inlet_Concentration', 'Bed_Height', 'Flow_Rate', 'Outlet_Concentration']]
y = df['Time_h']

# Create polynomial features to capture nonlinearities
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# Create a pipeline with StandardScaler and SVR
pipe = make_pipeline(StandardScaler(), SVR())

# Hyperparameter tuning grid
param_grid = {
    'svr__C': [0.1, 1, 10, 50],
    'svr__kernel': ['linear', 'rbf', 'poly'],
    'svr__gamma': ['scale', 'auto'],
    'svr__epsilon': [0.01, 0.1, 0.5, 1]
}

# Grid search for best parameters
grid = GridSearchCV(pipe, param_grid=param_grid, cv=5)
grid.fit(X_train, y_train)

# Best parameters
best_params = grid.best_params_
print("Best SVR Hyperparameters:", best_params)

# Predict using best model
y_pred = grid.predict(X_test)

# Performance metrics
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'Improved SVR Model - R^2: {r2:.2f}, RMSE: {rmse:.2f}')

# Try Random Forest as an alternative model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
r2_rf = r2_score(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
print(f'Random Forest Model - R^2: {r2_rf:.2f}, RMSE: {rmse_rf:.2f}')

# Plot Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', label='SVR Predictions')
plt.scatter(y_test, y_pred_rf, color='green', label='RF Predictions')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Ideal Line')
plt.xlabel('Actual Time (h)')
plt.ylabel('Predicted Time (h)')
plt.title(f'Actual vs Predicted Time\nSVR R² = {r2:.2f}, RF R² = {r2_rf:.2f}')
plt.legend()
plt.show()
