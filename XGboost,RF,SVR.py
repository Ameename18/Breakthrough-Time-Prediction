# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 22:52:12 2025

@author: ameen
"""
# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

# Load dataset from Excel file
file_path = "Minor Project-Data Set.xlsx"  # Change to your file path
df = pd.read_excel(r"C:\Users\ameen\OneDrive\Documents\Minor Project-Data Set.xlsx", sheet_name="Sheet1")

# Display first few rows to verify
print(df.head())

# Define features (X) and target (y)
X = df[['Inlet_Concentration', 'Bed_Height', 'Flow_Rate', 'Outlet_Concentration']]
y = df['Time_h']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
svr_pipe = make_pipeline(StandardScaler(), SVR())
rf_pipe = make_pipeline(StandardScaler(), RandomForestRegressor(random_state=42))
xgb_pipe = make_pipeline(StandardScaler(), XGBRegressor(objective="reg:squarederror", random_state=42))

# Define hyperparameter grids
svr_param_grid = {
    'svr__C': [0.1, 1, 10],
    'svr__kernel': ['linear', 'rbf', 'poly'],
    'svr__gamma': ['scale', 'auto'],
    'svr__degree': [2, 3, 4]
}

rf_param_grid = {
    'randomforestregressor__n_estimators': [50, 100, 200],
    'randomforestregressor__max_depth': [None, 5, 10],
    'randomforestregressor__min_samples_split': [2, 5],
    'randomforestregressor__min_samples_leaf': [1, 2]
}

xgb_param_grid = {
    'xgbregressor__n_estimators': [50, 100, 200],
    'xgbregressor__learning_rate': [0.01, 0.1, 0.2],
    'xgbregressor__max_depth': [3, 5, 7]
}

# Adjust cross-validation folds to avoid issues with small datasets
cv_folds = min(3, len(X_train))  # Use at most 3-fold CV if possible

# Train SVR model using GridSearchCV
svr_grid = GridSearchCV(svr_pipe, param_grid=svr_param_grid, cv=cv_folds, scoring='r2', n_jobs=-1)
svr_grid.fit(X_train, y_train)
svr_best = svr_grid.best_estimator_
y_pred_svr = svr_best.predict(X_test)
r2_svr = r2_score(y_test, y_pred_svr)
rmse_svr = np.sqrt(mean_squared_error(y_test, y_pred_svr))

# Train Random Forest model using GridSearchCV
rf_grid = GridSearchCV(rf_pipe, param_grid=rf_param_grid, cv=cv_folds, scoring='r2', n_jobs=-1)
rf_grid.fit(X_train, y_train)
rf_best = rf_grid.best_estimator_
y_pred_rf = rf_best.predict(X_test)
r2_rf = r2_score(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))

# Train XGBoost model using GridSearchCV
xgb_grid = GridSearchCV(xgb_pipe, param_grid=xgb_param_grid, cv=cv_folds, scoring='r2', n_jobs=-1)
xgb_grid.fit(X_train, y_train)
xgb_best = xgb_grid.best_estimator_
y_pred_xgb = xgb_best.predict(X_test)
r2_xgb = r2_score(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))

# Display results
results = {
    "SVR": {"R²": r2_svr, "RMSE": rmse_svr, "Best Params": svr_grid.best_params_},
    "Random Forest": {"R²": r2_rf, "RMSE": rmse_rf, "Best Params": rf_grid.best_params_},
    "XGBoost": {"R²": r2_xgb, "RMSE": rmse_xgb, "Best Params": xgb_grid.best_params_}
}

print("Model Comparison Results:")
for model, metrics in results.items():
    print(f"\n{model}:")
    print(f"  R² Score: {metrics['R²']:.4f}")
    print(f"  RMSE: {metrics['RMSE']:.4f}")
    print(f"  Best Parameters: {metrics['Best Params']}")

# Plot actual vs predicted values for the best model
best_model = max(results, key=lambda m: results[m]["R²"])  # Select model with highest R²
best_pred = {"SVR": y_pred_svr, "Random Forest": y_pred_rf, "XGBoost": y_pred_xgb}[best_model]

plt.figure(figsize=(8, 6))
plt.scatter(y_test, best_pred, color='blue', label='Actual vs Predicted')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Ideal Line')
plt.xlabel('Actual Time (h)')
plt.ylabel('Predicted Time (h)')
plt.title(f'Actual vs Predicted Time (h) using {best_model}\nR² = {results[best_model]["R²"]:.2f}, RMSE = {results[best_model]["RMSE"]:.2f}')
plt.legend()
plt.show()
