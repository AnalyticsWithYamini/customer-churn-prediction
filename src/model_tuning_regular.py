import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_absolute_percentage_error
import xgboost as xgb
import joblib
import os
import shap
import numpy as np

# Step 1: Load the preprocessed dataset
file_path = '../data/ABC_Bank_Customer_Churn_preprocessed.csv'
data = pd.read_csv(file_path)

# Step 2: Define features and target
X = data.drop(columns=['churn'])
y = data['churn']

# Step 3: Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Define Hyperparameters for Random Forest and XGBoost

# Random Forest hyperparameter tuning using GridSearchCV
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf_model = RandomForestClassifier(random_state=42)
rf_grid_search = GridSearchCV(estimator=rf_model, param_grid=rf_param_grid, cv=3, n_jobs=-1, verbose=2)
rf_grid_search.fit(X_train, y_train)

# Best Random Forest model
best_rf_model = rf_grid_search.best_estimator_

# XGBoost hyperparameter tuning using RandomizedSearchCV
xgb_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'reg_alpha': [0, 0.1, 0.5],  # Adding L1 regularization
    'reg_lambda': [1, 1.5, 2.0]  # Adding L2 regularization
}

xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_random_search = RandomizedSearchCV(estimator=xgb_model, param_distributions=xgb_param_grid, n_iter=10, cv=3, n_jobs=-1, verbose=2, random_state=42)
xgb_random_search.fit(X_train, y_train)

# Best XGBoost model
best_xgb_model = xgb_random_search.best_estimator_

# Step 5: Perform cross-validation for both models
def cross_validate_model(model, X_train, y_train):
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"Cross-Validation Accuracy Scores: {cv_scores}")
    print(f"Mean CV Accuracy: {cv_scores.mean():.4f}")
    return cv_scores.mean()

# Cross-validate Random Forest
print("\nRandom Forest Cross-Validation:")
rf_cv_mean = cross_validate_model(best_rf_model, X_train, y_train)

# Cross-validate XGBoost
print("\nXGBoost Cross-Validation:")
xgb_cv_mean = cross_validate_model(best_xgb_model, X_train, y_train)

# Step 6: Evaluate both models on the test set and include MAPE and SMAPE
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    
    # MAPE (Mean Absolute Percentage Error) & SMAPE (Symmetric Mean Absolute Percentage Error)
    def smape(y_true, y_pred):
        return 100 / len(y_true) * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

    mape_value = mean_absolute_percentage_error(y_test, y_pred)
    smape_value = smape(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"AUC-ROC: {auc_roc:.4f}")
    print(f"MAPE: {mape_value:.4f}")
    print(f"SMAPE: {smape_value:.4f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc_roc': auc_roc,
        'mape': mape_value,
        'smape': smape_value
    }

# Evaluate Random Forest
print("\nRandom Forest Performance on Test Set:")
rf_test_metrics = evaluate_model(best_rf_model, X_test, y_test)

# Evaluate XGBoost
print("\nXGBoost Performance on Test Set:")
xgb_test_metrics = evaluate_model(best_xgb_model, X_test, y_test)

# Step 7: Save the best model based on cross-validation and test performance
model_dir = '../models'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

if rf_cv_mean > xgb_cv_mean:
    print("\nSaving Random Forest as the best model...")
    best_model = best_rf_model
    joblib.dump(best_model, '../models/random_forest_churn_model.pkl')
else:
    print("\nSaving XGBoost as the best model...")
    best_model = best_xgb_model
    joblib.dump(best_model, '../models/xgboost_churn_model.pkl')

print(f"Best model saved: {best_model}")

# Step 8: Model Interpretation with SHAP

# SHAP for Random Forest
explainer_rf = shap.TreeExplainer(best_rf_model)
shap_values_rf = explainer_rf.shap_values(X_test)

print("\nSHAP Summary for Random Forest:")
shap.summary_plot(shap_values_rf, X_test, plot_type="bar")

# SHAP for XGBoost
explainer_xgb = shap.TreeExplainer(best_xgb_model)
shap_values_xgb = explainer_xgb.shap_values(X_test)

print("\nSHAP Summary for XGBoost:")
shap.summary_plot(shap_values_xgb, X_test, plot_type="bar")
