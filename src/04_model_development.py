#!/usr/bin/env python
# coding: utf-8

# # Task 05 – Machine Learning Model Development

# In[1]:


# Load the model-ready dataset
import pandas as pd
import numpy as np

df = pd.read_csv('../data/processed/model_ready_data.csv')
df.shape

# ## Preparing features and target
# 
# **Important — avoiding data leakage:** the numeric columns in this file (age, bmi, blood pressure,
# billing amounts, etc.) are still on their *raw* scale — they have **not** been scaled yet. That's
# deliberate: scaling must be fit only on the training data, *after* the train/test split, otherwise
# the scaler's mean/std would be calculated using statistics from the test set too, letting the model
# indirectly "see" test data before evaluation. So the order here is: split first, scale second.

# In[2]:


# Separate identifiers/other-task targets from the feature set
drop_cols = ['record_id', 'patient_id', 'appointment_date',
             'no_show', 'readmitted_30_days', 'disease_risk_numeric']

X = df.drop(columns=drop_cols + ['disease_risk_level'])
y = df['disease_risk_level']

# The one-hot/flag columns were saved as boolean dtype - convert to int for sklearn/XGBoost
bool_cols = X.select_dtypes(include='bool').columns
X[bool_cols] = X[bool_cols].astype(int)

print('Feature matrix shape:', X.shape)
print('Target distribution:')
y.value_counts()

# ## Train/test split
# 
# The target is imbalanced (Medium 469, High 400, Low 131), so we use **stratified** splitting to keep
# class proportions consistent between train and test sets.

# In[3]:


from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print('Train class distribution:')
print(y_train.value_counts(normalize=True).round(3))
print('\nTest class distribution:')
print(y_test.value_counts(normalize=True).round(3))

# ## Feature scaling (fit on training data only)
# 
# `StandardScaler` is **fit on `X_train` only**, then used to transform both `X_train` and `X_test`.
# This is the correct order to prevent data leakage: the test set is never seen during fitting.

# In[4]:


from sklearn.preprocessing import StandardScaler

numeric_cols = ['age', 'waiting_days', 'previous_appointments', 'missed_previous_appointments',
                 'length_of_stay_days', 'previous_admissions', 'systolic_bp', 'diastolic_bp',
                 'blood_sugar_mg_dl', 'cholesterol_mg_dl', 'bmi', 'lab_tests_count', 'treatments_count',
                 'consultation_fee_lkr', 'room_charge_lkr', 'lab_charge_lkr', 'medicine_charge_lkr',
                 'total_bill_lkr', 'missed_appointment_rate']

scaler = StandardScaler()
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])   # fit ONLY on training data
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])          # test data never influences the scaler

X_train[numeric_cols].describe().T[['mean', 'std']].round(3)

# In[5]:


# XGBoost requires a numeric target, so we also prepare a numeric-encoded version
risk_map = {'Low': 0, 'Medium': 1, 'High': 2}
y_train_num = y_train.map(risk_map)
y_test_num = y_test.map(risk_map)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ## Model 1 – Logistic Regression
# 
# A linear baseline. Tuned over the regularization strength `C` using 5-fold cross-validated
# macro-F1 (macro-F1 rather than accuracy, since the target is imbalanced and every class matters
# equally for a clinical risk tool).

# In[6]:


from sklearn.linear_model import LogisticRegression

lr_param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
lr_grid = GridSearchCV(
    LogisticRegression(max_iter=2000),
    lr_param_grid, cv=cv, scoring='f1_macro', n_jobs=-1
)
lr_grid.fit(X_train, y_train)

print('Best params:', lr_grid.best_params_)
print('Best CV macro-F1:', round(lr_grid.best_score_, 4))
best_lr = lr_grid.best_estimator_

# ## Model 2 – Random Forest
# 
# An ensemble of decision trees. Tuned over number of trees, tree depth, and minimum leaf size.
# (Tree-based models don't strictly need scaled features, but we keep the scaled `X_train`/`X_test`
# consistent across all three models for a fair, identical-input comparison.)

# In[7]:


from sklearn.ensemble import RandomForestClassifier

rf_param_grid = {
    'n_estimators': [200, 400],
    'max_depth': [None, 10, 20],
    'min_samples_leaf': [1, 2]
}
rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    rf_param_grid, cv=cv, scoring='f1_macro', n_jobs=-1
)
rf_grid.fit(X_train, y_train)

print('Best params:', rf_grid.best_params_)
print('Best CV macro-F1:', round(rf_grid.best_score_, 4))
best_rf = rf_grid.best_estimator_

# ## Model 3 – XGBoost
# 
# A gradient-boosted tree ensemble. Tuned over number of estimators, tree depth, and learning rate.

# In[8]:


from xgboost import XGBClassifier

xgb_param_grid = {
    'n_estimators': [200, 300],
    'max_depth': [3, 5],
    'learning_rate': [0.05, 0.1]
}
xgb_grid = GridSearchCV(
    XGBClassifier(eval_metric='mlogloss', random_state=42),
    xgb_param_grid, cv=cv, scoring='f1_macro', n_jobs=-1
)
xgb_grid.fit(X_train, y_train_num)

print('Best params:', xgb_grid.best_params_)
print('Best CV macro-F1:', round(xgb_grid.best_score_, 4))
best_xgb = xgb_grid.best_estimator_

# ## Comparative Analysis
# 
# Cross-validated (training) macro-F1 for each tuned model, side by side.

# In[9]:


comparison = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest', 'XGBoost'],
    'Best Hyperparameters': [lr_grid.best_params_, rf_grid.best_params_, xgb_grid.best_params_],
    'CV Macro-F1 (train)': [lr_grid.best_score_, rf_grid.best_score_, xgb_grid.best_score_]
}).sort_values('CV Macro-F1 (train)', ascending=False).reset_index(drop=True)

comparison

# **Observation:** Logistic Regression achieves the highest cross-validated macro-F1 by a clear
# margin, ahead of XGBoost and Random Forest. This is consistent with the correlation analysis from
# Task 3, which found that a small set of clinical features (age, blood_sugar_mg_dl, cholesterol_mg_dl,
# bmi, systolic_bp) are strongly and fairly linearly related to disease_risk_level — since this is a
# teaching dataset built around clinical thresholds, a linear decision boundary separates the classes
# well, which favors Logistic Regression over the tree ensembles here. Full test-set evaluation and
# final model justification is carried out in Task 6.

# In[10]:


# Save all three fitted models, the scaler, and the reusable train/test split for Task 6, 7 and 8
import joblib, os

os.makedirs('../models', exist_ok=True)
joblib.dump(best_lr, '../models/logistic_regression.pkl')
joblib.dump(best_rf, '../models/random_forest.pkl')
joblib.dump(best_xgb, '../models/xgboost_model.pkl')
joblib.dump(scaler, '../models/scaler.pkl')

joblib.dump({
    'X_train': X_train, 'X_test': X_test,
    'y_train': y_train, 'y_test': y_test,
    'y_train_num': y_train_num, 'y_test_num': y_test_num,
    'risk_map': risk_map,
    'feature_columns': list(X.columns),
    'numeric_cols': numeric_cols
}, '../models/train_test_split.pkl')

print('Saved: logistic_regression.pkl, random_forest.pkl, xgboost_model.pkl, scaler.pkl, train_test_split.pkl')
