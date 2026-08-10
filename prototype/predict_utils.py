"""
Shared preprocessing + prediction logic for the Disease Risk prototype.
Mirrors the exact feature engineering pipeline from notebook 02
(preprocessing_feature_engineering) and notebook 03 (model_development),
so a single new patient record is transformed identically to training data.
"""
import pandas as pd
import numpy as np
import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_artifacts():
    model = joblib.load(os.path.join(BASE_DIR, 'models', 'logistic_regression.pkl'))
    split = joblib.load(os.path.join(BASE_DIR, 'models', 'train_test_split.pkl'))
    prep = joblib.load(os.path.join(BASE_DIR, 'models', 'preprocessing_artifacts.pkl'))
    scaler = joblib.load(os.path.join(BASE_DIR, 'models', 'scaler.pkl'))
    prep['scaler'] = scaler
    prep['numeric_cols'] = split['numeric_cols']
    return model, split['feature_columns'], prep


def build_feature_row(raw_input: dict, feature_columns: list, prep: dict) -> pd.DataFrame:
    """
    raw_input: dict of raw patient values, keys matching cleaned_data.csv columns
               (age, gender, blood_group, department, diagnosis, appointment_status,
                waiting_days, previous_appointments, missed_previous_appointments,
                admitted, room_type, length_of_stay_days, previous_admissions,
                systolic_bp, diastolic_bp, blood_sugar_mg_dl, cholesterol_mg_dl, bmi,
                lab_tests_count, treatments_count, consultation_fee_lkr, room_charge_lkr,
                lab_charge_lkr, medicine_charge_lkr, total_bill_lkr,
                payment_status, payment_method)
    Returns a single-row, fully preprocessed (encoded + scaled) DataFrame
    ready for model.predict / model.predict_proba.
    """
    row = raw_input.copy()

    # 1. Feature engineering (same logic as notebook 02)
    systolic = row['systolic_bp']
    if systolic <= 120:
        bp_category = 'Normal'
    elif systolic <= 140:
        bp_category = 'Elevated'
    else:
        bp_category = 'High'

    bmi = row['bmi']
    if bmi <= 18.5:
        bmi_category = 'Underweight'
    elif bmi <= 25:
        bmi_category = 'Normal'
    elif bmi <= 30:
        bmi_category = 'Overweight'
    else:
        bmi_category = 'Obese'

    prev_appts = row['previous_appointments'] if row['previous_appointments'] != 0 else 1
    missed_rate = row['missed_previous_appointments'] / prev_appts

    # 2. Outlier flags, computed with the same IQR bounds fitted in notebook 02
    outlier_flags = {}
    for col in ['systolic_bp', 'diastolic_bp', 'blood_sugar_mg_dl', 'cholesterol_mg_dl', 'bmi']:
        lower, upper = prep['iqr_bounds'][col]
        outlier_flags[f'{col}_outlier_flag'] = not (lower <= row[col] <= upper)

    # 3. Missing-data flags: not applicable to a freshly entered record
    room_type_missing_flag = False
    room_charge_missing_flag = False

    # 4. Assemble the raw (pre-encoding) single-row record
    df_row = pd.DataFrame([{
        **row,
        'bp_category': bp_category,
        'bmi_category': bmi_category,
        'missed_appointment_rate': missed_rate,
        'room_type_missing_flag': room_type_missing_flag,
        'room_charge_missing_flag': room_charge_missing_flag,
        **outlier_flags,
    }])

    # 5. One-hot encode categoricals exactly as in notebook 02 (drop_first=True)
    cat_cols = ['gender', 'blood_group', 'department', 'diagnosis', 'appointment_status',
                'room_type', 'payment_status', 'payment_method', 'bp_category', 'bmi_category']
    df_encoded = pd.get_dummies(df_row, columns=cat_cols, drop_first=False)
    # drop_first=False here (single row can't represent "dropped" reference category),
    # then reindex to the trained column set below, which achieves the same drop_first effect.

    # 6. Scale numeric columns with the scaler fitted on cleaned_data.csv
    scaler = prep['scaler']
    numeric_cols = prep['numeric_cols']
    df_encoded[numeric_cols] = scaler.transform(df_encoded[numeric_cols])

    # 7. Align to the exact columns/order the model was trained on
    df_final = df_encoded.reindex(columns=feature_columns, fill_value=0)
    df_final = df_final.astype(float)

    return df_final


def predict_risk(raw_input: dict):
    model, feature_columns, prep = load_artifacts()
    X = build_feature_row(raw_input, feature_columns, prep)
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    proba_dict = dict(zip(model.classes_, proba))
    return pred, proba_dict, X
