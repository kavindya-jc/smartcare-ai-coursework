"""
Task 08 - AI Prototype Development
SmartCare Hospital - Disease Risk Prediction (Option C)

A Streamlit app that accepts patient information, runs it through the same
preprocessing pipeline used in training, and predicts disease risk level
(Low / Medium / High) using the Logistic Regression model selected in Task 6.
"""
import streamlit as st
import pandas as pd
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predict_utils import predict_risk, load_artifacts

st.set_page_config(page_title="SmartCare Disease Risk Predictor", page_icon="🏥", layout="centered")

st.title("🏥 SmartCare Hospital — Disease Risk Predictor")
st.caption(
    "AI-powered decision-support prototype (Task 8). "
    "Predicts a patient's disease risk level — Low, Medium, or High — "
    "using a Logistic Regression model trained on the SmartCare Hospital AI dataset."
)

_, _, prep = load_artifacts()
cat_options = prep['cat_options']
ranges = prep['raw_ranges']

with st.form("patient_form"):
    st.subheader("Patient Information")
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age", min_value=1, max_value=110, value=45)
    with c2:
        gender = st.selectbox("Gender", cat_options['gender'])
    with c3:
        blood_group = st.selectbox("Blood Group", cat_options['blood_group'])

    st.subheader("Clinical Measurements")
    c1, c2, c3 = st.columns(3)
    with c1:
        systolic_bp = st.number_input("Systolic BP", min_value=70, max_value=220, value=120)
        cholesterol_mg_dl = st.number_input("Cholesterol (mg/dL)", min_value=80, max_value=400, value=190)
    with c2:
        diastolic_bp = st.number_input("Diastolic BP", min_value=40, max_value=140, value=80)
        bmi = st.number_input("BMI", min_value=10.0, max_value=55.0, value=24.0, step=0.1)
    with c3:
        blood_sugar_mg_dl = st.number_input("Blood Sugar (mg/dL)", min_value=50, max_value=400, value=100)
        diagnosis = st.selectbox("Diagnosis", cat_options['diagnosis'])

    st.subheader("Hospital Visit / Operations")
    c1, c2, c3 = st.columns(3)
    with c1:
        department = st.selectbox("Department", cat_options['department'])
        waiting_days = st.number_input("Waiting Days", min_value=0, max_value=90, value=3)
        previous_appointments = st.number_input("Previous Appointments", min_value=0, max_value=30, value=2)
    with c2:
        appointment_status = st.selectbox("Appointment Status", cat_options['appointment_status'])
        missed_previous_appointments = st.number_input("Missed Previous Appointments", min_value=0, max_value=20, value=0)
        previous_admissions = st.number_input("Previous Admissions", min_value=0, max_value=20, value=0)
    with c3:
        admitted = st.selectbox("Currently Admitted?", ["No", "Yes"])
        room_type = st.selectbox("Room Type", cat_options['room_type'])
        length_of_stay_days = st.number_input("Length of Stay (days)", min_value=0, max_value=60, value=0)

    st.subheader("Lab / Treatment")
    c1, c2 = st.columns(2)
    with c1:
        lab_tests_count = st.number_input("Lab Tests Count", min_value=0, max_value=30, value=2)
    with c2:
        treatments_count = st.number_input("Treatments Count", min_value=0, max_value=30, value=1)

    st.subheader("Billing")
    c1, c2, c3 = st.columns(3)
    with c1:
        consultation_fee_lkr = st.number_input("Consultation Fee (LKR)", min_value=0, value=2500)
        room_charge_lkr = st.number_input("Room Charge (LKR)", min_value=0, value=0)
    with c2:
        lab_charge_lkr = st.number_input("Lab Charge (LKR)", min_value=0, value=1500)
        medicine_charge_lkr = st.number_input("Medicine Charge (LKR)", min_value=0, value=2000)
    with c3:
        payment_status = st.selectbox("Payment Status", cat_options['payment_status'])
        payment_method = st.selectbox("Payment Method", cat_options['payment_method'])

    total_bill_lkr = consultation_fee_lkr + room_charge_lkr + lab_charge_lkr + medicine_charge_lkr
    st.info(f"Total Bill (auto-calculated): **{total_bill_lkr:,} LKR**")

    submitted = st.form_submit_button("Predict Disease Risk", type="primary", width='stretch')

if submitted:
    raw_input = {
        'age': age, 'gender': gender, 'blood_group': blood_group,
        'department': department, 'diagnosis': diagnosis,
        'appointment_status': appointment_status, 'waiting_days': waiting_days,
        'previous_appointments': previous_appointments,
        'missed_previous_appointments': missed_previous_appointments,
        'admitted': 1 if admitted == "Yes" else 0, 'room_type': room_type,
        'length_of_stay_days': length_of_stay_days, 'previous_admissions': previous_admissions,
        'systolic_bp': systolic_bp, 'diastolic_bp': diastolic_bp,
        'blood_sugar_mg_dl': blood_sugar_mg_dl, 'cholesterol_mg_dl': cholesterol_mg_dl,
        'bmi': bmi, 'lab_tests_count': lab_tests_count, 'treatments_count': treatments_count,
        'consultation_fee_lkr': consultation_fee_lkr, 'room_charge_lkr': room_charge_lkr,
        'lab_charge_lkr': lab_charge_lkr, 'medicine_charge_lkr': medicine_charge_lkr,
        'total_bill_lkr': total_bill_lkr, 'payment_status': payment_status,
        'payment_method': payment_method,
    }

    pred, proba, X = predict_risk(raw_input)

    st.divider()
    st.subheader("Prediction Result")

    risk_colors = {"Low": "green", "Medium": "orange", "High": "red"}
    st.markdown(f"### Predicted Disease Risk: :{risk_colors[pred]}[**{pred}**]")

    proba_df = pd.DataFrame({
        "Risk Level": list(proba.keys()),
        "Probability": [round(v * 100, 1) for v in proba.values()]
    }).sort_values("Probability", ascending=False)

    st.bar_chart(proba_df.set_index("Risk Level"))
    st.dataframe(proba_df, hide_index=True, width='stretch')

    st.caption(
        "⚠️ This is a decision-support prototype for coursework purposes only, "
        "not a certified diagnostic tool. Predictions should be reviewed by a qualified clinician."
    )
