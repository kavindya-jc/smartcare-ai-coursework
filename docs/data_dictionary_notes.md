## Dataset Overview

The dataset contains 1000 hospital records with 33 columns, covering patient
demographics, clinical measurements, hospital operations, and billing data.
There are no duplicate rows. The dataset is shared across three possible AI
tasks (Options A, B, C) and includes three target columns — this project uses
disease_risk_level (Option C).

## Attribute Description

Columns are grouped into 6 categories:

- Identifiers: record_id, patient_id
- Patient info: age, gender, blood_group
- Clinical: diagnosis, systolic_bp, diastolic_bp, blood_sugar_mg_dl, cholesterol_mg_dl, bmi
- Operations: department, appointment_date, waiting_days, previous_appointments,
  missed_previous_appointments, appointment_status, admitted, room_type,
  length_of_stay_days, previous_admissions, lab_tests_count, treatments_count
- Financial: consultation_fee_lkr, room_charge_lkr, lab_charge_lkr,
  medicine_charge_lkr, total_bill_lkr, payment_status, payment_method
- Targets: no_show, readmitted_30_days, disease_risk_level

## Data Dictionary Interpretation

- No duplicate rows found.
- Only room_type has missing values: 906 out of 1000 rows (90.6%).
  - 670 of these are patients who were not admitted (admitted=0) — expected,
    since room_type doesn't apply if a patient wasn't admitted.
  - 236 of these are patients who WERE admitted (admitted=1) but still have no
    room_type recorded — this is a genuine data quality issue, not expected
    behavior, and will be handled separately from the 670 in preprocessing.
- disease_risk_level (our target) is imbalanced: Medium 469, High 400, Low 131.
  Low is under-represented, which will affect model training and evaluation
  in later tasks (stratified splitting, per-class metrics needed instead of
  plain accuracy).
- No impossible values found in numeric columns (no negative ages, charges, etc).
