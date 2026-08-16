#!/usr/bin/env python
# coding: utf-8

# In[27]:


import pandas as pd
df = pd.read_csv('../data/raw/smartcare_ai_dataset_1000.csv')

# In[2]:


# Flag the admitted=1 but room_type missing
df['room_type_missing_flag'] = df['room_type'].isnull() & (df['admitted'] == 1)

# In[3]:


# Fill the missing room_type values (906) identified in data understanding section
df['room_type'] = df['room_type'].fillna('Not Recorded')

# In[4]:


# Confirm no missing values remain
df.isnull().sum().sum()

# In[5]:


# Re-check duplicates
df.duplicated().sum() 

# In[6]:


# Outlier identification - IQR method applied consistently across all clinical columns
clinical_cols = ['systolic_bp','diastolic_bp','blood_sugar_mg_dl','cholesterol_mg_dl','bmi']

for col in clinical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5*IQR
    upper = Q3 + 1.5*IQR
    df[f'{col}_outlier_flag'] = ~df[col].between(lower, upper)
    print(col, '- outliers found:', df[f'{col}_outlier_flag'].sum())

# In[7]:


# Extra Investigation
df[df['bmi'] < 15][['age', 'bmi']]

# In[8]:


# Admitted patients should generally have some room charge - check for admitted=1 but room_charge=0
df[(df['admitted'] == 1) & (df['room_charge_lkr'] == 0)][['admitted', 'room_charge_lkr']]

# In[9]:


# Flag the admitted=1 but room_charge=0
df['room_charge_missing_flag'] = (df['admitted'] == 1) & (df['room_charge_lkr'] == 0)

# In[10]:


# Checks whether the "missing room type" group and the "zero room charge" group are the same 236 patients or different ones
(df['room_type_missing_flag'] & df['room_charge_missing_flag']).sum()

# In[11]:


# Admitted patients should have length_of_stay > 0
df[(df['admitted'] == 1) & (df['length_of_stay_days'] == 0)]

# In[12]:


# Non-admitted patients shouldn't have a room charge at all
df[(df['admitted'] == 0) & (df['room_charge_lkr'] > 0)]

# ## Data Cleaning Workflow
# 
# 1. Missing values: Only room_type had missing values (906 of 1000). 670 were 
#    patients not admitted (expected — filled as "Not Recorded"). The remaining 
#    236 were admitted patients missing room_type, flagged with 
#    room_type_missing_flag before filling, so the information isn't lost even 
#    though the column itself is now filled.
# 
# 2. Duplicates: No duplicate rows found (df.duplicated().sum() = 0).
# 
# 3. Outlier identification: Applied the IQR method (1.5×IQR rule) consistently 
#    across all clinical columns (systolic_bp, diastolic_bp, blood_sugar_mg_dl, 
#    cholesterol_mg_dl, bmi), creating a flag column for each rather than 
#    checking columns individually by different methods. Results: systolic_bp 
#    (3), diastolic_bp (3), blood_sugar_mg_dl (10), cholesterol_mg_dl (6), bmi 
#    (9). These counts are small (under 1% of records per column) and consistent 
#    with what's expected from natural variation in real clinical data using a 
#    standard statistical threshold. BMI was investigated further as a case 
#    study — cross-checked against age, confirming 8 of 9 low-BMI cases were 
#    adults rather than children, ruling out age as an explanation. None of the 
#    flagged rows were removed, to avoid losing valid data without strong 
#    justification; they remain available for exclusion or weighting during 
#    Task 5 modeling if needed.
# 
# 4. Cross-column consistency check: found that the same 236 admitted patients 
#    missing room_type also had zero room_charge_lkr (confirmed by 
#    (df['room_type_missing_flag'] & df['room_charge_missing_flag']).sum() = 236, 
#    a full match). This means it's one single data entry gap for this subgroup, 
#    not two separate issues. Flagged with room_charge_missing_flag rather than 
#    removed, since dropping 236 rows would mean losing 23.6% of all admitted 
#    patients. No issues found with length_of_stay_days or non-admitted billing 
#    consistency.

# In[13]:


# Create a blood pressure category
df['bp_category'] = pd.cut(df['systolic_bp'], bins=[0,120,140,200], labels=['Normal','Elevated','High'])

# In[14]:


# Create a BMI category
df['bmi_category'] = pd.cut(df['bmi'], bins=[0,18.5,25,30,50], labels=['Underweight','Normal','Overweight','Obese'])

# In[15]:


# Create a missed appointment rate
df['missed_appointment_rate'] = df['missed_previous_appointments'] / df['previous_appointments'].replace(0,1)

# In[16]:


# Encode all categorical columns
df_encoded = pd.get_dummies(df, columns=['gender','blood_group','department','diagnosis','appointment_status','room_type','payment_status','payment_method','bp_category','bmi_category'], drop_first=True)

# In[17]:


# Check the new shape
df_encoded.shape

# In[18]:


# Import the scaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# In[19]:


# List which columns to scale
numeric_cols = ['age','waiting_days','previous_appointments','missed_previous_appointments','length_of_stay_days','previous_admissions','systolic_bp','diastolic_bp','blood_sugar_mg_dl','cholesterol_mg_dl','bmi','lab_tests_count','treatments_count','consultation_fee_lkr','room_charge_lkr','lab_charge_lkr','medicine_charge_lkr','total_bill_lkr','missed_appointment_rate']

# In[20]:


# Apply the scaling
# df_encoded[numeric_cols] = scaler.fit_transform(df_encoded[numeric_cols])

# In[21]:


# Confirm numeric columns are still raw (unscaled) - scaling deferred to Task 5
df_encoded[numeric_cols].describe().T[['mean','std']]

# In[22]:


# Note on feature-target correlation
risk_map = {'Low': 0, 'Medium': 1, 'High': 2}
df_encoded['disease_risk_numeric'] = df['disease_risk_level'].map(risk_map)
correlations = df_encoded.select_dtypes(include='number').corr()['disease_risk_numeric'].sort_values(ascending=False)
correlations

# In[23]:


# Check for redundant features
df_encoded.select_dtypes(include='number').corr()

# ## Feature Engineering Process
# 
# 1. Created bp_category (Normal/Elevated/High) from systolic_bp and bmi_category 
#    (Underweight/Normal/Overweight/Obese) from bmi, to capture standard clinical 
#    threshold effects that raw numbers alone might not represent well to a model.
# 
# 2. Created missed_appointment_rate (missed_previous_appointments / 
#    previous_appointments) to express missed appointments as a proportion rather 
#    than a raw count, making patients with different appointment histories more 
#    comparable.
# 
# 3. Feature encoding: applied one-hot encoding (pd.get_dummies, drop_first=True) 
#    to all categorical columns (gender, blood_group, department, diagnosis, 
#    appointment_status, room_type, payment_status, payment_method, bp_category, 
#    bmi_category), since ML models require numeric input. Dataset shape after 
#    cleaning and feature engineering was (1000, 43); after encoding and adding 
#    the temporary numeric target mapping for correlation analysis, the final 
#    dataset shape is (1000, 73). Numeric columns remain unscaled at this stage 
#    (see point 4).
# 
# 4. Feature scaling: StandardScaler will be used, but applied only after the 
#    train/test split (in Task 5), not here. Scaling before splitting would let 
#    statistics from the test set (mean, standard deviation) leak into the 
#    training process, since the scaler would be calculated using all 1000 rows 
#    including future test data. To demonstrate the technique is understood, 
#    here's how it will be applied later:
#    
#    X_train_scaled = scaler.fit_transform(X_train)   # fit only on training data
#    X_test_scaled = scaler.transform(X_test)          # test data never influences the scaler
#    
#    model_ready_data.csv therefore contains encoded categorical features and 
#    engineered features, but numeric columns remain unscaled — scaling is 
#    deferred to Task 5.
# 
# 5. Feature selection: checked correlation between all numeric features to 
#    identify redundancy — no pairs exceeded 0.9, though room_charge_lkr/
#    total_bill_lkr (0.88), lab_tests_count/lab_charge_lkr (0.876), and admitted/
#    length_of_stay_days (0.834) were notably correlated, as expected given their 
#    natural relationship. None were removed since none crossed the redundancy 
#    threshold.
# 
#    Also checked correlation against the target (disease_risk_level, temporarily 
#    mapped to 0/1/2 for this check only). Clinical features showed the strongest 
#    relationship: age (0.54), blood_sugar_mg_dl (0.47), cholesterol_mg_dl (0.45), 
#    bmi (0.36), and systolic_bp (0.31). Financial and operational features (e.g. 
#    billing, lab test counts) showed very weak correlation (under 0.05), 
#    suggesting clinical measurements will likely be the strongest predictors in 
#    Task 5. This numeric mapping was for exploratory purposes only — the actual 
#    target encoding for modeling will be finalized in Task 5.

# In[30]:


# Save cleaned and model-ready datasets
df.to_csv('../data/processed/cleaned_data.csv', index=False) 
df_encoded.to_csv('../data/processed/model_ready_data.csv', index=False)

# df (cleaned, with flags and new categories, but still readable text like "Male", "High") — best for Task 4 EDA, since charts and plots are easier to read with actual category names instead of 0s and 1s.
# 
# df_encoded (one-hot encoded + scaled) — best for Task 5 modeling, since models need numbers, not text.

# In[25]:


df.shape

# In[26]:


df_encoded.shape
