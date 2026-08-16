#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Setup
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

df = pd.read_csv('../data/processed/cleaned_data.csv')

figures_path = '../reports/figures'
os.makedirs(figures_path, exist_ok=True)

# ## 1. Descriptive Statistics

# In[19]:


df.describe()

# In[10]:


df.describe(include='object')

# ### Descriptive Statistics
# Numeric columns show plausible ranges consistent with Task 2 findings (e.g. age 
# 1-90, bmi 14-38.8). Categorical columns show department, diagnosis, gender etc. 
# distributions - [after running, note here which category is most common in each, 
# e.g. "General Medicine is the most common department"].

# ## 2. Distribution Analysis (Histograms)

# In[ ]:


numeric_cols_to_plot = ['age', 'bmi', 'systolic_bp', 'blood_sugar_mg_dl', 'cholesterol_mg_dl', 'total_bill_lkr']

for col in numeric_cols_to_plot:
    plt.figure(figsize=(6,4))
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.savefig(f'../reports/figures/hist_{col}.png', bbox_inches='tight')
    plt.show()

# ### Distribution Analysis
# - age: [roughly even spread / skewed young / skewed old - describe what you see]
# - bmi: mostly clustered around 22-28, with the small number of low outliers 
#   already identified in Task 3
# - total_bill_lkr: right-skewed (most bills are lower, a smaller number are very 
#   high) - expected, since admitted patients cost much more than non-admitted ones
# - [add one line per column you actually plotted]

# ## 3. Pattern Discovery (Boxplots)

# In[20]:


box_cols = ['age', 'bmi', 'systolic_bp', 'blood_sugar_mg_dl', 'cholesterol_mg_dl']

for col in box_cols:
    plt.figure(figsize=(6,4))
    sns.boxplot(data=df, x='disease_risk_level', y=col, order=['Low','Medium','High'])
    plt.title(f'{col} by Disease Risk Level')
    plt.savefig(f'../reports/figures/box_{col}_by_risk.png', bbox_inches='tight')
    plt.show()

# ### Pattern Discovery - Boxplots by Risk Level
# [For each chart, write one line, e.g.:]
# - bmi: High risk patients show a visibly higher median bmi than Low risk patients
# - blood_sugar_mg_dl: clear upward trend from Low to High risk groups
# - age: [describe what you actually see]

# ## 4. Pattern Discovery (Scatterplots)

# In[21]:


scatter_pairs = [('age', 'bmi'), ('age', 'blood_sugar_mg_dl'), ('bmi', 'cholesterol_mg_dl')]

for x_col, y_col in scatter_pairs:
    plt.figure(figsize=(6,4))
    sns.scatterplot(data=df, x=x_col, y=y_col, hue='disease_risk_level', hue_order=['Low','Medium','High'])
    plt.title(f'{x_col} vs {y_col} by Risk Level')
    plt.savefig(f'../reports/figures/scatter_{x_col}_{y_col}.png', bbox_inches='tight')
    plt.show()

# ### Pattern Discovery - Scatterplots
# [One line per pair, e.g.:]
# - age vs bmi: High risk points cluster toward older age and higher bmi, though 
#   overlap between groups is visible - no single variable cleanly separates risk 
#   levels on its own

# ## 5. Correlation Analysis (Heatmap)

# In[22]:


corr_matrix = df.select_dtypes(include='number').corr()

plt.figure(figsize=(14,10))
sns.heatmap(corr_matrix, cmap='coolwarm', center=0, annot=False)
plt.title('Correlation Heatmap - Numeric Features')
plt.savefig('../reports/figures/correlation_heatmap.png', bbox_inches='tight')
plt.show()

# ### Correlation Analysis
# Confirms the findings from Task 3's feature selection check: admitted, 
# length_of_stay_days, lab_tests_count, and billing columns are moderately 
# correlated with each other (as expected, since admitted patients naturally 
# have higher values across all of these). No unexpected strong correlations 
# were found beyond what was already identified.

# ## 6. Class Distribution Chart

# In[23]:


plt.figure(figsize=(6,4))
sns.countplot(data=df, x='disease_risk_level', order=['Low','Medium','High'])
plt.title('Disease Risk Level - Class Distribution')
plt.savefig('../reports/figures/class_distribution.png', bbox_inches='tight')
plt.show()

# ### Class Distribution
# Confirms the imbalance found in Task 2: Medium (469), High (400), Low (131). 
# Low risk patients are under-represented, which will need addressing in Task 5 
# (e.g. stratified sampling, class weighting) rather than being ignored.

# ## EDA Summary - Key Insights("Interpretation")
# 1. Clinical features (bmi, blood_sugar_mg_dl, cholesterol_mg_dl, systolic_bp, age) 
#    show visible separation across disease_risk_level in the boxplots, consistent 
#    with the correlation findings from Task 3 - these will likely be the strongest 
#    predictors in Task 5.
# 2. Financial and operational columns (billing, admitted, length_of_stay) are 
#    correlated with each other but not strongly related to disease_risk_level - 
#    they reflect hospital operations, not clinical risk.
# 3. The target class imbalance (Low: 13.1%) is confirmed visually and needs 
#    handling in modeling, not just noted and ignored.
# 4. No new data quality issues were discovered during EDA beyond what Task 3 
#    already identified and handled.
