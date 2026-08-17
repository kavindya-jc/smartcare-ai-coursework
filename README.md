# smartcare-ai-coursework

🔗 **Live Demo:** [smartcare-ai-coursework.streamlit.app](https://smartcare-ai-coursework.streamlit.app/)

## Project Structure

```
smartcare-ai-coursework/
├── data/
│   ├── raw/                    # original CSVs, never edited
│   └── processed/              # cleaned data saved after Task 3
├── notebooks/
│   ├── 01_dataset_understanding.ipynb               # Task 2
│   ├── 02_preprocessing_feature_engineering.ipynb   # Task 3
│   ├── 03_eda.ipynb                                 # Task 4
│   ├── 04_model_development.ipynb                   # Task 5
│   ├── 05_model_evaluation.ipynb                    # Task 6
│   └── 06_explainable_ai.ipynb                      # Task 7
├── src/                        # reusable Python functions (later)
├── reports/
│   ├── figures/                # saved charts
│   └── technical_report(PDF)   # Task 9
├── models/                     # trained model files (Task 5)
├── prototype/                  # Streamlit/Flask app (Task 8)
├── docs/
│   ├── literature_review.md       # Task 1
└── README.md

```

## Setup

1. Clone the repo and `cd` into it
2. Create a virtual environment: `python -m venv venv`
3. Activate it: `venv\Scripts\activate` (Windows)
4. Install dependencies: `pip install -r requirements.txt`
5. Open notebooks in VS Code and select the `venv` kernel

## Reproducing This Project

1. Clone the repo and set up the environment (see Setup above).
2. Run notebooks in order: 01 → 02 → 03 → 04 → 05 → 06. Each notebook
   loads its input from data/raw/ or data/processed/ and saves its output
   there for the next notebook to use.
3. The Streamlit prototype requires logistic_regression.pkl, scaler.pkl, and preprocessing_artifacts.pkl from models/. The first two are produced by notebook 04; preprocessing_artifacts.pkl (containing IQR outlier bounds and dropdown category options for the UI) must also be present in models/ — confirm with the prototype's author whether this is committed to the repo or needs a separate generation step.
4. To run the prototype locally: cd prototype && streamlit run app.py
5. Live deployed version: (https://smartcare-ai-coursework.streamlit.app/)

## Dependencies

See requirements.txt for the full environment, or prototype/requirements.txt
for the minimal set needed just to run the Streamlit app.
