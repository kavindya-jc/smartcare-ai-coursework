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
│   ├── 03_eda.ipynb            # Task 4
│   ├── 04_modeling.ipynb
│   ├── 05_evaluation.ipynb     # Task 6
│   └── 06_explainability.ipynb # Task 7
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
