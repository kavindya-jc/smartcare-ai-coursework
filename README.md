# smartcare-ai-coursework

## Project Structure

smartcare-ai-coursework/
├── data/
│ ├── raw/ # original CSVs, never edit these directly
│ └── processed/ # cleaned/engineered datasets
├── notebooks/
│ ├── 01_dataset_understanding.ipynb
│ ├── 02_preprocessing_feature_engineering.ipynb
│ ├── 03_eda.ipynb
│ ├── 04_modeling.ipynb
│ ├── 05_evaluation.ipynb
│ └── 06_explainability.ipynb
├── src/ # reusable Python functions
├── reports/
│ ├── figures/ # saved charts
│ └── technical_report.docx # Task 9
├── models/ # trained .pkl / .joblib files
├── prototype/ # Streamlit or Flask app, Task 8
└── docs/
├── literature_review.md # Task 1
├── data_dictionary_notes.md
└── deliverables_tracker.md

## Setup

1. Clone the repo and `cd` into it
2. Create a virtual environment: `python -m venv venv`
3. Activate it: `venv\Scripts\activate` (Windows)
4. Install dependencies: `pip install pandas numpy matplotlib seaborn scikit-learn jupyter ipykernel`
5. Open notebooks in VS Code and select the `venv` kernel

## Branch Workflow

Work on a feature branch, not directly on `main`:
git checkout -b yourname/task-name
Then open a Pull Request to merge into main when your part is ready.
