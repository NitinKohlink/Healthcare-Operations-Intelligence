# Development Prompt — Healthcare Operations Intelligence

Build the project exactly as a reproducible, educational healthcare data analytics application.

## Product
Healthcare Operations Intelligence

## Source
Synthetic healthcare dataset with 150,500 raw rows and 14 original columns.

## Core requirements
- Never alter the raw data file.
- Implement cleaning as reusable Python functions.
- Remove exact duplicates.
- Normalize categorical values.
- Cap oxygen saturation values above 100 at 100.
- Replace extreme systolic BP values (<60 or >200) with NaN and median-impute.
- Median-impute numeric missing values.
- Mode-impute categorical missing values.
- Generate row-level quality flags.
- Create age group, duration band and pain band features.
- Build a Streamlit dashboard.
- Provide interactive filters.
- Provide department workload intelligence.
- Provide data-quality audit.
- Provide symptom/dept leakage audit.
- Provide ML benchmark using department as target and excluding symptoms.
- Compare majority baseline, Logistic Regression, Decision Tree and Random Forest.
- Report accuracy, balanced accuracy and macro-F1.
- Include clear healthcare disclaimer on every user-facing page.
- Do not provide diagnosis or treatment advice.
- Add automated tests.

## Dashboard pages
1. Executive Overview
2. Data Quality
3. Department Intelligence
4. Patient & Symptom Explorer
5. ML Audit
6. Downloads

## Engineering rules
- Python 3.10+
- pandas/numpy/scikit-learn
- Plotly + Streamlit
- Keep code modular
- Use random_state=42
- Make the project reproducible
- No hard-coded model results in the dashboard when artifacts can be generated from the provided dataset
