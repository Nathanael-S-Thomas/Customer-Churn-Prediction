# Customer Churn Prediction and Retention Analysis Using Machine Learning

A medium-level college Data Science project built with Python, Scikit-learn, Plotly and Streamlit.

## Features
- Executive dashboard with customer KPIs and churn visualizations
- Customer-level churn probability estimator
- Logistic Regression vs Random Forest comparison
- Accuracy, precision, recall, F1-score, ROC-AUC and confusion matrix
- Retention insights and suggested follow-up actions

## Run locally
1. Install Python 3.10 or newer.
2. Open Terminal in this folder.
3. Create and activate a virtual environment:
   - macOS/Linux: `python3 -m venv .venv && source .venv/bin/activate`
   - Windows: `py -m venv .venv` then `.venv\\Scripts\\activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Run: `streamlit run app.py`

## Dataset note
`data/customer_churn_demo.csv` is synthetic demonstration data generated for this starter project. It allows the app to run offline and is not real customer data. For final submission, replace it with a documented public customer churn dataset, align the feature names in `app.py`, and retrain/evaluate the models. Do not claim the demo model's metrics represent real-world performance.

## Suggested project workflow
1. Describe the business problem and objective.
2. Explain the dataset and its limitations.
3. Perform exploratory data analysis.
4. Preprocess numeric and categorical features.
5. Train Logistic Regression and Random Forest.
6. Evaluate using a held-out test set.
7. Demonstrate the dashboard and customer risk form.
8. Discuss ethical use, false positives/negatives, and limitations.

## Folder structure
- `app.py` - Streamlit app, preprocessing pipeline, model training, dashboard and predictor
- `data/customer_churn_demo.csv` - synthetic demo data
- `requirements.txt` - Python dependencies
