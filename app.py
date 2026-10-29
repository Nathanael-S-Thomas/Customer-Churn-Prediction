import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

st.set_page_config(page_title="ChurnGuard | Customer Analytics", page_icon="📊", layout="wide")
ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "customer_churn_demo.csv"
MODEL_PATH = ROOT / "models" / "churn_model.joblib"
FEATURES = ["tenure_months","monthly_charges","contract_type","internet_service","payment_method","tech_support","paperless_billing","senior_citizen","dependents"]
CAT = ["contract_type","internet_service","payment_method","tech_support","paperless_billing","dependents"]
NUM = ["tenure_months","monthly_charges","senior_citizen"]

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def train_models(data):
    X = data[FEATURES]
    y = data["churn"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pre = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUM),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore"))]), CAT)
    ])
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=250, max_depth=10, min_samples_leaf=3, class_weight="balanced", random_state=42)
    }
    fitted, metrics = {}, []
    for name, clf in models.items():
        pipe = Pipeline([("preprocess", pre), ("model", clf)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        score = pipe.predict_proba(X_test)[:,1]
        fitted[name] = pipe
        metrics.append({"Model":name,"Accuracy":accuracy_score(y_test,pred),"Precision":precision_score(y_test,pred,zero_division=0),
                        "Recall":recall_score(y_test,pred,zero_division=0),"F1-score":f1_score(y_test,pred,zero_division=0),
                        "ROC-AUC":roc_auc_score(y_test,score)})
    return fitted, pd.DataFrame(metrics), X_test, y_test

data = load_data()
models, metrics, X_test, y_test = train_models(data)
st.sidebar.title("📊 ChurnGuard")
st.sidebar.caption("Customer Churn Prediction & Retention Analysis")
page = st.sidebar.radio("Navigate", ["Executive Dashboard","Customer Risk Predictor","Model Performance","Retention Insights","About Project"])
st.sidebar.markdown("---")
st.sidebar.caption("Academic demonstration • Demo dataset is synthetic")

if page == "Executive Dashboard":
    st.title("Customer Churn Analytics")
    st.write("Understand customer churn patterns and explore risk indicators.")
    total = len(data); churn_rate = data["churn"].mean()*100
    avg_charge = data["monthly_charges"].mean(); avg_tenure = data["tenure_months"].mean()
    a,b,c,d = st.columns(4)
    a.metric("Customers analysed", f"{total:,}")
    b.metric("Churn rate", f"{churn_rate:.1f}%")
    c.metric("Average monthly charge", f"₹{avg_charge:,.0f}")
    d.metric("Average tenure", f"{avg_tenure:.1f} months")
    st.markdown("### Churn overview")
    left,right = st.columns(2)
    with left:
        counts = data["churn"].map({0:"Stayed",1:"Churned"}).value_counts().rename_axis("Customer status").reset_index(name="Customers")
        st.plotly_chart(px.pie(counts, names="Customer status", values="Customers", hole=.55, title="Customer retention split"), use_container_width=True)
    with right:
        contract_rate = data.groupby("contract_type", observed=True)["churn"].mean().mul(100).reset_index(name="Churn rate (%)")
        st.plotly_chart(px.bar(contract_rate, x="contract_type", y="Churn rate (%)", color="contract_type", title="Churn rate by contract type"), use_container_width=True)
    left,right = st.columns(2)
    with left:
        st.plotly_chart(px.histogram(data, x="tenure_months", color=data["churn"].map({0:"Stayed",1:"Churned"}), barmode="overlay", opacity=.72, title="Tenure distribution by churn status", labels={"color":"Status"}), use_container_width=True)
    with right:
        st.plotly_chart(px.box(data, x=data["churn"].map({0:"Stayed",1:"Churned"}), y="monthly_charges", color=data["churn"].map({0:"Stayed",1:"Churned"}), title="Monthly charges by churn status", labels={"x":"Customer status","color":"Status"}), use_container_width=True)
    st.info("Interpretation: use these charts to investigate patterns. Association does not prove that a feature causes churn.")

elif page == "Customer Risk Predictor":
    st.title("Individual Customer Risk Predictor")
    st.write("Enter a customer profile to estimate churn probability. This is a model estimate, not a guarantee.")
    with st.form("predict_form"):
        c1,c2,c3 = st.columns(3)
        with c1:
            tenure_v = st.slider("Tenure (months)", 1, 72, 12)
            monthly_v = st.slider("Monthly charges (₹)", 18.0, 120.0, 75.0, 1.0)
            contract_v = st.selectbox("Contract type", ["Month-to-month","One year","Two year"])
        with c2:
            internet_v = st.selectbox("Internet service", ["Fiber optic","DSL","No"])
            payment_v = st.selectbox("Payment method", ["Electronic check","Mailed check","Bank transfer","Credit card"])
            support_v = st.selectbox("Tech support", ["No","Yes"])
        with c3:
            paperless_v = st.selectbox("Paperless billing", ["Yes","No"])
            senior_v = st.selectbox("Senior citizen", [0,1], format_func=lambda x:"Yes" if x else "No")
            dependents_v = st.selectbox("Has dependents", ["No","Yes"])
        submitted = st.form_submit_button("Estimate churn risk", type="primary")
    if submitted:
        row = pd.DataFrame([{"tenure_months":tenure_v,"monthly_charges":monthly_v,"contract_type":contract_v,"internet_service":internet_v,
                             "payment_method":payment_v,"tech_support":support_v,"paperless_billing":paperless_v,"senior_citizen":senior_v,"dependents":dependents_v}])
        probability = float(models["Random Forest"].predict_proba(row[FEATURES])[0,1])
        risk = "High" if probability >= .65 else ("Medium" if probability >= .35 else "Low")
        col1,col2 = st.columns([1,2])
        col1.metric("Estimated churn probability", f"{probability*100:.1f}%")
        col1.subheader(f"Risk level: {risk}")
        col2.progress(probability)
        if risk == "High":
            st.warning("Suggested follow-up: review service issues, offer a plan review, and contact the customer with a relevant retention option.")
        elif risk == "Medium":
            st.info("Suggested follow-up: monitor usage and satisfaction, and consider a proactive service check-in.")
        else:
            st.success("Suggested follow-up: continue regular service and satisfaction monitoring.")
        st.caption("Risk bands are illustrative thresholds for this academic demo, not validated business policy.")

elif page == "Model Performance":
    st.title("Machine Learning Model Evaluation")
    st.write("Both models use the same stratified train/test split (80% training, 20% testing).")
    display = metrics.copy()
    st.dataframe(display.style.format({c:"{:.3f}" for c in ["Accuracy","Precision","Recall","F1-score","ROC-AUC"]}), use_container_width=True)
    st.plotly_chart(px.bar(display.melt(id_vars="Model",var_name="Metric",value_name="Score"), x="Metric", y="Score", color="Model", barmode="group", title="Model comparison"), use_container_width=True)
    chosen = st.selectbox("Inspect confusion matrix for", list(models.keys()))
    from sklearn.metrics import ConfusionMatrixDisplay
    pred = models[chosen].predict(X_test)
    cm = confusion_matrix(y_test,pred)
    cm_df = pd.DataFrame(cm, index=["Actual stayed","Actual churned"], columns=["Predicted stayed","Predicted churned"])
    st.markdown(f"#### Confusion matrix — {chosen}")
    st.dataframe(cm_df, use_container_width=True)
    st.caption("Recall is important because a false negative means a customer who churns was not flagged. Precision matters because unnecessary retention offers can cost money.")

elif page == "Retention Insights":
    st.title("Retention Insights")
    st.write("Explore customer groups and possible retention actions. These are suggestions for review, not automated decisions.")
    contract = data.groupby("contract_type", observed=True).agg(customers=("churn","size"), churn_rate=("churn","mean")).reset_index()
    contract["churn_rate"] *= 100
    st.plotly_chart(px.bar(contract, x="contract_type", y="churn_rate", text=contract["churn_rate"].round(1), title="Churn rate by contract (%)", labels={"churn_rate":"Churn rate (%)","contract_type":"Contract"}), use_container_width=True)
    tenure_bins = pd.cut(data["tenure_months"], bins=[0,12,24,48,72], labels=["1–12","13–24","25–48","49–72"])
    tenure = data.assign(tenure_group=tenure_bins).groupby("tenure_group", observed=True)["churn"].mean().mul(100).reset_index(name="Churn rate (%)")
    st.plotly_chart(px.bar(tenure, x="tenure_group", y="Churn rate (%)", title="Churn by customer tenure", labels={"tenure_group":"Tenure (months)"}), use_container_width=True)
    st.markdown("### Suggested retention actions")
    st.markdown("- **New / short-tenure customers:** provide onboarding and check-in support.\n- **Month-to-month customers:** consider explaining longer-term plans and their benefits.\n- **Customers with support needs:** resolve service issues before offering discounts.\n- **High-charge customers:** review plan fit and transparent pricing.")
    st.caption("Recommendations are general examples; validate them against real customer research before using them commercially.")

elif page == "About Project":
    st.title("About the Project")
    st.markdown("""
**Project title:** Customer Churn Prediction and Retention Analysis Using Machine Learning

**Problem:** Businesses lose revenue when customers discontinue services. Understanding churn patterns can help prioritize retention efforts.

**Workflow:** Dataset → Data cleaning / preprocessing → Exploratory data analysis → Model training → Evaluation → Risk prediction → Retention insights.

**Algorithms:** Logistic Regression and Random Forest.

**Libraries:** Python, Pandas, NumPy, Scikit-learn, Plotly, Streamlit.

**Important limitation:** The included dataset is synthetic demo data generated for software testing and classroom demonstration. It is not real telecom customer data, and the scores should not be presented as real-world performance. For final academic evaluation, replace it with a documented public dataset and retrain the models.
""")
    st.success("The app trains the models on startup, so no separate model download is required.")
