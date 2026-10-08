
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

st.set_page_config(
    page_title="Streamline | Bank Churn Risk",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Styling ----------
st.markdown("""
<style>
.block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
.metric-card {
    border: 1px solid rgba(128,128,128,.25);
    border-radius: 14px;
    padding: 18px;
    background: rgba(128,128,128,.05);
}
.risk-box {
    border-radius: 16px;
    padding: 22px;
    text-align: center;
    border: 1px solid rgba(128,128,128,.25);
}
.small-muted {color: #777; font-size: .9rem;}

<style>
.block-container {padding: 1.4rem 2rem 2.5rem;}
[data-testid="stSidebar"] {border-right: 1px solid rgba(128,128,128,.18);}
[data-testid="stMetric"] {
    border: 1px solid rgba(128,128,128,.20);
    border-radius: 14px;
    padding: 12px 14px;
    background: rgba(128,128,128,.035);
}
.metric-card, .risk-box {
    border: 1px solid rgba(128,128,128,.20);
    border-radius: 16px;
    padding: 20px;
    background: rgba(128,128,128,.035);
}
.risk-box h2 {margin-bottom: .35rem;}
.small-muted {color: #777; font-size: .9rem;}
.hero {
    border: 1px solid rgba(128,128,128,.20);
    border-radius: 20px;
    padding: 28px;
    margin-bottom: 20px;
    background: linear-gradient(135deg, rgba(128,128,128,.08), rgba(128,128,128,.025));
}
.hero h1 {margin: 0 0 6px 0; font-size: 2.25rem;}
.hero p {margin: 0; color: #777;}
.section-title {margin-top: 12px; margin-bottom: 8px;}
div.stButton > button {border-radius: 10px;}
</style>

""", unsafe_allow_html=True)

# ---------- Data / model ----------
@st.cache_data
def load_data():
    return pd.read_csv("European_Bank.csv")

@st.cache_resource
def train_models(data):
    # CustomerId, Surname and Year are excluded from prediction.
    features = [
        "CreditScore", "Geography", "Gender", "Age", "Tenure",
        "Balance", "NumOfProducts", "HasCrCard",
        "IsActiveMember", "EstimatedSalary"
    ]
    X = data[features]
    y = data["Exited"].astype(int)

    categorical = ["Geography", "Gender"]
    numeric = [
        "CreditScore", "Age", "Tenure", "Balance",
        "NumOfProducts", "HasCrCard", "IsActiveMember",
        "EstimatedSalary"
    ]

    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler())
        ]), numeric),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    models = {
        "Logistic Regression": Pipeline([
            ("prep", preprocessor),
            ("model", LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42))
        ]),
        "Random Forest": Pipeline([
            ("prep", preprocessor),
            ("model", RandomForestClassifier(
                n_estimators=350,
                max_depth=10,
                min_samples_leaf=3,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            ))
        ])
    }

    metrics = {}
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        prob = pipe.predict_proba(X_test)[:, 1]
        metrics[name] = {
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1": f1_score(y_test, pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, prob),
        }

    # Random Forest is the primary dashboard model because it provides
    # direct feature importances and captures non-linear relationships.
    primary = models["Random Forest"]

    return primary, models, metrics, X_test, y_test

data = load_data()
model, all_models, metrics, X_test, y_test = train_models(data)

# ---------- Helpers ----------
FEATURES = [
    "CreditScore", "Geography", "Gender", "Age", "Tenure",
    "Balance", "NumOfProducts", "HasCrCard",
    "IsActiveMember", "EstimatedSalary"
]

def risk_label(prob):
    if prob < 0.30:
        return "LOW"
    if prob < 0.60:
        return "MEDIUM"
    return "HIGH"

def risk_message(prob):
    level = risk_label(prob)
    if level == "LOW":
        return "Customer is currently predicted to have a relatively low churn risk."
    if level == "MEDIUM":
        return "Customer shows a moderate churn signal and may benefit from proactive engagement."
    return "Customer shows a high churn signal and may warrant targeted retention action."

def predict_one(values):
    row = pd.DataFrame([values])
    return float(model.predict_proba(row)[0, 1])

# ---------- Sidebar ----------
st.sidebar.title("🏦 Streamline")
st.sidebar.caption("Bank Customer Churn Risk Platform")
st.sidebar.divider()
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Risk Calculator", "Customer Analytics",
     "Feature Importance", "What-if Simulator", "Model Performance"]
)

st.sidebar.divider()
st.sidebar.caption("Predictive target: Exited")
st.sidebar.caption("Primary model: Random Forest")

# ---------- Overview ----------
if page == "Overview":
    st.markdown("""
    <div class="hero">
        <h1>Streamline</h1>
        <p>Bank Customer Churn Risk Intelligence</p>
        <p class="small-muted">Predict. Understand. Retain.</p>
    </div>
    """, unsafe_allow_html=True)
    st.write(
        "A predictive analytics interface for estimating customer churn risk "
        "and exploring the factors associated with customer attrition."
    )

    total = len(data)
    churned = int(data["Exited"].sum())
    churn_rate = churned / total

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers", f"{total:,}")
    c2.metric("Exited Customers", f"{churned:,}")
    c3.metric("Observed Churn Rate", f"{churn_rate:.1%}")
    c4.metric("Model ROC-AUC", f"{metrics['Random Forest']['ROC-AUC']:.3f}")

    st.divider()

    left, right = st.columns(2)

    with left:
        st.markdown("### Churn Distribution")
        counts = data["Exited"].value_counts().sort_index()
        fig, ax = plt.subplots()
        ax.bar(["Stayed", "Exited"], [counts.get(0, 0), counts.get(1, 0)])
        ax.set_ylabel("Customers")
        ax.set_title("Observed Customer Churn")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with right:
        st.markdown("### Churn by Geography")
        geo = data.groupby("Geography")["Exited"].mean().sort_values(ascending=False)
        fig, ax = plt.subplots()
        ax.bar(geo.index, geo.values * 100)
        ax.set_ylabel("Churn rate (%)")
        ax.set_title("Observed Churn Rate by Geography")
        ax.tick_params(axis="x", rotation=0)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.info(
        "Use the Risk Calculator to score an individual customer, or the "
        "What-if Simulator to see how changing customer characteristics affects "
        "the model's predicted churn probability."
    )

# ---------- Risk Calculator ----------
elif page == "Risk Calculator":
    st.title("🎯 Customer Churn Risk Calculator")
    st.write("Enter customer characteristics to generate an individual churn-risk estimate.")

    with st.form("risk_form"):
        a, b, c = st.columns(3)

        with a:
            credit = st.number_input("Credit Score", 300, 900, 650, 1)
            geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
            gender = st.selectbox("Gender", ["Female", "Male"])
            age = st.number_input("Age", 18, 100, 40, 1)

        with b:
            tenure = st.number_input("Tenure (years)", 0, 10, 5, 1)
            balance = st.number_input("Balance", 0.0, 300000.0, 75000.0, 500.0)
            products = st.number_input("Number of Products", 1, 4, 1, 1)
            card = st.selectbox("Has Credit Card", ["Yes", "No"])

        with c:
            active = st.selectbox("Is Active Member", ["Yes", "No"])
            salary = st.number_input("Estimated Salary", 0.0, 250000.0, 100000.0, 500.0)

        submitted = st.form_submit_button("Calculate Churn Risk", use_container_width=True)

    if submitted:
        values = {
            "CreditScore": credit,
            "Geography": geography,
            "Gender": gender,
            "Age": age,
            "Tenure": tenure,
            "Balance": balance,
            "NumOfProducts": products,
            "HasCrCard": 1 if card == "Yes" else 0,
            "IsActiveMember": 1 if active == "Yes" else 0,
            "EstimatedSalary": salary
        }

        prob = predict_one(values)
        level = risk_label(prob)

        st.divider()
        st.markdown("### Prediction")

        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Predicted Churn Probability", f"{prob:.1%}")
        with col2:
            st.markdown(
                f'<div class="risk-box"><h2>{level} RISK</h2>'
                f'<p>{risk_message(prob)}</p></div>',
                unsafe_allow_html=True
            )

        st.progress(min(max(prob, 0.0), 1.0))

        st.caption(
            "This is a model prediction, not a guarantee of customer behavior. "
            "Risk thresholds are dashboard decision bands and can be adjusted."
        )

# ---------- Customer Analytics ----------
elif page == "Customer Analytics":
    st.title("📊 Customer Analytics")

    metric = st.selectbox(
        "Analyze churn by",
        ["Age", "CreditScore", "Balance", "Tenure", "NumOfProducts",
         "EstimatedSalary", "IsActiveMember", "Geography", "Gender"]
    )

    if metric in ["Geography", "Gender", "IsActiveMember"]:
        if metric == "IsActiveMember":
            grp = data.groupby(metric)["Exited"].agg(["count", "mean"])
            labels = ["Inactive", "Active"]
            xvals = [0, 1]
            vals = [grp.loc[x, "mean"] * 100 if x in grp.index else 0 for x in xvals]
        else:
            grp = data.groupby(metric)["Exited"].agg(["count", "mean"]).sort_values("mean", ascending=False)
            labels = grp.index.astype(str).tolist()
            vals = (grp["mean"] * 100).tolist()

        fig, ax = plt.subplots()
        ax.bar(labels, vals)
        ax.set_ylabel("Churn rate (%)")
        ax.set_title(f"Churn Rate by {metric}")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        summary = grp.copy()
        summary["churn_rate"] = summary["mean"]
        summary = summary.drop(columns=["mean"])
        st.dataframe(summary, use_container_width=True)

    else:
        bins = st.slider("Number of bins", 5, 30, 10)
        temp = data[[metric, "Exited"]].copy()
        temp["bin"] = pd.cut(temp[metric], bins=bins)
        grouped = temp.groupby("bin", observed=False)["Exited"].agg(["count", "mean"]).reset_index()

        fig, ax = plt.subplots()
        ax.plot(grouped["bin"].astype(str), grouped["mean"] * 100, marker="o")
        ax.set_ylabel("Churn rate (%)")
        ax.set_xlabel(metric)
        ax.set_title(f"Churn Rate Across {metric}")
        ax.tick_params(axis="x", rotation=60)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.dataframe(grouped.rename(columns={"mean": "churn_rate"}), use_container_width=True)

# ---------- Feature Importance ----------
elif page == "Feature Importance":
    st.title("🎯 Feature Importance")
    st.write(
        "Random Forest feature importance shows how much each encoded input "
        "feature contributes to the model's predictive decisions."
    )

    prep = model.named_steps["prep"]
    rf = model.named_steps["model"]
    names = prep.get_feature_names_out()
    importances = rf.feature_importances_

    imp = pd.DataFrame({"Feature": names, "Importance": importances})
    imp["Feature"] = imp["Feature"].str.replace(r"^(num|cat)__", "", regex=True)
    imp = imp.sort_values("Importance", ascending=False).head(15)

    fig, ax = plt.subplots()
    ax.barh(imp["Feature"].iloc[::-1], imp["Importance"].iloc[::-1])
    ax.set_xlabel("Importance")
    ax.set_title("Top Predictive Features")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.dataframe(imp.reset_index(drop=True), use_container_width=True)

    st.warning(
        "Feature importance indicates predictive contribution, not causation. "
        "It should not be interpreted as proof that changing a feature will cause churn to change."
    )

# ---------- What-if Simulator ----------
elif page == "What-if Simulator":
    st.title("🔄 What-if Scenario Simulator")
    st.write(
        "Set a baseline customer, then modify one or more characteristics to "
        "see how the predicted churn probability changes."
    )

    st.markdown("### Baseline customer")
    a, b, c = st.columns(3)

    with a:
        base_credit = st.slider("Credit Score", 300, 900, 650)
        base_geo = st.selectbox("Geography", ["France", "Germany", "Spain"], key="base_geo")
        base_gender = st.selectbox("Gender", ["Female", "Male"], key="base_gender")
        base_age = st.slider("Age", 18, 100, 40)

    with b:
        base_tenure = st.slider("Tenure", 0, 10, 5)
        base_balance = st.slider("Balance", 0.0, 300000.0, 75000.0, 1000.0)
        base_products = st.slider("Products", 1, 4, 1)
        base_card = st.selectbox("Credit Card", ["Yes", "No"], key="base_card")

    with c:
        base_active = st.selectbox("Active Member", ["Yes", "No"], key="base_active")
        base_salary = st.slider("Estimated Salary", 0.0, 250000.0, 100000.0, 1000.0)

    baseline = {
        "CreditScore": base_credit, "Geography": base_geo, "Gender": base_gender,
        "Age": base_age, "Tenure": base_tenure, "Balance": base_balance,
        "NumOfProducts": base_products, "HasCrCard": 1 if base_card == "Yes" else 0,
        "IsActiveMember": 1 if base_active == "Yes" else 0,
        "EstimatedSalary": base_salary
    }

    baseline_prob = predict_one(baseline)

    st.divider()
    st.markdown("### Scenario adjustments")

    x, y, z = st.columns(3)
    with x:
        new_active = st.selectbox("Scenario: Active Member", ["Keep baseline", "Yes", "No"])
        new_products = st.selectbox("Scenario: Products", ["Keep baseline", 1, 2, 3, 4])
    with y:
        new_geo = st.selectbox("Scenario: Geography", ["Keep baseline", "France", "Germany", "Spain"])
        new_age = st.number_input("Scenario: Age", 18, 100, base_age)
    with z:
        new_credit = st.number_input("Scenario: Credit Score", 300, 900, base_credit)
        new_balance = st.number_input("Scenario: Balance", 0.0, 300000.0, float(base_balance), 1000.0)

    scenario = baseline.copy()
    if new_active != "Keep baseline":
        scenario["IsActiveMember"] = 1 if new_active == "Yes" else 0
    if new_products != "Keep baseline":
        scenario["NumOfProducts"] = int(new_products)
    if new_geo != "Keep baseline":
        scenario["Geography"] = new_geo
    scenario["Age"] = new_age
    scenario["CreditScore"] = new_credit
    scenario["Balance"] = new_balance

    scenario_prob = predict_one(scenario)
    delta = scenario_prob - baseline_prob

    c1, c2, c3 = st.columns(3)
    c1.metric("Baseline Risk", f"{baseline_prob:.1%}")
    c2.metric("Scenario Risk", f"{scenario_prob:.1%}", f"{delta:+.1%}")
    c3.metric("Scenario Band", risk_label(scenario_prob))

    if delta < 0:
        st.success(f"The scenario lowers predicted risk by {abs(delta):.1%}.")
    elif delta > 0:
        st.warning(f"The scenario raises predicted risk by {delta:.1%}.")
    else:
        st.info("The scenario produces the same predicted risk as the baseline.")

    st.caption(
        "What-if results are model-based sensitivity estimates. They do not establish "
        "that changing a customer attribute will causally change their behavior."
    )

# ---------- Model Performance ----------
else:
    st.title("📈 Model Performance")
    st.write("Performance on a stratified 20% holdout test set.")

    rows = []
    for name, vals in metrics.items():
        rows.append({"Model": name, **vals})
    results = pd.DataFrame(rows).set_index("Model")

    st.dataframe(
        results.style.format("{:.3f}"),
        use_container_width=True
    )

    best = results["ROC-AUC"].idxmax()
    st.success(f"Highest ROC-AUC in this comparison: **{best}**")

    st.markdown("### Interpretation")
    st.write(
        "ROC-AUC measures how well the model separates customers who exited from "
        "those who stayed across probability thresholds. Precision, recall and F1 "
        "provide additional views of classification performance."
    )

