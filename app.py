import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="European Banking Customer Churn Dashboard",
    page_icon="🏦",
    layout="wide"
)

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("European_Bank.csv")
    return df

df = load_data()

# ---------------------------------------------------------
# CLEAN COLUMN NAMES
# ---------------------------------------------------------

df.columns = df.columns.str.strip()
st.write("ACTUAL CSV COLUMNS:", df.columns.tolist())

# ---------------------------------------------------------
# TITLE
# ---------------------------------------------------------

st.title("🏦 European Banking Customer Churn Analytics Dashboard")
st.markdown(
    "Interactive analysis and prediction of customer churn "
    "across European banking customers."
)

st.divider()

# ---------------------------------------------------------
# SIDEBAR FILTERS
# ---------------------------------------------------------

st.sidebar.header("🔎 Dashboard Filters")

geographies = st.sidebar.multiselect(
    "Select Geography",
    options=sorted(df["Geography"].dropna().unique()),
    default=sorted(df["Geography"].dropna().unique())
)

genders = st.sidebar.multiselect(
    "Select Gender",
    options=sorted(df["Gender"].dropna().unique()),
    default=sorted(df["Gender"].dropna().unique())
)

filtered_df = df[
    df["Geography"].isin(geographies)
    & df["Gender"].isin(genders)
]

# ---------------------------------------------------------
# KPI SECTION
# ---------------------------------------------------------

total_customers = len(filtered_df)

churned_customers = filtered_df["Exited"].sum()

churn_rate = (
    churned_customers / total_customers * 100
    if total_customers > 0 else 0
)

average_balance = filtered_df["Balance"].mean()
average_salary = filtered_df["Estimated Salary"].mean()

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "👥 Total Customers",
    f"{total_customers:,}"
)

col2.metric(
    "🚪 Churned Customers",
    f"{int(churned_customers):,}"
)

col3.metric(
    "📉 Churn Rate",
    f"{churn_rate:.2f}%"
)

col4.metric(
    "💰 Average Balance",
    f"${average_balance:,.0f}"
)

st.divider()

# ---------------------------------------------------------
# OVERALL CHURN ANALYSIS
# ---------------------------------------------------------

st.header("📊 Overall Churn Analysis")

col1, col2 = st.columns(2)

with col1:

    churn_counts = filtered_df["Exited"].value_counts()

    labels = ["Stayed", "Churned"]

    values = [
        churn_counts.get(0, 0),
        churn_counts.get(1, 0)
    ]

    fig, ax = plt.subplots()

    ax.bar(labels, values)

    ax.set_title("Customer Churn Distribution")
    ax.set_ylabel("Number of Customers")

    st.pyplot(fig)

with col2:

    geography_churn = (
        filtered_df.groupby("Geography")["Exited"]
        .mean()
        .mul(100)
        .reset_index()
    )

    fig, ax = plt.subplots()

    ax.bar(
        geography_churn["Geography"],
        geography_churn["Exited"]
    )

    ax.set_title("Churn Rate by Geography")
    ax.set_ylabel("Churn Rate (%)")

    st.pyplot(fig)

st.divider()

# ---------------------------------------------------------
# AGE & TENURE ANALYSIS
# ---------------------------------------------------------

st.header("👤 Age & Tenure Analysis")

col1, col2 = st.columns(2)

with col1:

    fig, ax = plt.subplots()

    ax.hist(
        filtered_df[filtered_df["Exited"] == 0]["Age"],
        bins=20,
        alpha=0.6,
        label="Stayed"
    )

    ax.hist(
        filtered_df[filtered_df["Exited"] == 1]["Age"],
        bins=20,
        alpha=0.6,
        label="Churned"
    )

    ax.set_title("Age Distribution: Churned vs Stayed")
    ax.set_xlabel("Age")
    ax.set_ylabel("Customers")
    ax.legend()

    st.pyplot(fig)

with col2:

    tenure_churn = (
        filtered_df.groupby("Tenure")["Exited"]
        .mean()
        .mul(100)
        .reset_index()
    )

    fig, ax = plt.subplots()

    ax.plot(
        tenure_churn["Tenure"],
        tenure_churn["Exited"],
        marker="o"
    )

    ax.set_title("Churn Rate by Customer Tenure")
    ax.set_xlabel("Tenure")
    ax.set_ylabel("Churn Rate (%)")

    st.pyplot(fig)

st.divider()

# ---------------------------------------------------------
# PRODUCT / ENGAGEMENT ANALYSIS
# ---------------------------------------------------------

st.header("💳 Product & Engagement Analysis")

col1, col2 = st.columns(2)

with col1:

    product_churn = (
        filtered_df.groupby("Number Of Products")["Exited"]
        .mean()
        .mul(100)
        .reset_index()
    )

    fig, ax = plt.subplots()

    ax.bar(
        product_churn["Number Of Products"].astype(str),
        product_churn["Exited"]
    )

    ax.set_title("Churn Rate by Number of Products")
    ax.set_xlabel("Number of Products")
    ax.set_ylabel("Churn Rate (%)")

    st.pyplot(fig)

with col2:

    active_churn = (
        filtered_df.groupby("Is Active Member")["Exited"]
        .mean()
        .mul(100)
        .reset_index()
    )

    labels = ["Inactive", "Active"]

    values = [
        active_churn.loc[
            active_churn["Is Active Member"] == 0,
            "Exited"
        ].values[0]
        if 0 in active_churn["Is Active Member"].values else 0,

        active_churn.loc[
            active_churn["Is Active Member"] == 1,
            "Exited"
        ].values[0]
        if 1 in active_churn["Is Active Member"].values else 0
    ]

    fig, ax = plt.subplots()

    ax.bar(labels, values)

    ax.set_title("Churn Rate: Active vs Inactive Members")
    ax.set_ylabel("Churn Rate (%)")

    st.pyplot(fig)

st.divider()

# ---------------------------------------------------------
# HIGH VALUE CUSTOMER EXPLORER
# ---------------------------------------------------------

st.header("💎 High-Value Customer Churn Explorer")

balance_threshold = st.slider(
    "Minimum Customer Balance",
    min_value=float(df["Balance"].min()),
    max_value=float(df["Balance"].max()),
    value=float(df["Balance"].quantile(0.75))
)

high_value = filtered_df[
    filtered_df["Balance"] >= balance_threshold
]

st.write(
    f"Customers with balance ≥ ${balance_threshold:,.0f}"
)

st.dataframe(
    high_value[
        [
            "Customer ID",
            "Surname",
            "Geography",
            "Age",
            "Balance",
            "Number Of Products",
            "Is Active Member",
            "Exited"
        ]
    ].sort_values(
        "Balance",
        ascending=False
    ),
    use_container_width=True
)

st.divider()

# ---------------------------------------------------------
# MACHINE LEARNING MODEL
# ---------------------------------------------------------

st.header("🤖 Customer Churn Prediction Model")

model_features = [
    "Credit Score",
    "Geography",
    "Gender",
    "Age",
    "Tenure",
    "Balance",
    "Number Of Products",
    "Has Cr Card",
    "Is Active Member",
    "Estimated Salary"
]

X = df[model_features]
y = df["Exited"]

categorical_features = [
    "Geography",
    "Gender"
]

numeric_features = [
    "Credit Score",
    "Age",
    "Tenure",
    "Balance",
    "Number Of Products",
    "Has Cr Card",
    "Is Active Member",
    "Estimated Salary"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numeric",
            "passthrough",
            numeric_features
        )
    ]
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=150,
                random_state=42,
                class_weight="balanced"
            )
        )
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

st.metric(
    "Model Accuracy",
    f"{accuracy * 100:.2f}%"
)

st.divider()

# ---------------------------------------------------------
# CHURN RISK CALCULATOR
# ---------------------------------------------------------

st.header("🧮 Customer Churn Risk Calculator")

col1, col2, col3 = st.columns(3)

with col1:

    input_credit = st.number_input(
        "Credit Score",
        min_value=300,
        max_value=900,
        value=650
    )

    input_age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=40
    )

    input_tenure = st.number_input(
        "Tenure",
        min_value=0,
        max_value=20,
        value=5
    )

    input_balance = st.number_input(
        "Balance",
        min_value=0.0,
        value=50000.0
    )

with col2:

    input_products = st.number_input(
        "Number of Products",
        min_value=1,
        max_value=4,
        value=2
    )

    input_card = st.selectbox(
        "Has Credit Card?",
        ["Yes", "No"]
    )

    input_active = st.selectbox(
        "Active Member?",
        ["Yes", "No"]
    )

    input_salary = st.number_input(
        "Estimated Salary",
        min_value=0.0,
        value=50000.0
    )

with col3:

    input_geo = st.selectbox(
        "Geography",
        sorted(df["Geography"].dropna().unique())
    )

    input_gender = st.selectbox(
        "Gender",
        sorted(df["Gender"].dropna().unique())
    )

    st.write("")
    st.write("")
    st.write("")

    predict_button = st.button(
        "🔮 Calculate Churn Risk",
        type="primary"
    )

if predict_button:

    customer_input = pd.DataFrame(
        {
            "Credit Score": [input_credit],
            "Geography": [input_geo],
            "Gender": [input_gender],
            "Age": [input_age],
            "Tenure": [input_tenure],
            "Balance": [input_balance],
            "Number Of Products": [input_products],
            "Has Cr Card": [
                1 if input_card == "Yes" else 0
            ],
            "Is Active Member": [
                1 if input_active == "Yes" else 0
            ],
            "Estimated Salary": [input_salary]
        }
    )

    probability = model.predict_proba(
        customer_input
    )[0][1]

    st.subheader(
        f"Estimated Churn Probability: {probability * 100:.2f}%"
    )

    if probability >= 0.70:
        st.error(
            "🔴 High Churn Risk — immediate retention action recommended."
        )

    elif probability >= 0.40:
        st.warning(
            "🟠 Medium Churn Risk — customer should be monitored."
        )

    else:
        st.success(
            "🟢 Low Churn Risk — customer currently shows lower churn probability."
        )

    st.progress(float(probability))

st.divider()

# ---------------------------------------------------------
# PROBABILITY DISTRIBUTION
# ---------------------------------------------------------

st.header("📈 Churn Probability Distribution")

all_probabilities = model.predict_proba(
    df[model_features]
)[:, 1]

fig, ax = plt.subplots()

ax.hist(
    all_probabilities,
    bins=20
)

ax.set_title(
    "Distribution of Predicted Customer Churn Probabilities"
)

ax.set_xlabel("Predicted Churn Probability")
ax.set_ylabel("Number of Customers")

st.pyplot(fig)

st.divider()

# ---------------------------------------------------------
# FEATURE IMPORTANCE
# ---------------------------------------------------------

st.header("🎯 Feature Importance Dashboard")

rf_model = model.named_steps["classifier"]
preprocessor_model = model.named_steps["preprocessor"]

feature_names = (
    preprocessor_model
    .get_feature_names_out()
)

importance = rf_model.feature_importances_

importance_df = pd.DataFrame(
    {
        "Feature": feature_names,
        "Importance": importance
    }
).sort_values(
    "Importance",
    ascending=False
).head(15)

fig, ax = plt.subplots()

ax.barh(
    importance_df["Feature"][::-1],
    importance_df["Importance"][::-1]
)

ax.set_title(
    "Top Factors Influencing Churn Prediction"
)

ax.set_xlabel("Importance")

st.pyplot(fig)

st.divider()

# ---------------------------------------------------------
# WHAT-IF SCENARIO SIMULATOR
# ---------------------------------------------------------

st.header("🔄 What-If Scenario Simulator")

st.write(
    "Adjust customer characteristics to see how the predicted "
    "churn probability changes."
)

scenario_age = st.slider(
    "Customer Age",
    18,
    100,
    40
)

scenario_balance = st.slider(
    "Customer Balance",
    0.0,
    float(max(df["Balance"].max(), 100000)),
    50000.0
)

scenario_products = st.slider(
    "Number of Products",
    1,
    4,
    2
)

scenario_active = st.selectbox(
    "Active Member",
    ["Yes", "No"],
    key="scenario_active"
)

scenario_customer = pd.DataFrame(
    {
        "Credit Score": [650],
        "Geography": [df["Geography"].mode()[0]],
        "Gender": [df["Gender"].mode()[0]],
        "Age": [scenario_age],
        "Tenure": [5],
        "Balance": [scenario_balance],
        "Number Of Products": [scenario_products],
        "Has Cr Card": [1],
        "Is Active Member": [
            1 if scenario_active == "Yes" else 0
        ],
        "Estimated Salary": [
            df["Estimated Salary"].median()
        ]
    }
)

scenario_probability = model.predict_proba(
    scenario_customer
)[0][1]

st.metric(
    "Scenario Churn Probability",
    f"{scenario_probability * 100:.2f}%"
)

st.progress(float(scenario_probability))

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.divider()

st.caption(
    "European Banking Customer Churn Analytics | "
    "Internship Project Dashboard"
)
