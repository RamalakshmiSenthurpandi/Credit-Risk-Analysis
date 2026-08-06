import streamlit as st
import pandas as pd
import joblib
import os

# -------------------------------
# PAGE CONFIGURATION
# -------------------------------
st.set_page_config(
    page_title="Credit Risk Analysis",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Credit Risk Analysis")
st.write("Predict whether a customer is **Good Debt** or **Bad Debt**.")

# -------------------------------
# LOAD MODEL
# -------------------------------
MODEL_PATH = "models/model.pkl"
COLUMNS_PATH = "models/columns.pkl"

if not os.path.exists(MODEL_PATH):
    st.error("model.pkl not found.")
    st.stop()

if not os.path.exists(COLUMNS_PATH):
    st.error("columns.pkl not found.")
    st.stop()

model = joblib.load(MODEL_PATH)
model_columns = joblib.load(COLUMNS_PATH)

# -------------------------------
# FILE UPLOAD
# -------------------------------
uploaded_file = st.file_uploader(
    "Upload Excel Dataset",
    type=["xlsx"]
)

if uploaded_file is None:
    st.info("Please upload your dataset.")
    st.stop()

df = pd.read_excel(uploaded_file)

# Remove spaces in column names
df.columns = df.columns.str.strip()

st.success("Dataset uploaded successfully.")

# -------------------------------
# REMOVE EXTRA COLUMNS
# -------------------------------
df = df.drop(columns=["TARGET", "OUTCOME"], errors="ignore")

# -------------------------------
# NUMERIC COLUMNS
# -------------------------------
numeric_cols = df.select_dtypes(include="number").columns.tolist()

if "Customer" in numeric_cols:
    numeric_cols.remove("Customer")

agg = {c: "mean" for c in numeric_cols}

# -------------------------------
# CUSTOMER LEVEL DATA
# -------------------------------
df_cust = df.groupby("Customer", as_index=False).agg(agg)

customer_list = sorted(df_cust["Customer"].astype(str).unique())

customer_id = st.selectbox(
    "Select Customer ID",
    customer_list
)

# -------------------------------
# PREDICT
# -------------------------------
if st.button("Predict"):

    row = df_cust[df_cust["Customer"].astype(str) == customer_id]

    X = row.drop(columns=["Customer"], errors="ignore")

    # Keep only model features
    X = X[model_columns]

    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("🟢 GOOD DEBT")
    else:
        st.error("🔴 BAD DEBT")

    st.metric(
        "Probability of Good Debt",
        f"{probability*100:.2f}%"
    )

    st.progress(float(probability))

    st.subheader("Customer Details")

    st.dataframe(row)
