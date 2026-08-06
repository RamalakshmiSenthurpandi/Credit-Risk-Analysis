import streamlit as st
import pandas as pd
import joblib
import os

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(
    page_title="Credit Risk Analysis",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Credit Risk Analysis")
st.write("Predict whether a customer is likely to be **Good Debt** or **Bad Debt**.")

# ---------------------------
# Upload Dataset
# ---------------------------
uploaded_file = st.file_uploader(
    "Upload Customer Dataset (Excel)",
    type=["xlsx"]
)

if uploaded_file is None:
    st.info("Please upload the dataset to continue.")
    st.stop()

# Read dataset
df = pd.read_excel(uploaded_file)

st.success("Dataset uploaded successfully!")

st.subheader("Dataset Preview")
st.dataframe(df.head())

# ---------------------------
# Detect Customer ID
# ---------------------------
cust_col = next(
    (c for c in df.columns if "customer" in c.lower()),
    None
)

if cust_col is None:
    st.error("Customer ID column not found.")
    st.stop()

# ---------------------------
# Load Model
# ---------------------------
MODEL_PATH = "models/model.pkl"
COL_PATH = "models/columns.pkl"

if not os.path.exists(MODEL_PATH):
    st.warning("Model not found.")
    st.info("Train the model first using the notebook.")
    st.stop()

model = joblib.load(MODEL_PATH)
columns = joblib.load(COL_PATH)

# ---------------------------
# Customer Selection
# ---------------------------
customer_id = st.selectbox(
    "Select Customer ID",
    df[cust_col].astype(str).unique()
)

row = df[df[cust_col].astype(str) == customer_id]

if st.button("Predict"):

    X = row[columns]

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

    st.write("---")

    st.subheader("Customer Details")

    st.dataframe(row)
