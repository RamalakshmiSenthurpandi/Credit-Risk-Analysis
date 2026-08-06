import streamlit as st
import pandas as pd
import joblib
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import os

st.set_page_config(page_title="Bad Debt Prediction", layout="wide")
st.title("💳 Customer Good / Bad Debt Prediction")

# ---------------- FIXED COLAB PATH ----------------
FILE_PATH = "/content/drive/MyDrive/Bad_debt_prediction/Merged_data.xlsx"

st.write("Using file path:")
st.code(FILE_PATH)

# ---------------- LOAD DATA ----------------
if not os.path.exists(FILE_PATH):
    st.error("❌ File not found. Make sure Google Drive is mounted.")
    st.stop()

df = pd.read_excel(FILE_PATH)
st.success("File loaded successfully from Google Drive")
st.write("Raw data preview", df.head())

# -------- Detect Customer ID --------
cust_col = next((c for c in df.columns if "customer" in c.lower()), None)
if cust_col is None:
    st.error("Customer ID column not found")
    st.stop()

# -------- Detect Target --------
target_col = next(
    (c for c in df.columns if c.lower() in ["target", "good_bad", "debt_flag", "outcome"]),
    None
)
if target_col is None:
    st.error("Target column (good/bad debt) not found")
    st.stop()

# -------- Leakage Protection --------
leakage_cols = [c for c in df.columns if "status" in c.lower() or "result" in c.lower()]
df.drop(columns=leakage_cols, inplace=True, errors="ignore")

# -------- Customer-level Aggregation --------
agg = {c: "mean" for c in df.columns if c not in [cust_col, target_col]}
df_cust = df.groupby(cust_col).agg(agg)
df_cust[target_col] = df.groupby(cust_col)[target_col].max()
df_cust.reset_index(inplace=True)

st.subheader("Customer-level dataset")
st.write(df_cust.head())

# ---------------- TRAIN MODEL ----------------
if st.button("🚀 Train Model"):
    X = df_cust.drop(columns=[cust_col, target_col])
    y = df_cust[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = CatBoostClassifier(
        iterations=300,
        depth=6,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="AUC",
        verbose=0
    )

    model.fit(X_train, y_train)
    df_cust.to_csv("/content/drive/MyDrive/Bad_debt_prediction/customer_features.csv", index=False)



    # -------- Save artifacts to Drive --------
    SAVE_DIR = "/content/drive/MyDrive/Bad_debt_prediction/"
    joblib.dump(model, SAVE_DIR + "model.pkl")
    joblib.dump(X.columns.tolist(), SAVE_DIR + "columns.pkl")
    df_cust.to_csv(SAVE_DIR + "customer_features.csv", index=False)

    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    st.success("✅ Model trained and saved to Google Drive")

    st.write("### 📊 Model Performance")
    st.write("Accuracy:", accuracy_score(y_test, preds))
    st.write("Precision:", precision_score(y_test, preds))
    st.write("Recall:", recall_score(y_test, preds))
    st.write("F1 Score:", f1_score(y_test, preds))
    st.write("ROC-AUC:", roc_auc_score(y_test, proba))

# ---------------- PREDICTION ----------------
st.divider()
st.subheader("🔮 Predict Using Customer ID")

customer_id = st.text_input("Enter Customer ID")

if st.button("Predict"):
    MODEL_PATH = "/content/drive/MyDrive/Bad_debt_prediction/model.pkl"

    if not os.path.exists(MODEL_PATH):
        st.error("❌ Train the model first")
        st.stop()

    model = joblib.load(MODEL_PATH)
    cols = joblib.load("/content/drive/MyDrive/Bad_debt_prediction/columns.pkl")
    df_cust = pd.read_csv("/content/drive/MyDrive/Bad_debt_prediction/customer_features.csv")

    row = df_cust[df_cust.iloc[:, 0].astype(str) == customer_id]

    if row.empty:
        st.error("Customer ID not found")
    else:
        X = row[cols]
        pred = model.predict(X)[0]
        prob = model.predict_proba(X)[0][1]

        result = "🟢 GOOD DEBT (1)" if pred == 1 else "🔴 BAD DEBT (0)"
        st.success(result)
        st.write(f"Probability of GOOD debt: **{prob:.2f}**")

