# loan_eligibility_app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="🏦 Loan Eligibility Prediction", layout="centered")

st.title("🏦 Loan Eligibility Prediction App")
st.markdown(
    """
    Welcome — fill in the simple details below to check whether a loan application is likely to be **Approved** ✅ or **Rejected** ❌.
    The app automatically applies the same preprocessing used in training (label encoding + scaling).
    """
)

# -----------------------------
# Load model (pickle)
# -----------------------------
MODEL_FILE = "loan_eligibility_model.pkl"
DATA_FILE = "loan_approval_dataset.csv"

try:
    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error(f"Model file not found: {MODEL_FILE}. Place your pickled XGBoost model in the app folder.")
    st.stop()

# -----------------------------
# Load dataset to compute scaler/encoders
# -----------------------------
@st.cache_data(show_spinner=False)
def load_and_prep_dataset(path: str):
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        return None, None

    # normalize column names (strip spaces)
    df.columns = [c.strip() for c in df.columns]

    # expected numeric features used for scaling (based on your notebook & dataset)
    numeric_cols = [
        "income_annum", "loan_amount", "loan_term", "cibil_score",
        "residential_assets_value", "commercial_assets_value",
        "luxury_assets_value", "bank_asset_value"
    ]
    # keep only those that exist
    numeric_cols = [c for c in numeric_cols if c in df.columns]

    # prepare label mapping candidates from dataset for education and self_employed
    # We'll use simple binary mapping: Graduate -> 1 else 0; Yes ->1 else 0
    # But compute modes for safety (not strictly necessary)
    scaler = None
    if numeric_cols:
        scaler = StandardScaler()
        scaler.fit(df[numeric_cols].fillna(df[numeric_cols].median()))

    return df, scaler

df_train, scaler = load_and_prep_dataset(DATA_FILE)
if df_train is None:
    st.warning(f"Training dataset '{DATA_FILE}' not found. The app will still run but scaling will be skipped.")
    scaler = None


# -----------------------------
# Input fields (same order & names as dataset excluding loan_status)
# -----------------------------
st.header("📋 Enter Applicant Details")

loan_id = st.text_input("🆔 Loan ID (optional)")

# Note: dataset originally had column name with leading space ' no_of_dependents'
# but we present a clean label to user while preserving expected order.
no_of_dependents = st.number_input("👨‍👩‍👧 Number of Dependents", min_value=0, max_value=20, value=0, step=1)
education = st.selectbox("🎓 Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("💼 Self Employed", ["Yes", "No"])
income_annum = st.number_input("💰 Annual Income (₹)", min_value=0, value=300000, step=10000)
loan_amount = st.number_input("🏠 Loan Amount (₹)", min_value=0, value=500000, step=10000)
loan_term = st.number_input("⏳ Loan Term (in Months)", min_value=1, value=60, step=1)
cibil_score = st.number_input("💳 CIBIL Score", min_value=300, max_value=900, value=650, step=1)
residential_assets_value = st.number_input("🏡 Residential Assets Value (₹)", min_value=0, value=500000, step=10000)
commercial_assets_value = st.number_input("🏢 Commercial Assets Value (₹)", min_value=0, value=0, step=10000)
luxury_assets_value = st.number_input("🚗 Luxury Assets Value (₹)", min_value=0, value=0, step=10000)
bank_asset_value = st.number_input("🏦 Bank Asset Value (₹)", min_value=0, value=100000, step=10000)

st.markdown("---")

# -----------------------------
# Preprocessing helper
# -----------------------------
def preprocess_for_model(
    no_of_dependents, education, self_employed,
    income_annum, loan_amount, loan_term, cibil_score,
    residential_assets_value, commercial_assets_value,
    luxury_assets_value, bank_asset_value,
    scaler_obj=None
):
    # Binary encode education and self_employed consistent with notebook
    edu_val = 1 if str(education).strip().lower() == "graduate" else 0
    self_emp_val = 1 if str(self_employed).strip().lower() == "yes" else 0

    # numeric features in the same order used in training (we will scale these)
    numeric_order = [
        income_annum, loan_amount, loan_term, cibil_score,
        residential_assets_value, commercial_assets_value,
        luxury_assets_value, bank_asset_value
    ]

    numeric_arr = np.array(numeric_order, dtype=float).reshape(1, -1)

    if scaler_obj is not None:
        # fill nan safe: not expected from UI but keep safe
        numeric_arr = scaler_obj.transform(numeric_arr)

    # Construct final feature vector in the same order your model expects:
    # [no_of_dependents, education, self_employed, income_annum, loan_amount,
    #  loan_term, cibil_score, residential_assets_value, commercial_assets_value,
    #  luxury_assets_value, bank_asset_value]
    # Note: numeric values above may be scaled if scaler applied
    final = np.hstack([
        np.array([[no_of_dependents, edu_val, self_emp_val]]),
        numeric_arr
    ])
    return final  # shape (1, 11)

# -----------------------------
# Prediction button
# -----------------------------
if st.button("🔍 Check Loan Eligibility"):
    # create processed input
    processed_X = preprocess_for_model(
        no_of_dependents, education, self_employed,
        income_annum, loan_amount, loan_term, cibil_score,
        residential_assets_value, commercial_assets_value,
        luxury_assets_value, bank_asset_value,
        scaler_obj=scaler
    )

    # predict
    try:
        prediction = model.predict(processed_X)[0]
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    # render result
    if int(prediction) == 1:
        st.markdown(
            """
            <div style='background-color:#d4edda;padding:20px;border-radius:10px;text-align:center;'>
            <h2 style='color:#155724;'>✅ Loan Approved!</h2>
            <p style='color:#155724;font-size:18px;'>Congratulations — the applicant is likely eligible for the loan.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.balloons()
        result_text = "Approved"
    else:
        st.markdown(
            """
            <div style='background-color:#f8d7da;padding:20px;border-radius:10px;text-align:center;'>
            <h2 style='color:#721c24;'>❌ Loan Rejected</h2>
            <p style='color:#721c24;font-size:18px;'>The applicant is unlikely to be eligible for the loan.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        result_text = "Rejected"

    # Save for PDF
    st.session_state["report_data"] = {
        "Loan ID": loan_id,
        "no_of_dependents": no_of_dependents,
        "education": education,
        "self_employed": self_employed,
        "income_annum": income_annum,
        "loan_amount": loan_amount,
        "loan_term": loan_term,
        "cibil_score": cibil_score,
        "residential_assets_value": residential_assets_value,
        "commercial_assets_value": commercial_assets_value,
        "luxury_assets_value": luxury_assets_value,
        "bank_asset_value": bank_asset_value,
        "Result": result_text
    }

    # show short explanation
    st.markdown("**Why this result?**")
    st.write("- Lower requested loan relative to income improves chances.")
    st.write("- Self-employed and education factors are considered by the model.")

# -----------------------------
# Export PDF
# -----------------------------
if "report_data" in st.session_state:
    if st.button("📄 Export PDF Report"):
        data = st.session_state["report_data"]
        pdf_buffer = BytesIO()
        c = canvas.Canvas(pdf_buffer, pagesize=letter)
        c.setTitle("Loan Eligibility Report")

        c.setFont("Helvetica-Bold", 16)
        c.drawString(160, 760, "🏦 Loan Eligibility Report")
        c.setFont("Helvetica", 12)
        c.drawString(50, 730, f"Result: {data['Result']}")
        c.drawString(50, 710, "---------------------------------------------")
        y = 690
        for k, v in data.items():
            if k == "Result":
                continue
            c.drawString(50, y, f"{k}: {v}")
            y -= 18
            if y < 60:
                c.showPage()
                y = 750

        c.showPage()
        c.save()
        pdf_buffer.seek(0)

        st.download_button(
            label="📥 Download PDF Report",
            data=pdf_buffer,
            file_name=f"Loan_Report_{data['Loan ID'] if data.get('Loan ID') else 'Applicant'}.pdf",
            mime="application/pdf"
        )

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit & XGBoost — Preprocessing (encoding + scaling) applied automatically from training data.")
