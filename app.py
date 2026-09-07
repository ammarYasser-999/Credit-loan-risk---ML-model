import streamlit as st
import pandas as pd
import joblib
import numpy as np


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Credit Risk Prediction",
    page_icon="💳",
    layout="centered"
)


# ============================================================
# LOAD MODEL & PREPROCESSING ARTIFACTS
# ============================================================

@st.cache_resource
def load_artifacts():

    model = joblib.load(
        "models/xgboost_weighted.pkl"
    )

    preprocessor = joblib.load(
        "models/preprocessor_unscaled.pkl"
    )

    income_threshold = joblib.load(
        "models/income_threshold.pkl"
    )

    return model, preprocessor, income_threshold


model, preprocessor, income_threshold = load_artifacts()


# ============================================================
# HEADER
# ============================================================

st.title("💳 Credit Risk Prediction")

st.write(
    "Enter the customer's information below to predict the probability of loan default."
)

st.markdown("---")


# ============================================================
# PERSONAL INFORMATION
# ============================================================

st.subheader("👤 Personal Information")

col1, col2 = st.columns(2)

with col1:

    person_age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30
    )

    person_income = st.number_input(
        "Annual Income",
        min_value=0.0,
        value=50000.0,
        step=1000.0
    )

    person_home_ownership = st.selectbox(
        "Home Ownership",
        [
            "MORTGAGE",
            "OWN",
            "RENT",
            "OTHER"
        ]
    )


with col2:

    person_emp_length = st.number_input(
        "Employment Length (Years)",
        min_value=0.0,
        max_value=50.0,
        value=5.0,
        step=1.0
    )

    cb_person_default_on_file = st.selectbox(
        "Previous Default",
        [
            "N",
            "Y"
        ],
        format_func=lambda x: (
            "No" if x == "N" else "Yes"
        )
    )

    cb_person_cred_hist_length = st.number_input(
        "Credit History Length (Years)",
        min_value=0.0,
        max_value=100.0,
        value=5.0,
        step=1.0
    )


st.markdown("---")


# ============================================================
# LOAN INFORMATION
# ============================================================

st.subheader("💰 Loan Information")

col1, col2 = st.columns(2)

with col1:

    loan_intent = st.selectbox(
        "Loan Purpose",
        [
            "DEBTCONSOLIDATION",
            "EDUCATION",
            "HOMEIMPROVEMENT",
            "MEDICAL",
            "PERSONAL",
            "VENTURE"
        ],
        format_func=lambda x: {
            "DEBTCONSOLIDATION": "Debt Consolidation",
            "EDUCATION": "Education",
            "HOMEIMPROVEMENT": "Home Improvement",
            "MEDICAL": "Medical",
            "PERSONAL": "Personal",
            "VENTURE": "Business / Venture"
        }[x]
    )

    loan_grade = st.selectbox(
        "Loan Grade",
        [
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
            "G"
        ]
    )

    loan_amnt = st.number_input(
        "Loan Amount",
        min_value=0.0,
        value=10000.0,
        step=1000.0
    )


with col2:

    loan_int_rate = st.number_input(
        "Interest Rate (%)",
        min_value=0.0,
        max_value=100.0,
        value=10.0,
        step=0.1
    )

    loan_percent_income = st.number_input(
        "Loan to Income Ratio",
        min_value=0.0,
        max_value=1.0,
        value=0.20,
        step=0.01,
        format="%.2f"
    )


st.markdown("---")


# ============================================================
# PREDICT BUTTON
# ============================================================

if st.button(
    "🔮 Predict Credit Risk",
    use_container_width=True,
    type="primary"
):

    try:

        # ====================================================
        # FEATURE ENGINEERING
        # ====================================================

        emp_length_ratio = (
            person_emp_length / person_age
            if person_age != 0
            else 0
        )


        # ====================================================
        # CREATE INPUT DATAFRAME
        # ====================================================

        input_data = pd.DataFrame([{

            "person_age": person_age,

            "person_income": person_income,

            "person_home_ownership": person_home_ownership,

            "person_emp_length": person_emp_length,

            "loan_intent": loan_intent,

            "loan_grade": loan_grade,

            "loan_amnt": loan_amnt,

            "loan_int_rate": loan_int_rate,

            "loan_percent_income": loan_percent_income,

            "cb_person_default_on_file": cb_person_default_on_file,

            "cb_person_cred_hist_length":
                cb_person_cred_hist_length,

            "emp_length_ratio":
                emp_length_ratio

        }])


        # ====================================================
        # APPLY SAME INCOME THRESHOLD LOGIC
        # ====================================================

        input_data.loc[
            input_data["person_income"] > income_threshold,
            "person_income"
        ] = np.nan


        # ====================================================
        # PREPROCESSING
        # ====================================================

        X_processed = preprocessor.transform(
            input_data
        )


        # ====================================================
        # PREDICTION
        # ====================================================

        prediction = model.predict(
            X_processed
        )[0]


        probability = model.predict_proba(
            X_processed
        )[0, 1]


        # ====================================================
        # RESULT
        # ====================================================

        st.markdown("---")

        st.subheader("📊 Prediction Result")


        if prediction == 1:

            st.error(
                "⚠️ High Credit Risk"
            )

            st.write(
                "The model indicates a higher probability of loan default."
            )

        else:

            st.success(
                "✅ Low Credit Risk"
            )

            st.write(
                "The model indicates a lower probability of loan default."
            )


        # Probability display

        st.metric(
            "Estimated Default Probability",
            f"{probability * 100:.2f}%"
        )

        st.progress(
            float(probability)
        )


    except Exception as e:

        st.error(
            f"Prediction Error: {str(e)}"
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Credit Risk Prediction System"
)