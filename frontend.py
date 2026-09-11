
import streamlit as st
import requests


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊"
)


# ============================================================
# TITLE
# ============================================================

st.title("📊 Customer Churn Predictor")

st.write(
    "Enter the customer's information below to predict "
    "their probability of churn."
)


# ============================================================
# CUSTOMER INFORMATION
# ============================================================

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

senior_citizen = st.selectbox(
    "Senior Citizen",
    [0, 1]
)

partner = st.selectbox(
    "Partner",
    ["Yes", "No"]
)

dependents = st.selectbox(
    "Dependents",
    ["Yes", "No"]
)

tenure = st.number_input(
    "Tenure (months)",
    min_value=0,
    value=5
)

phone_service = st.selectbox(
    "Phone Service",
    ["Yes", "No"]
)

multiple_lines = st.selectbox(
    "Multiple Lines",
    ["Yes", "No", "No phone service"]
)

internet_service = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

online_security = st.selectbox(
    "Online Security",
    ["Yes", "No", "No internet service"]
)

online_backup = st.selectbox(
    "Online Backup",
    ["Yes", "No", "No internet service"]
)

device_protection = st.selectbox(
    "Device Protection",
    ["Yes", "No", "No internet service"]
)

tech_support = st.selectbox(
    "Tech Support",
    ["Yes", "No", "No internet service"]
)

streaming_tv = st.selectbox(
    "Streaming TV",
    ["Yes", "No", "No internet service"]
)

streaming_movies = st.selectbox(
    "Streaming Movies",
    ["Yes", "No", "No internet service"]
)

contract = st.selectbox(
    "Contract",
    [
        "Month-to-month",
        "One year",
        "Two year"
    ]
)

paperless_billing = st.selectbox(
    "Paperless Billing",
    ["Yes", "No"]
)

payment_method = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

monthly_charges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=70.0
)

total_charges = st.number_input(
    "Total Charges",
    min_value=0.0,
    value=350.0
)


# ============================================================
# PREDICTION
# ============================================================

if st.button("Predict Churn"):

    # Create the data that will be sent to FastAPI.

    customer_data = {

        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }


    # Send the data to our FastAPI backend.

    response = requests.post(
        "http://127.0.0.1:8000/predict",
        json=customer_data
    )


    # Check whether the request was successful.

    if response.status_code == 200:

        result = response.json()

        prediction = result["prediction"]
        probability = result["churn_probability"]


        # Display probability.

        st.subheader("Prediction Result")

        st.metric(
            "Churn Probability",
            f"{probability:.1%}"
        )


        # Display final prediction.

        if prediction == 1:

            st.error(
                "⚠️ High Risk: Customer is predicted to churn."
            )

        else:

            st.success(
                "✅ Low Risk: Customer is predicted to stay."
            )


    else:

        st.error(
            "Something went wrong while contacting the API."
        )

