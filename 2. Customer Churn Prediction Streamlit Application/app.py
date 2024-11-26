import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import pickle

# Load the trained model, scaler, and encoders
model = load_model("customer_churn_model.h5")
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)


# Function to preprocess user input
def preprocess_input(data):
    # Encode categorical features
    for column, encoder in label_encoders.items():
        if column in data:
            if isinstance(data[column], list):
                data[column] = data[column][0]  # Ensure it's not a list
            if data[column] not in encoder.classes_:
                st.error(f"Invalid value '{data[column]}' for column '{column}'.")
                return None
            data[column] = encoder.transform([data[column]])[0]

    # Validate and scale numerical features
    numerical_features = ["tenure", "MonthlyCharges", "TotalCharges"]
    try:
        scaled_features = scaler.transform(
            [[data["tenure"], data["MonthlyCharges"], data["TotalCharges"]]]
        )[
            0
        ]  # Ensure the scaler processes as expected
        for i, feature in enumerate(numerical_features):
            data[feature] = scaled_features[i]
    except Exception as e:
        st.error(f"Error during scaling: {e}")
        return None

    # Arrange features in the correct order
    feature_order = [
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "tenure",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
        "MonthlyCharges",
        "TotalCharges",
    ]
    missing_features = [col for col in feature_order if col not in data]
    if missing_features:
        st.error(f"Missing features in data: {missing_features}")
        return None

    return np.array([data[col] for col in feature_order]).reshape(1, -1)


# Streamlit app UI
# Streamlit app UI
st.markdown(
    """
    <div style="text-align: center;">
        <h1>Customer Churn Prediction App</h1>
    </div>
    """, 
    unsafe_allow_html=True
)
st.write(
    "Use this app to predict the likelihood of a customer churning based on their information."
)
# Group 1: Customer Demographics
st.header("1. Customer Demographics")
gender = st.selectbox(
    "Gender", ["Male", "Female"], help="Select the customer's gender."
)
SeniorCitizen = st.radio(
    "Senior Citizen", [0, 1], help="Is the customer a senior citizen? (0 = No, 1 = Yes)"
)
Partner = st.selectbox(
    "Partner", ["Yes", "No"], help="Does the customer have a partner?"
)
Dependents = st.selectbox(
    "Dependents", ["Yes", "No"], help="Does the customer have dependents?"
)

# Group 2: Service Information
st.header("2. Service Information")
tenure = st.slider(
    "Tenure (months)",
    0,
    72,
    24,
    help="How long has the customer been with the service? (Range: 0-72)",
)
PhoneService = st.selectbox(
    "Phone Service", ["Yes", "No"], help="Does the customer have phone service?"
)
MultipleLines = st.selectbox(
    "Multiple Lines",
    ["Yes", "No", "No phone service"],
    help="Does the customer have multiple phone lines?",
)
InternetService = st.selectbox(
    "Internet Service", ["DSL", "Fiber optic", "No"], help="Type of internet service."
)
OnlineSecurity = st.selectbox(
    "Online Security",
    ["Yes", "No", "No internet service"],
    help="Does the customer have online security?",
)
OnlineBackup = st.selectbox(
    "Online Backup",
    ["Yes", "No", "No internet service"],
    help="Does the customer have online backup?",
)
DeviceProtection = st.selectbox(
    "Device Protection",
    ["Yes", "No", "No internet service"],
    help="Does the customer have device protection?",
)
TechSupport = st.selectbox(
    "Tech Support",
    ["Yes", "No", "No internet service"],
    help="Does the customer have technical support?",
)
StreamingTV = st.selectbox(
    "Streaming TV",
    ["Yes", "No", "No internet service"],
    help="Does the customer have streaming TV?",
)
StreamingMovies = st.selectbox(
    "Streaming Movies",
    ["Yes", "No", "No internet service"],
    help="Does the customer have streaming movies?",
)

# Group 3: Contract and Billing
st.header("3. Contract and Billing Information")
Contract = st.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"],
    help="The contract type of the customer.",
)
PaperlessBilling = st.selectbox(
    "Paperless Billing", ["Yes", "No"], help="Is the customer using paperless billing?"
)
PaymentMethod = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
    help="How does the customer pay?",
)
MonthlyCharges = st.number_input(
    "Monthly Charges ($)",
    min_value=0.0,
    max_value=120.0,
    step=0.1,
    help="Monthly charges billed to the customer. (Range: 0-120)",
)
TotalCharges = st.number_input(
    "Total Charges ($)",
    min_value=0.0,
    max_value=8000.0,
    step=0.1,
    help="Total charges billed to the customer. (Range: 0-8000)",
)

# Collect inputs into a dictionary
user_data = {
    "gender": gender,
    "SeniorCitizen": SeniorCitizen,
    "Partner": Partner,
    "Dependents": Dependents,
    "tenure": tenure,
    "PhoneService": PhoneService,
    "MultipleLines": MultipleLines,
    "InternetService": InternetService,
    "OnlineSecurity": OnlineSecurity,
    "OnlineBackup": OnlineBackup,
    "DeviceProtection": DeviceProtection,
    "TechSupport": TechSupport,
    "StreamingTV": StreamingTV,
    "StreamingMovies": StreamingMovies,
    "Contract": Contract,
    "PaperlessBilling": PaperlessBilling,
    "PaymentMethod": PaymentMethod,
    "MonthlyCharges": MonthlyCharges,
    "TotalCharges": TotalCharges,
}

# Predict button
if st.button("Predict Churn"):
    try:
        input_data = preprocess_input(user_data)
        if input_data is not None:
            prediction = model.predict(input_data)[0][0]
            st.write(f"**Churn Probability:** {prediction:.2f}")
            if prediction > 0.5:
                st.error("This customer is likely to churn.")
            else:
                st.success("This customer is unlikely to churn.")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
