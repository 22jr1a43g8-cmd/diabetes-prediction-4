
import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="centered"
)

# Load trained model
model = joblib.load("diabetes_model.pkl")

# Application title
st.title("🩺 Diabetes Prediction System")

st.write(
    "Enter the medical measurements below "
    "to obtain an educational model prediction."
)

# User inputs
pregnancies = st.number_input(
    "Pregnancies", min_value=0,
    max_value=20, value=2
)

glucose = st.number_input(
    "Glucose", min_value=1,
    max_value=300, value=120
)

blood_pressure = st.number_input(
    "Blood Pressure", min_value=1,
    max_value=200, value=70
)

skin_thickness = st.number_input(
    "Skin Thickness", min_value=1,
    max_value=100, value=25
)

insulin = st.number_input(
    "Insulin", min_value=1,
    max_value=1000, value=80
)

bmi = st.number_input(
    "BMI", min_value=1.0,
    max_value=100.0, value=28.5
)

dpf = st.number_input(
    "Diabetes Pedigree Function",
    min_value=0.0, max_value=3.0,
    value=0.5
)

age = st.number_input(
    "Age", min_value=1,
    max_value=120, value=30
)

# Prediction button
if st.button("Predict Diabetes"):

    input_data = pd.DataFrame({
        'Pregnancies': [pregnancies],
        'Glucose': [glucose],
        'BloodPressure': [blood_pressure],
        'SkinThickness': [skin_thickness],
        'Insulin': [insulin],
        'BMI': [bmi],
        'DiabetesPedigreeFunction': [dpf],
        'Age': [age]
    })

    prediction = model.predict(input_data)

    probability = model.predict_proba(
        input_data
    )[0][1]

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.warning(
            "Model prediction: Diabetes indicated"
        )
    else:
        st.success(
            "Model prediction: Diabetes not indicated"
        )

    st.write(
        f"Predicted probability: {probability * 100:.2f}%"
    )

    st.info(
        "This educational model is not a medical "
        "diagnostic tool. Consult a qualified "
        "healthcare professional for assessment."
    )
