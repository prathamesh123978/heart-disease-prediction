import streamlit as st
import numpy as np
import joblib
import os

st.set_page_config(page_title="Heart Disease Predictor", layout="centered")

st.title("Heart Disease Risk Predictor")
st.markdown("Fill in the patient details and click **Predict** to get the result.")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    age      = st.number_input("Age", min_value=20, max_value=80, value=50)
    sex      = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female (0)" if x == 0 else "Male (1)")
    cp       = st.selectbox("Chest Pain Type", [0,1,2,3],
                             format_func=lambda x: {0:"Typical Angina",1:"Atypical Angina",2:"Non-Anginal",3:"Asymptomatic"}[x])
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    chol     = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
    fbs      = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0,1], format_func=lambda x: "No (0)" if x==0 else "Yes (1)")
    restecg  = st.selectbox("Resting ECG Result", [0,1,2],
                             format_func=lambda x: {0:"Normal",1:"ST-T Abnormality",2:"Left Ventricular Hypertrophy"}[x])

with col2:
    thalach  = st.number_input("Max Heart Rate Achieved", 60, 220, 150)
    exang    = st.selectbox("Exercise Induced Angina", [0,1], format_func=lambda x: "No (0)" if x==0 else "Yes (1)")
    oldpeak  = st.number_input("ST Depression (Oldpeak)", 0.0, 6.2, 1.0, step=0.1)
    slope    = st.selectbox("Slope of Peak ST Segment", [0,1,2],
                             format_func=lambda x: {0:"Upsloping",1:"Flat",2:"Downsloping"}[x])
    ca       = st.selectbox("Major Vessels Colored by Fluoroscopy (0-3)", [0,1,2,3])
    thal     = st.selectbox("Thalassemia", [1,2,3],
                             format_func=lambda x: {1:"Normal",2:"Fixed Defect",3:"Reversible Defect"}[x])

st.markdown("---")

if st.button("Predict Risk", type="primary", use_container_width=True):
    model_path  = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_model.pkl')
    scaler_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'scaler.pkl')

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        st.error("Model files not found. Please run notebooks 01 to 04 first to train and save the model.")
    else:
        model  = joblib.load(model_path)
        scaler = joblib.load(scaler_path)

        features = np.array([[age, sex, cp, trestbps, chol, fbs,
                               restecg, thalach, exang, oldpeak, slope, ca, thal]])
        scaled      = scaler.transform(features)
        prediction  = model.predict(scaled)[0]
        probability = model.predict_proba(scaled)[0][1] * 100

        st.markdown("### Result")
        if prediction == 1:
            st.error(f"High Risk of Heart Disease — **{probability:.1f}%** probability")
        else:
            st.success(f"Low Risk of Heart Disease — **{probability:.1f}%** probability")

        st.markdown("---")
        st.markdown("**Input Summary**")
        input_dict = {
            "Age": age, "Sex": sex, "Chest Pain": cp, "Blood Pressure": trestbps,
            "Cholesterol": chol, "Fasting BS": fbs, "ECG": restecg,
            "Max HR": thalach, "Exercise Angina": exang, "Oldpeak": oldpeak,
            "Slope": slope, "CA": ca, "Thal": thal
        }
        st.json(input_dict)
