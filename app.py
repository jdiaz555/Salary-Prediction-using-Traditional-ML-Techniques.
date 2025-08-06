import streamlit as st
import joblib
import numpy as np
import os

# Load both models
lr_model = joblib.load("linear_regression_model (1).pkl")
rf_model = joblib.load("random_forest_model (1).pkl")

# Custom CSS for styling
st.markdown("""
    <style>
        .main {
            background-color: #f7f9fc;
        }
        .title {
            font-size: 32px;
            color: #2c3e50;
            text-align: center;
        }
        .subtitle {
            color: #7f8c8d;
            font-size: 18px;
            text-align: center;
        }
        .footer {
            font-size: 12px;
            text-align: center;
            color: #bdc3c7;
            margin-top: 60px;
        }
    </style>
""", unsafe_allow_html=True)

# Title & Subtitle
st.markdown('<div class="title"> Salary Prediction App</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Powered by Traditional ML Models</div>', unsafe_allow_html=True)
st.write("")

# Sidebar - About & Model selection
with st.sidebar:
    st.header(" About")
    st.write("Predict your estimated salary based on years of experience using ML models.")
    st.write("Choose between Linear Regression and Random Forest.")
    st.markdown("---")
    model_choice = st.radio(" Select Model", ["Linear Regression", "Random Forest"])

# Input: Years of Experience
experience = st.slider(" Enter your years of experience:", 0.0, 20.0, step=0.1)

# Predict Button
if st.button(" Predict Salary"):
    input_data = np.array([[experience]])
    
    if model_choice == "Linear Regression":
        prediction = lr_model.predict(input_data)[0]
        selected_model = "Linear Regression"
    else:
        prediction = rf_model.predict(input_data)[0]
        selected_model = "Random Forest Regressor"
    
    st.success(f" Estimated Salary: **${prediction:,.2f}**")
    st.caption(f" Prediction based on: `{selected_model}`")

    # Progress animation
    st.progress(min(1.0, prediction / 200000))
    st.balloons()

# Footer
st.markdown('<div class="footer"> Developed with  using Streamlit and scikit-learn</div>', unsafe_allow_html=True)
