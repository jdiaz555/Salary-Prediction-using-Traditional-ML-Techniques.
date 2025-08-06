import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("salary_predictor.ipynb")

# Custom CSS Styling
st.markdown("""
    <style>
        .main {
            background-color: #f0f2f6;
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
            margin-bottom: 20px;
        }
        .footer {
            font-size: 12px;
            text-align: center;
            color: #bdc3c7;
            margin-top: 60px;
        }
    </style>
""", unsafe_allow_html=True)

# Page Title
st.markdown('<div class="title"> Salary Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Estimate your salary based on your experience</div>', unsafe_allow_html=True)
st.write("")

# Sidebar Info
with st.sidebar:
    st.header(" About This App")
    st.write("This is a machine learning-powered app that predicts the estimated salary based on years of experience.")
    st.write("The model is trained using linear regression for high accuracy.")
    st.markdown("---")
    st.write(" Developed by [Your Name]")

# User Input
st.subheader(" Enter your information:")
experience = st.slider("Years of Experience", 0.0, 20.0, step=0.1)

# Predict Salary
if st.button(" Predict Salary"):
    input_data = np.array([[experience]])
    prediction = model.predict(input_data)[0]
    st.success(f" Estimated Salary: **${prediction:,.2f}**")

    # Bonus visual feedback
    st.progress(min(1.0, prediction / 200000))  # basic progress bar visual
    st.balloons()

# Footer
st.markdown('<div class="footer"> Powered by Machine Learning | Styled with  using Streamlit</div>', unsafe_allow_html=True)
