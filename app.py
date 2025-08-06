import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Load both models
lr_model = joblib.load("linear_regression_model (1).pkl")
rf_model = joblib.load("random_forest_model (1).pkl")

# Custom CSS Styling for bold title
st.markdown("""
    <style>
        .main {
            background-color: #f7f9fc;
        }
        .title {
            font-size: 36px;
            font-weight: bold;
            color: #1a73e8;
            text-align: center;
            margin-bottom: 0px;
        }
        .subtitle {
            font-size: 18px;
            color: #555;
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

# Title & Subtitle
st.markdown('<div class="title"> Salary Prediction App</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Select ML model & predict your salary based on experience</div>', unsafe_allow_html=True)

# Sidebar - Info & Model choice
with st.sidebar:
    st.header(" About the App")
    st.write("This app predicts your estimated salary based on years of experience using ML regression models.")
    model_choice = st.radio("🔍 Select Model", ["Linear Regression", "Random Forest"])

# Input
experience = st.slider(" Enter your years of experience:", 0.0, 20.0, step=0.1)

# Predict
if st.button(" Predict Salary"):
    input_data = np.array([[experience]])

    if model_choice == "Linear Regression":
        model = lr_model
        model_name = "Linear Regression"
    else:
        model = rf_model
        model_name = "Random Forest Regressor"

    prediction = model.predict(input_data)

    st.success(f" Estimated Salary: **${prediction[0]:,.2f}**")
    st.caption(f" Based on: `{model_name}`")

    # Progress bar + balloons
    st.progress(min(1.0, prediction[0] / 200000))
    st.balloons()

    # Visualization
    fig, ax = plt.subplots()
    X_plot = np.linspace(0, 20, 100).reshape(-1, 1)
    y_plot = model.predict(X_plot)
    ax.plot(X_plot, y_plot, label="Prediction Line", color="blue")
    ax.scatter(experience, prediction[0], color="red", s=100, label="Your Prediction")
    ax.set_xlabel("Years of Experience")
    ax.set_ylabel("Predicted Salary")
    ax.set_title("Salary vs. Experience")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

# Footer
st.markdown('<div class="footer"> Built with Streamlit, scikit-learn & by Zubair</div>', unsafe_allow_html=True)
