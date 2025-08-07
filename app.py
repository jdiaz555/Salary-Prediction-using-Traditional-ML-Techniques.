import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import os
import datetime
import time

# ------------------- Page Config -------------------
st.set_page_config(
    page_title="Salary Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------- Load Models -------------------
@st.cache_resource
def load_models():
    rf = pickle.load(open("random_forest_model.pkl", "rb"))
    lr = pickle.load(open("linear_regression_model.pkl", "rb"))
    return {"Random Forest": rf, "Linear Regression": lr}

models = load_models()

# ------------------- Sidebar Inputs -------------------
with st.sidebar:
    st.title("User Input")
    education_options = [
        'Associate Degree', 'Bachelor’s Degree', 'Master’s Degree', 'Doctorate',
        'High School', 'Professional Certification', 'Diploma'
    ]
    job_title_options = [
        'Software Engineer', 'Data Analyst', 'Project Manager', 'Sales Manager',
        'Product Manager', 'Data Scientist', 'UX Designer', 'Marketing Manager',
        'HR Manager', 'Operations Manager'
    ]

    education = st.selectbox("Education Level", education_options)
    job_title = st.selectbox("Job Title", job_title_options)
    experience = st.slider("Years of Experience", 0.0, 40.0, 2.0, 0.5)
    selected_model = st.selectbox("Choose Model", list(models.keys()))

# ------------------- Main Title -------------------
st.markdown("<h1 style='text-align: center;'>Salary Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Enter your details to predict your expected salary.</p>", unsafe_allow_html=True)
st.markdown("---")

# ------------------- Encodings -------------------
edu_mapping = {val: idx for idx, val in enumerate(education_options)}
job_mapping = {val: idx for idx, val in enumerate(job_title_options)}
edu_encoded = edu_mapping.get(education, 0)
job_encoded = job_mapping.get(job_title, 0)
input_data = np.array([[experience, edu_encoded, job_encoded]])

# ------------------- Predict Button -------------------
if st.button("Predict Salary"):
    with st.spinner("Generating prediction..."):
        time.sleep(1)
        model = models[selected_model]
        predicted_salary = model.predict(input_data)[0]

    st.markdown("### Prediction Result")
    col1, col2 = st.columns(2)
    col1.metric(label="Predicted Salary", value=f"${predicted_salary:,.2f}")
    col2.metric(label="Experience", value=f"{experience} years")

    # ------------------- Save Logs -------------------
    log_entry = {
        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'education': education,
        'job_title': job_title,
        'experience': experience,
        'model': selected_model,
        'predicted_salary': predicted_salary
    }

    log_file = 'prediction_log.csv'
    if os.path.exists(log_file):
        df = pd.read_csv(log_file)
        df = pd.concat([df, pd.DataFrame([log_entry])], ignore_index=True)
    else:
        df = pd.DataFrame([log_entry])

    df.to_csv(log_file, index=False)

    # ------------------- Plotly Chart -------------------
    st.markdown("### Predicted Salary Trend")
    x_vals = np.linspace(0, 40, 100)
    y_vals = [model.predict([[x, edu_encoded, job_encoded]])[0] for x in x_vals]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', name='Prediction Line', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=[experience], y=[predicted_salary], mode='markers', name='Your Prediction', marker=dict(size=10, color='red')))
    fig.update_layout(
        xaxis_title='Years of Experience',
        yaxis_title='Predicted Salary',
        template='plotly_white',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

# ------------------- View Logs -------------------
with st.expander(" View Prediction Logs"):
    if os.path.exists("prediction_log.csv"):
        log_data = pd.read_csv("prediction_log.csv")
        st.dataframe(log_data.tail(20), use_container_width=True)
        st.download_button("⬇ Download Log", data=log_data.to_csv(index=False), file_name="prediction_log.csv")
    else:
        st.info("No predictions yet.")

# ------------------- Model Metrics -------------------
with st.expander(" Model Performance"):
    st.markdown("""
    | Model              | MAE     | RMSE    | R² Score |
    |--------------------|---------|---------|----------|
    | Linear Regression  | 6,286.45| 7,059.04| **0.90** |
    | Random Forest      | 6,872.01| 7,982.55| 0.88     |
    """)

# ------------------- About -------------------
with st.expander("ℹ About This App"):
    st.markdown("""
    This application predicts salaries based on a user's years of experience, education level, and job title using machine learning models (Random Forest & Linear Regression).  
    Developed by **Syed Zubair Hussain Shah**.
    """)
