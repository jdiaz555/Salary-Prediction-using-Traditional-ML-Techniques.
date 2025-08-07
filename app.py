import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import os
import datetime
import time

# Page config
st.set_page_config(
    page_title="Salary Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load models
@st.cache_resource
def load_models():
    rf = pickle.load(open("random_forest_model (2).pkl", "rb"))
    lr = pickle.load(open("linear_regression_model (2).pkl", "rb"))
    return {"Random Forest": rf, "Linear Regression": lr}

models = load_models()

# Encoded mappings
education_options = [
    'Associate Degree', 'Bachelor’s Degree', 'Master’s Degree', 'Doctorate',
    'High School', 'Professional Certification', 'Diploma'
]

job_title_options = [
    'Software Engineer', 'Data Analyst', 'Project Manager', 'Sales Manager',
    'Product Manager', 'Data Scientist', 'UX Designer', 'Marketing Manager',
    'HR Manager', 'Operations Manager'
]

edu_mapping = {val: idx for idx, val in enumerate(education_options)}
job_mapping = {val: idx for idx, val in enumerate(job_title_options)}

# Sidebar - user input
with st.sidebar:
    st.header("Input Features")
    education = st.selectbox("Education Level", education_options)
    job_title = st.selectbox("Job Title", job_title_options)
    experience = st.slider("Years of Experience", 0.0, 40.0, 2.0, 0.5)
    selected_model = st.selectbox("Select Model", list(models.keys()))

# Main title and intro
st.markdown("""
    <h1 style='text-align: center;'>Salary Prediction App</h1>
    <p style='text-align: center;'>Predict your expected salary using machine learning.</p>
    <hr style="border: 1px solid #ccc;">
""", unsafe_allow_html=True)

# Predict button
if st.button("Predict Salary"):
    edu_encoded = edu_mapping.get(education, 0)
    job_encoded = job_mapping.get(job_title, 0)
    input_data = np.array([[experience, edu_encoded, job_encoded]])
    model = models[selected_model]

    with st.spinner("Generating prediction..."):
        time.sleep(1.2)
        predicted_salary = model.predict(input_data)[0]

    st.subheader("Prediction Result")
    col1, col2 = st.columns(2)
    col1.metric(label="Predicted Salary", value=f"${predicted_salary:,.2f}")
    col2.metric(label="Years of Experience", value=f"{experience} years")

    # Save log
    log_entry = {
        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'education_level': education,
        'job_title': job_title,
        'years_experience': experience,
        'model_used': selected_model,
        'predicted_salary': predicted_salary
    }

    log_file = 'prediction_log.csv'
    if os.path.exists(log_file):
        log_df = pd.read_csv(log_file)
        log_df = pd.concat([log_df, pd.DataFrame([log_entry])], ignore_index=True)
    else:
        log_df = pd.DataFrame([log_entry])

    log_df.to_csv(log_file, index=False)
    st.success("Prediction logged successfully!")

    # Plotly chart for salary trend
    st.subheader("Predicted Salary Trend")
    x_vals = np.linspace(0, 40, 100)
    y_vals = [model.predict([[x, edu_encoded, job_encoded]])[0] for x in x_vals]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', name='Trend', line=dict(color='#00BFFF')))
    fig.add_trace(go.Scatter(x=[experience], y=[predicted_salary], mode='markers',
                             name='Your Prediction', marker=dict(size=10, color='red')))
    fig.update_layout(title=f"{selected_model} Prediction: {job_title} with {education}",
                      xaxis_title='Years of Experience',
                      yaxis_title='Salary',
                      template='plotly_white',
                      height=400)
    st.plotly_chart(fig, use_container_width=True)

# Model metrics
with st.expander("Model Performance"):
    st.markdown("""
    | Model              | MAE      | RMSE      | R² Score |
    |-------------------|----------|-----------|----------|
    | Linear Regression | 6,286.45 | 7,059.04  | 0.90     |
    | Random Forest     | 6,872.01 | 7,982.55  | 0.88     |
    """)

# Logs
with st.expander("Prediction Logs"):
    if os.path.exists("prediction_log.csv"):
        log_data = pd.read_csv("prediction_log.csv")
        st.dataframe(log_data.tail(20), use_container_width=True)
        st.download_button("Download Full Log as CSV", data=log_data.to_csv(index=False), file_name="prediction_log.csv", mime="text/csv")
    else:
        st.info("No predictions logged yet.")

# About
with st.expander("About This App"):
    st.info("""
    This app predicts salaries based on education, job title, and experience level.
    Built with Python, Streamlit, and Scikit-learn.
    Includes both Linear Regression and Random Forest models.
    """)
