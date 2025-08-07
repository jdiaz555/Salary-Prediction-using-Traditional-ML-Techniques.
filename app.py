import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# Set Streamlit page config
st.set_page_config(page_title="Salary Predictor", layout="centered")

# Load models
@st.cache_resource
def load_models():
    lr = joblib.load("linear_regression_model.pkl")
    rf = joblib.load("random_forest_model.pkl")
    default = joblib.load("salary_predictor.pkl")
    return {"Linear Regression": lr, "Random Forest": rf, "Default": default}

models = load_models()

# Sidebar Inputs
st.sidebar.title("Salary Predictor Inputs")

experience = st.sidebar.slider("Years of Experience", 0.0, 40.0, 2.0, 0.5)

education_options = {
    'High School': 0,
    'Associate Degree': 1,
    'Diploma': 2,
    'Professional Certification': 3,
    'Bachelor’s Degree': 4,
    'Master’s Degree': 5,
    'Doctorate': 6
}
education = st.sidebar.selectbox("Education Level", list(education_options.keys()))
education_code = education_options[education]

job_roles = {
    'Software Engineer': 0,
    'Data Analyst': 1,
    'Project Manager': 2,
    'Sales Manager': 3,
    'Product Manager': 4,
    'Data Scientist': 5,
    'UX Designer': 6,
    'Marketing Manager': 7,
    'HR Manager': 8,
    'Operations Manager': 9
}
job_title = st.sidebar.selectbox("Job Role", list(job_roles.keys()))
job_code = job_roles[job_title]

skill_rating = st.sidebar.slider("Skill Rating (1-5)", 1, 5, 3)
company_tier = st.sidebar.slider("Company Tier (0 = Low, 2 = Top)", 0, 2, 1)

# Model selector
model_choice = st.sidebar.radio("Select Model", ("Linear Regression", "Random Forest"))

# Main Title
st.title("💼 Salary Prediction Web App")

if st.button("Predict Salary"):
    input_data = np.array([[experience, education_code, job_code, skill_rating, company_tier]])
    model = models.get(model_choice)
    prediction = model.predict(input_data)[0]

    st.success(f"🎯 Predicted Salary: ${prediction:,.2f}")
    st.write("**Inputs:**")
    st.json({
        "Years Experience": experience,
        "Education": education,
        "Job Role": job_title,
        "Skill Rating": skill_rating,
        "Company Tier": company_tier
    })
