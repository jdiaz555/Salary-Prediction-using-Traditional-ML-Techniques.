import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os
import datetime

# --- PAGE CONFIG ---
st.set_page_config(page_title="Salary Predictor", layout="wide")

# --- LOAD MODELS ---
@st.cache_resource
def load_models():
    lr = joblib.load("linear_regression_model.pkl")
    rf = joblib.load("random_forest_model.pkl")
    default = joblib.load("salary_predictor.pkl")
    return {"Linear Regression": lr, "Random Forest": rf, "Default": default}

models = load_models()

# --- ENCODING MAPS ---
education_options = {
    'High School': 0,
    'Associate Degree': 1,
    'Diploma': 2,
    'Professional Certification': 3,
    'Bachelor’s Degree': 4,
    'Master’s Degree': 5,
    'Doctorate': 6
}
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

# --- SIDEBAR INPUT ---
st.sidebar.title("User Input")

experience = st.sidebar.slider("Years of Experience", 0.0, 40.0, 2.0, 0.5)
education = st.sidebar.selectbox("Education Level", list(education_options.keys()))
job_title = st.sidebar.selectbox("Job Role", list(job_roles.keys()))
skill_rating = st.sidebar.slider("Skill Rating (1–5)", 1, 5, 3)
company_tier = st.sidebar.slider("Company Tier (0 = Low, 2 = Top)", 0, 2, 1)
model_choice = st.sidebar.radio("Choose Model", ("Linear Regression", "Random Forest"))

# --- MAIN CONTENT ---
st.title("Interactive Salary Predictor")

if st.button("Predict Salary"):
    edu_code = education_options[education]
    job_code = job_roles[job_title]
    input_data = np.array([[experience, edu_code, job_code, skill_rating, company_tier]])
    model = models.get(model_choice)
    predicted_salary = model.predict(input_data)[0]

    st.success(f"Predicted Salary: **${predicted_salary:,.2f}**")

    with st.expander("Prediction Summary", expanded=True):
        st.write({
            "Years of Experience": experience,
            "Education": education,
            "Job Role": job_title,
            "Skill Rating": skill_rating,
            "Company Tier": company_tier,
            "Model Used": model_choice
        })

    # --- LOGGING ---
    log_entry = {
        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'experience': experience,
        'education': education,
        'job_title': job_title,
        'skill_rating': skill_rating,
        'company_tier': company_tier,
        'model': model_choice,
        'predicted_salary': predicted_salary
    }

    if os.path.exists("prediction_logs.csv"):
        df_log = pd.read_csv("prediction_logs.csv")
        df_log = pd.concat([df_log, pd.DataFrame([log_entry])], ignore_index=True)
    else:
        df_log = pd.DataFrame([log_entry])
    df_log.to_csv("prediction_logs.csv", index=False)

    # --- VISUALIZATION ---
    st.subheader("Salary vs. Experience Curve")
    exp_range = np.linspace(0, 40, 100)
    salary_curve = [
        model.predict(np.array([[x, edu_code, job_code, skill_rating, company_tier]]))[0]
        for x in exp_range
    ]

    fig, ax = plt.subplots()
    ax.plot(exp_range, salary_curve, color='blue', label="Salary Prediction Curve")
    ax.scatter(experience, predicted_salary, color='red', s=100, label="Your Input")
    ax.set_xlabel("Years of Experience")
    ax.set_ylabel("Predicted Salary")
    ax.set_title("Salary Prediction vs. Experience")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    # --- OPTIONAL LOG VIEW ---
    with st.expander("View Last 10 Predictions"):
        st.dataframe(df_log.tail(10), use_container_width=True)

    # --- DOWNLOAD LOGS ---
    st.download_button("Download Full Log", df_log.to_csv(index=False), "prediction_logs.csv", "text/csv")

# --- FOOTER ---
st.markdown("""
---
Built with  using Streamlit & scikit-learn • [GitHub](https://github.com/Zubair-hussain)
""")
