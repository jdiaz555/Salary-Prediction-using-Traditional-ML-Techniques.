import streamlit as st
import numpy as np
import pandas as pd
import pickle
import plotly.graph_objects as go
import os
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Page configuration
st.set_page_config(page_title="Salary Predictor", layout="wide")

# Load Models
@st.cache_resource
def load_models():
    lr = pickle.load(open("linear_regression_model.pkl", "rb"))
    rf = pickle.load(open("random_forest_model.pkl", "rb"))
    return {"Linear Regression": lr, "Random Forest": rf}

models = load_models()

# Title
st.title("Interactive Salary Prediction App")
st.markdown("Predict your salary using years of experience, education, job role, skill rating, and company tier.")

# Sidebar Inputs
st.sidebar.header("Input Parameters")

experience = st.sidebar.slider("Years of Experience", 0.0, 40.0, 2.0, 0.5)
education = st.sidebar.selectbox("Education Level", ['High School', 'Bachelor', 'Master', 'PhD'])
job_role = st.sidebar.selectbox("Job Role", ['Developer', 'Analyst', 'Manager', 'Consultant'])
skill_rating = st.sidebar.slider("Skill Rating (1-10)", 1, 10, 5)
company_tier = st.sidebar.selectbox("Company Tier", ['Tier 1', 'Tier 2', 'Tier 3'])
model_name = st.sidebar.radio("Select Model", list(models.keys()))

# Encode categorical variables
edu_map = {'High School': 0, 'Bachelor': 1, 'Master': 2, 'PhD': 3}
role_map = {'Developer': 0, 'Analyst': 1, 'Manager': 2, 'Consultant': 3}
tier_map = {'Tier 1': 0, 'Tier 2': 1, 'Tier 3': 2}

edu_code = edu_map[education]
job_code = role_map[job_role]
tier_code = tier_map[company_tier]

input_features = np.array([[experience, edu_code, job_code, skill_rating, tier_code]])
model = models[model_name]
predicted_salary = model.predict(input_features)[0]

# Display result
st.subheader("Prediction Result")
st.success(f"Estimated Salary: ${predicted_salary:,.2f}")

# Interactive Graph
st.subheader("📊 Salary vs. Years of Experience")

exp_range = np.linspace(0, 40, 100)
salary_curve = [
    model.predict(np.array([[x, edu_code, job_code, skill_rating, tier_code]]))[0]
    for x in exp_range
]

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=exp_range, y=salary_curve, mode='lines', name='Prediction Curve',
    line=dict(color='royalblue', width=3)
))
fig.add_trace(go.Scatter(
    x=[experience], y=[predicted_salary], mode='markers+text',
    name='Your Input', marker=dict(size=10, color='red'),
    text=[f"${predicted_salary:,.2f}"], textposition="top center"
))
fig.update_layout(
    title="Salary Prediction vs. Years of Experience",
    xaxis_title="Years of Experience",
    yaxis_title="Predicted Salary",
    template="plotly_white",
    height=450,
    hovermode="x unified"
)
st.plotly_chart(fig, use_container_width=True)

# Model Evaluation (Static - Sample Values)
st.subheader("📈 Model Evaluation (On Test Data)")
evaluation_data = {
    "Model": ["Linear Regression", "Random Forest"],
    "MAE": [6286.45, 6872.01],
    "RMSE": [7059.04, 7982.55],
    "R² Score": [0.90, 0.88]
}
eval_df = pd.DataFrame(evaluation_data)
st.dataframe(eval_df)

# Save to log
log_entry = {
    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    'education': education,
    'job_role': job_role,
    'experience': experience,
    'skill_rating': skill_rating,
    'company_tier': company_tier,
    'model_used': model_name,
    'predicted_salary': predicted_salary
}

log_file = 'prediction_log.csv'
if os.path.exists(log_file):
    log_df = pd.read_csv(log_file)
    log_df = pd.concat([log_df, pd.DataFrame([log_entry])], ignore_index=True)
else:
    log_df = pd.DataFrame([log_entry])

log_df.to_csv(log_file, index=False)

# Download CSV
st.subheader("Download Prediction Logs")
st.download_button("Download CSV", data=log_df.to_csv(index=False), file_name="prediction_log.csv", mime="text/csv")


# --- FOOTER ---
st.markdown("""
---
Built with ❤ using Streamlit & scikit-learn • [GitHub](https://github.com/Zubair-hussain)
""")
