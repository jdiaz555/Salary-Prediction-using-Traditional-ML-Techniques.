import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import os
import datetime
import time

# Set Streamlit page config
st.set_page_config(
    page_title="Salary Prediction App",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load models
@st.cache_resource
def load_models():
    models = {}
    try:
        with open("random_forest_model.pkl", "rb") as f:
            models["Random Forest"] = pickle.load(f)
    except Exception as e:
        st.warning(f"⚠️ Could not load Random Forest model: {e}")

    try:
        with open("linear_regression_model.pkl", "rb") as f:
            models["Linear Regression"] = pickle.load(f)
    except Exception as e:
        st.warning(f"⚠️ Could not load Linear Regression model: {e}")

    try:
        with open("salary_predictor.pkl", "rb") as f:
            models["Default"] = pickle.load(f)
    except Exception as e:
        st.warning(f"⚠️ Could not load Default model: {e}")

    if not models:
        st.error("❌ No models loaded. Please verify your .pkl files.")
    return models

models = load_models()

# Sidebar inputs
with st.sidebar:
    st.header("User Input")
    experience = st.slider("Years of Experience", 0.0, 40.0, 2.0, 0.5)
    model_name = st.selectbox("Select Model", list(models.keys()))

# Main content
st.markdown("<h1 style='text-align: center;'>Salary Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Enter your experience and choose a model to estimate your salary.</p>", unsafe_allow_html=True)
st.markdown("---")

# Predict salary
if st.button("Predict Salary"):
    model = models[model_name]
    input_data = np.array([[experience]])
    prediction = model.predict(input_data)[0]

    col1, col2 = st.columns(2)
    col1.metric(label="Predicted Salary", value=f"${prediction:,.2f}")
    col2.metric(label="Years of Experience", value=f"{experience} years")

    # Save log
    log_entry = {
        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'model_used': model_name,
        'experience': experience,
        'predicted_salary': prediction
    }

    log_file = 'prediction_log.csv'
    if os.path.exists(log_file):
        log_df = pd.read_csv(log_file)
        log_df = pd.concat([log_df, pd.DataFrame([log_entry])], ignore_index=True)
    else:
        log_df = pd.DataFrame([log_entry])
    log_df.to_csv(log_file, index=False)

    st.success("Prediction logged successfully!")

    # Visualization
    st.markdown("### Salary Trend Visualization")
    x_vals = np.linspace(0, 40, 100).reshape(-1, 1)
    y_vals = model.predict(x_vals)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_vals.flatten(), y=y_vals, mode='lines', name='Prediction Trend'))
    fig.add_trace(go.Scatter(x=[experience], y=[prediction], mode='markers',
                             name='Your Prediction', marker=dict(size=10, color='red')))
    fig.update_layout(title="Salary vs. Experience",
                      xaxis_title="Years of Experience",
                      yaxis_title="Predicted Salary",
                      height=450)
    st.plotly_chart(fig, use_container_width=True)

# Model metrics (optional)
with st.expander("Model Evaluation Metrics"):
    st.markdown("""
    | Model              | MAE     | RMSE    | R² Score |
    |--------------------|---------|---------|----------|
    | Linear Regression  | 6,286.45| 7,059.04| 0.90     |
    | Random Forest      | 6,872.01| 7,982.55| 0.88     |
    """)

# View log
with st.expander("View Prediction Logs"):
    if os.path.exists("prediction_log.csv"):
        log_data = pd.read_csv("prediction_log.csv")
        st.dataframe(log_data.tail(20), use_container_width=True)
        st.download_button("⬇ Download Full Log as CSV", data=log_data.to_csv(index=False), file_name="prediction_log.csv", mime="text/csv")
    else:
        st.info("No predictions logged yet.")
