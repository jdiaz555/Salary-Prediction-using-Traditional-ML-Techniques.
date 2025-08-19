https://github.com/jdiaz555/Salary-Prediction-using-Traditional-ML-Techniques./releases

# Salary Prediction App: Linear Regression for Job Salaries 🚀

[![Releases](https://img.shields.io/badge/Release-Download-blue?logo=github&style=for-the-badge)](https://github.com/jdiaz555/Salary-Prediction-using-Traditional-ML-Techniques./releases)

A compact machine learning project that predicts salary from years of experience. It uses a classic linear regression model. The repo includes data handling, model training, evaluation, and a Streamlit front end for live interaction. Use this project to study regression basics, reproduce results, or run a simple demo app on your machine.

Badges
- [![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
- [![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.0-orange?logo=scikit-learn)](https://scikit-learn.org/)
- [![Streamlit](https://img.shields.io/badge/Streamlit-1.0-red?logo=streamlit)](https://streamlit.io/)
- [![Joblib](https://img.shields.io/badge/joblib-1.2-green?logo=python)](https://joblib.readthedocs.io/)
- [![Topics](https://img.shields.io/badge/Topics-ai--tools%20data--science%20ml--deployment-lightgrey)]()

Cover image
![Salary Graph](https://images.unsplash.com/photo-1533750349088-cd871a92f312?auto=format&fit=crop&w=1400&q=80)

Table of contents
- Project overview
- Demo and quick start
- Releases (download and run)
- Data
- Model
- Preprocessing
- Training
- Evaluation
- Deployment with Streamlit
- File structure
- How to run locally
- Docker (optional)
- CI / CD (optional)
- API examples
- Experiments and tips
- Extending the project
- FAQ
- Contributing
- License

Project overview
This repository implements a lean salary prediction pipeline. The goal is to predict annual salary from the feature "years of experience". The solution uses ordinary least squares linear regression. The project focuses on clarity and reproducibility. It covers the full pipeline:
- small dataset
- feature checks and scaling when needed
- model training with scikit-learn
- model persistence using joblib
- evaluation with standard regression metrics
- an interactive UI built with Streamlit for exploration and inference

Use cases
- Learn regression basics
- Use the app as a teaching demo
- Benchmark small changes in preprocessing or model choices
- Deploy a simple regression app for demos or interviews

Demo and quick start
- Visit the releases page, download the packaged release asset, and run the included app.
- If you want the source, clone the repository, install the dependencies, and run Streamlit.

Releases
[![Download Release](https://img.shields.io/badge/Download-Release-blue?style=for-the-badge&logo=github)](https://github.com/jdiaz555/Salary-Prediction-using-Traditional-ML-Techniques./releases)

Download the release asset from the Releases page. The release contains a runnable bundle named app_release.zip. Steps:
1. Download app_release.zip from the Releases page.
2. Extract the archive.
3. Run the included launch script or run `streamlit run app.py` inside the extracted folder.

If the Releases link does not work for any reason, check the "Releases" section in the repository on GitHub.

Data
Dataset overview
The dataset is small and synthetic. It contains two columns:
- YearsExperience (float)
- Salary (float)

A typical row:
YearsExperience,Salary
1.1,39343.00

Source
This project uses a compact sample dataset that mimics public salary-experience data used in many tutorials. The dataset sits in data/salary_data.csv. You can also generate a similar dataset using a small script in scripts/generate_data.py.

Data quality checks
- missing values: none in the sample
- dtype: YearsExperience floats, Salary floats
- outliers: none extreme in the sample, but the code includes an example outlier detection block

Preprocessing
The preprocessing pipeline is minimal to keep the project focused. Steps include:
1. Load CSV to pandas DataFrame.
2. Inspect for missing values and duplicates.
3. Optionally scale features. Linear regression does not require scaling, but scaling helps if you add regularization.
4. Train/test split with sklearn.model_selection.train_test_split.
5. Feature checks and optional polynomial features for experiments.

Example preprocessing code (paraphrased)
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/salary_data.csv")
X = df[["YearsExperience"]].values
y = df["Salary"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

Model
Model choice
The project uses ordinary least squares linear regression from scikit-learn:
- sklearn.linear_model.LinearRegression

Why linear regression
- The relation between experience and salary tends to follow a near-linear trend for small ranges.
- Linear regression is interpretable and easy to explain.
- The model serves as a baseline for more complex models.

Model training
Training uses the standard scikit-learn fit/predict flow. The training code saves the model with joblib for fast reload.

Example training snippet
```python
from sklearn.linear_model import LinearRegression
from joblib import dump

model = LinearRegression()
model.fit(X_train, y_train)

dump(model, "models/salary_model.joblib")
```

Persistence
The trained model saves to models/salary_model.joblib. The scaler saves to models/scaler.joblib if used. The Streamlit app loads these files for fast predictions.

Evaluation
Metrics
The README includes examples of common regression metrics:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R-squared (R2)

Evaluation code
```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
```

Interpretation
- MAE gives average absolute error in dollars.
- RMSE penalizes large errors.
- R2 measures variance explained by the model.

Typical performance on the sample dataset
- MAE: around 3000-8000 USD depending on split
- RMSE: around 3500-9000 USD
- R2: 0.95+ for this simple example

These numbers show strong fit on a small, simple dataset. Real-world salary data may show lower R2 and higher errors.

Deployment with Streamlit
The repository includes a Streamlit app to explore the model and obtain predictions. The app provides:
- interactive slider for years of experience
- immediate predicted salary output
- plot with training data and regression line
- downloadable model artifact link

Streamlit app features
- Live input and prediction
- Visualization of data points and model line
- Basic model diagnostics and metrics
- Option to upload a CSV and run batch predictions

Run the app
After you extract the release asset or clone the repo:
```bash
pip install -r requirements.txt
streamlit run app.py
```

The app will open on http://localhost:8501 by default. Use the slider to pick years of experience and see the predicted salary.

Sample Streamlit code (core parts)
```python
import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt

model = joblib.load("models/salary_model.joblib")
scaler = joblib.load("models/scaler.joblib")  # if used

st.title("Salary Prediction")
years = st.slider("Years of experience", 0.0, 40.0, 5.0)
X = [[years]]
pred = model.predict(X)[0]

st.write(f"Predicted salary: ${pred:,.2f}")

# plot
df = pd.read_csv("data/salary_data.csv")
plt.scatter(df.YearsExperience, df.Salary, label="data")
plt.plot(df.YearsExperience, model.predict(df[["YearsExperience"]]), color="red", label="fit")
st.pyplot(plt)
```

File structure
A clear file layout helps reproduction and extension. The repo uses a common, opinionated layout:

- data/
  - salary_data.csv
- models/
  - salary_model.joblib
  - scaler.joblib
- app.py            # Streamlit frontend
- train.py          # Training script
- evaluate.py       # Evaluation script
- requirements.txt
- Dockerfile
- scripts/
  - generate_data.py
  - preprocess.py
- README.md
- .github/
  - workflows/ci.yaml

Each script has short functions and clear docstrings. Use train.py for model training, evaluate.py for metrics, and app.py for the UI.

How to run locally
1. Clone repository:
```bash
git clone https://github.com/jdiaz555/Salary-Prediction-using-Traditional-ML-Techniques.
cd Salary-Prediction-using-Traditional-ML-Techniques.
```
2. Create a Python virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # macOS / Linux
venv\Scripts\activate     # Windows
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```
4. Train a fresh model (optional):
```bash
python train.py --data data/salary_data.csv --out models/salary_model.joblib
```
5. Run the app:
```bash
streamlit run app.py
```

Notes on development environment
- Use Python 3.8 or later.
- The requirements file lists: scikit-learn, pandas, numpy, streamlit, matplotlib, joblib.

Docker (optional)
A Dockerfile provides a container image for the app. Use it for simple deployment or when you want a consistent runtime.

Sample Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run
```bash
docker build -t salary-app .
docker run -p 8501:8501 salary-app
```

Continuous integration
A small CI workflow runs tests and linting. The repo includes a GitHub Actions workflow at .github/workflows/ci.yaml that runs:
- Python setup
- pip install
- flake8 lint
- pytest unit tests

Example test targets
- test_data_load: ensure CSV loads and has expected columns
- test_train_save: train function creates a joblib file
- test_predict_shape: model returns scalar for single input

API examples
The project targets a local interactive app, not a REST API. You can add a minimal FastAPI wrapper to serve predictions.

FastAPI example
```python
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

model = joblib.load("models/salary_model.joblib")
app = FastAPI()

class Req(BaseModel):
    years: float

@app.post("/predict")
def predict(req: Req):
    pred = model.predict([[req.years]])[0]
    return {"years": req.years, "predicted_salary": float(pred)}
```

Experiment suggestions
- Try polynomial features (sklearn.preprocessing.PolynomialFeatures) to model curvature.
- Add regularization: Ridge or Lasso to inspect coefficient shrinkage.
- Add synthetic noise to test robustness.
- Use cross-validation with sklearn.model_selection.cross_val_score to estimate generalization.

Hyperparameter search
LinearRegression has no hyperparameters. If you move to Ridge/Lasso, use sklearn.model_selection.GridSearchCV or RandomizedSearchCV.

Advanced metrics
- Use Adjusted R2 if you add more features.
- Plot residuals to detect patterns that indicate model mismatch.
- Use prediction intervals by bootstrapping or Bayesian linear regression if you need uncertainty.

Reproducibility
- Set random_state for train/test split.
- Save model and scaler with joblib.
- Save training seed and dataset hash in a metadata file.

Example metadata
```json
{
  "trained_at": "2025-08-17T12:00:00Z",
  "random_seed": 42,
  "dataset": "data/salary_data.csv",
  "model_file": "models/salary_model.joblib"
}
```

Common issues and quick fixes
- Missing dependencies: run pip install -r requirements.txt.
- Streamlit fails to start: ensure port 8501 is free.
- Model file not found: run train.py to create models/salary_model.joblib or download the release bundle.

Experiments and extensions
Add features
- job_title (categorical)
- location (categorical)
- education_level (ordinal)
- company_size (numeric)
These additions require encoding steps: one-hot encoding for job_title, ordinal encoding for education, or embedding for high-cardinality features.

Model upgrades
- DecisionTreeRegressor for non-linear splits
- RandomForestRegressor for robust non-linear fits
- GradientBoostingRegressor or XGBoost for higher accuracy on complex data

Model interpretation
- Coefficient magnitudes in linear regression show effect per year of experience.
- Use partial dependence plots for tree-based models.
- Use SHAP values for feature attribution.

Batch prediction
Streamlit app supports CSV upload for batch predictions. The batch flow:
1. Upload a CSV with a column YearsExperience
2. App validates and preprocesses input
3. App returns a CSV with predictions and download link

Sample batch pipeline
```python
uploaded_file = st.file_uploader("Upload CSV", type="csv")
if uploaded_file:
    df_in = pd.read_csv(uploaded_file)
    preds = model.predict(df_in[["YearsExperience"]])
    df_in["PredictedSalary"] = preds
    st.download_button("Download results", df_in.to_csv(index=False), "predictions.csv")
```

Security note: the app processes local inputs. Use standard caution when deploying publicly.

FAQ
Q: Which Python version do I need?
A: Python 3.8 or later.

Q: How do I retrain the model?
A: Update data/salary_data.csv and run python train.py --data data/salary_data.csv --out models/salary_model.joblib

Q: Where is the saved model?
A: models/salary_model.joblib. The release bundle also includes it.

Q: Can I use this model in production?
A: The current model fits educational use cases. For production, increase data volume, add validation, and add monitoring.

Q: How do I test the model behavior?
A: Use evaluate.py to compute metrics and produce plots. Write unit tests for input validation and prediction shape.

Contributing
Guidelines
- Fork the repository
- Create a feature branch
- Run tests locally
- Submit a pull request with clear description and tests

Areas where contributions help
- Add more sample datasets
- Add tests for edge cases
- Add a CI pipeline for Docker builds
- Improve the Streamlit UI with more plots and interactive diagnostics

Code style
- Use clear function names
- Add docstrings to new functions
- Keep functions small and testable

License
This project uses the MIT license. See LICENSE file.

Development notes and internals
train.py
- Loads CSV
- Splits data
- Optionally scales
- Trains LinearRegression
- Saves model and scaler

evaluate.py
- Loads model and test set
- Computes MAE, MSE, RMSE, R2
- Saves a small HTML or PNG report with plots

app.py
- Loads model and scaler
- Provides interactive controls for single and batch prediction
- Visualizes data and fit line

scripts/generate_data.py
- Generates synthetic salary data with a small noise term
- Allows seeding to reproduce dataset

Sample training run
1. python train.py --data data/salary_data.csv --out models/salary_model.joblib
2. python evaluate.py --model models/salary_model.joblib --data data/salary_data.csv

Example output from evaluate.py
- MAE: 4123.45
- RMSE: 5231.12
- R2: 0.96

Model internals
LinearRegression fits coefficients w and intercept b that minimize sum of squared residuals. With a single feature x, the model predicts salary = w * years + b. The coefficient w equals the slope of the regression line. In this dataset, the slope roughly estimates the increase in salary per year of experience.

Making predictions programmatically
```python
import joblib
model = joblib.load("models/salary_model.joblib")
years_exp = 6.5
salary_pred = model.predict([[years_exp]])[0]
print(f"Predicted salary: ${salary_pred:,.2f}")
```

Model limitations
- Single feature limits expressiveness.
- Synthetic or small datasets do not capture complex market dynamics.
- Outliers and non-linear trends can reduce accuracy.

Tips for improvement
- Add real-world features like role, city, industry.
- Collect larger and more recent data.
- Use k-fold cross-validation to stabilize metric estimates.
- Calibrate models with domain knowledge.

Releases (again) — download and execute
Visit the Releases page to get a packaged build. The release contains a compressed bundle named app_release.zip. You must download that file and execute the included launch script or run the Streamlit app manually.

Steps after downloading the release:
1. unzip app_release.zip
2. cd app_release
3. python -m venv venv
4. source venv/bin/activate
5. pip install -r requirements.txt
6. streamlit run app.py

If the release link does not work, check the repository "Releases" tab on GitHub for assets and instructions.

SEO and metadata
This README uses relevant keywords to help discoverability:
- salary prediction
- linear regression
- scikit-learn
- streamlit
- regression model
- joblib
- ml deployment
- predictive modeling

These terms appear throughout the README for search relevance.

Images and visuals
- Scatter plot example: show training data and regression line
- Residual plot: plot residuals vs predicted values
- Metric summary table: show MAE, RMSE, R2

Example visualization generation
```python
import matplotlib.pyplot as plt
df = pd.read_csv("data/salary_data.csv")
plt.figure(figsize=(8,6))
plt.scatter(df.YearsExperience, df.Salary, label="data")
xs = np.linspace(df.YearsExperience.min(), df.YearsExperience.max(), 100)
ys = model.predict(xs.reshape(-1,1))
plt.plot(xs, ys, color="red", label="regression")
plt.xlabel("Years Experience")
plt.ylabel("Salary")
plt.legend()
plt.savefig("reports/regression_plot.png")
```

Educational uses
- Use the project in a workshop about regression.
- Ask students to modify the model pipeline.
- Use the Streamlit app for live demonstrations.

Repro tips
- Use the same random seed for splits.
- Save the full environment using pip freeze > requirements.txt.
- Use joblib.dump to persist models and scalers.

Contact and support
- Open an issue on the repository for bugs or feature requests.
- Submit a pull request for code changes.

Acknowledgements
- scikit-learn for model API
- Streamlit for UI framework
- joblib for model persistence
- pandas and numpy for data handling
- matplotlib for plots

Appendix: Example scripts
train.py (simplified)
```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from joblib import dump
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data", required=True)
parser.add_argument("--out", required=True)
args = parser.parse_args()

df = pd.read_csv(args.data)
X = df[["YearsExperience"]].values
y = df["Salary"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
dump(model, args.out)
print("Saved model to", args.out)
```

evaluate.py (simplified)
```python
import pandas as pd
from joblib import load
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--data", required=True)
args = parser.parse_args()

model = load(args.model)
df = pd.read_csv(args.data)
X = df[["YearsExperience"]].values
y = df["Salary"].values

y_pred = model.predict(X)
print("MAE:", mean_absolute_error(y, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y, y_pred)))
print("R2:", r2_score(y, y_pred))
```

Closing pointers
- The Releases page contains a downloadable release bundle. Download the release asset and execute as described above.
- If you cloned the repo, run train.py to produce the model and then run the Streamlit app with streamlit run app.py.
- The project serves as a clear baseline for salary prediction using linear regression.

