
#  Salary Prediction using Traditional ML Techniques

This project demonstrates how to predict employee salaries using machine learning regression techniques. By analyzing years of experience, the model can forecast potential salary. It includes an interactive web app built with Streamlit and allows switching between two trained ML models.

---

##  Project Overview

An end-to-end machine learning solution built using:

- **Pandas & Numpy** for data handling  
- **Matplotlib & Seaborn** for visualization  
- **Scikit-learn** for training models  
- **Streamlit** for deployment with an interactive UI  

Users can input their years of experience and choose between **Linear Regression** and **Random Forest** models to get a predicted salary.

---

##  Dataset

- **Source:** [Kaggle Dataset](https://www.kaggle.com/datasets/korpionn/salary-prediction-dataset)
- **Columns:**
  - `YearsExperience` – Number of years the employee has worked
  - `Salary` – Annual salary in USD

---

##  Features & Workflow

-  Dataset cleaning and visualization  
-  Trained two regression models  
- Linear Regression  
- Random Forest Regressor  
- Evaluation using MAE, RMSE, R² Score  
- Streamlit app with model selection  
- Deployed via Streamlit Cloud

---

##  Model Evaluation

| Model              | MAE     | RMSE    | R² Score |
|--------------------|---------|---------|----------|
| Linear Regression  | 6,286.45| 7,059.04| **0.90** |
| Random Forest      | 6,872.01| 7,982.55| 0.88     |

🔎 **Linear Regression** performed better and is used as the default model in `salary_predictor.pkl`.

---

##  Streamlit Web App

>  **Live Demo:** https://salary-prediction-using-traditional-ml-techniques-d5mdi8bw6iu5.streamlit.app/

###  App Features:
- Clean modern UI with sidebar
- Model selector: Linear Regression or Random Forest
- Real-time salary prediction
- Visual feedback: progress bar + balloons

---

##  Repository Structure

```

salary-prediction-ml/
├── app.py                         # Streamlit app
├── salary\_predictor.pkl           # Default model (Linear Regression)
├── linear\_regression\_model.pkl    # Saved LR model
├── random\_forest\_model.pkl        # Saved RF model
├── requirements.txt               # Project dependencies
├── README.md                      # Project overview
└── .gitignore

````

---

##  How to Run Locally

```
# Clone the repo
git clone https://github.com/your-username/salary-prediction-ml.git
cd salary-prediction-ml

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
````

---

##  Google Colab Notebook

👉 [Open in Google Colab](https://colab.research.google.com/drive/1ObI1yeyQ3ar5oUDO7hXj0g6UewGWzTr8?usp=sharing)

---

## Credits

Developed by **Syed Zubair Hussain Shah**
[🌐 Portfolio](https://zubair-hussain-shah.vercel.app/) • [🔗 LinkedIn](https://www.linkedin.com/in/syed-zubair-hussain-shah-491294376?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app)


```
