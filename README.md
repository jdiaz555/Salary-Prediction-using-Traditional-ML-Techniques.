
# Salary Prediction using Traditional ML Techniques

This project demonstrates how to predict employee salaries using machine learning regression techniques. By analyzing years of experience, the model can accurately forecast potential salary, making it a powerful tool for HR, recruiters, and career planning platforms.

---

##  Project Overview

This is a complete end-to-end machine learning workflow built using:

- **Pandas & Numpy** for data handling  
- **Matplotlib & Seaborn** for visualization  
- **Scikit-learn** for modeling  
- **Streamlit** for deployment  

Users can enter their years of experience and instantly get a predicted salary value through an interactive web app.

---

##  Dataset

- **Source:** [Kaggle Dataset](https://www.kaggle.com/datasets/korpionn/salary-prediction-dataset)
- **Columns:**
  - `YearsExperience` – Number of years the employee has worked
  - `Salary` – Annual salary in USD

---

##  Features & Workflow

-  Cleaned and explored dataset  
-  Trained a **Linear Regression** model  
-  Evaluated with MAE, RMSE, and R² Score  
-  Built a responsive **Streamlit web app**  
-  Deployed the model and web app using **Render**  
-  All code and assets organized for GitHub and Colab  

---

##  Model Evaluation

| Metric      | Score     |
|-------------|-----------|
| MAE         | 6,286.45  |
| RMSE        | 7,059.04  |
| R² Score    | 0.90      |

A strong R² score of 0.90 means the model explains 90% of the variance in salary using only one feature — **Years of Experience**.

---

##  Streamlit Web App

> 🔗 **Live Demo:** [https://your-render-link.com](https://your-render-link.com)  


###  App Features:
- Clean modern UI with Streamlit  
- Sidebar with project description  
- Real-time prediction and feedback  
- Progress bar + animation effects  
- Mobile responsive

---

## Repository Structure

```

salary-prediction-ml/
├── app.py                   # Streamlit app code
├── salary\_predictor.pkl     # Trained model
├── requirements.txt         # Python dependencies
├── README.md                # Project overview
└── .gitignore

````

---

## How to Run Locally

```bash
# Clone the repo
git clone https://github.com/your-username/salary-prediction-ml.git
cd salary-prediction-ml

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
````

---

## Google Colab Notebook

Click below to explore the full model training and evaluation notebook:

 [Open in Google Colab](https://colab.research.google.com/drive/1ObI1yeyQ3ar5oUDO7hXj0g6UewGWzTr8?usp=sharing)

---

##  Credits

Developed by **Syed Zubair Hussain Shah**
[Portfolio](https://zubair-hussain-shah.vercel.app/) • [LinkedIn](https://linkedin.com/in/syed-zubair-hussain-shah)

---

```


Let me know if you want help **deploying to Render** or writing your first GitHub commit message.
```
