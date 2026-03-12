# Customer Churn Prediction

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-green)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-orange)
![Classification](https://img.shields.io/badge/Model-Classification-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

End-to-End Machine Learning Pipeline for Predicting Telecom Customer Churn using classification models.

This project builds a complete machine learning workflow to predict whether a telecom customer is likely to churn (leave the service). The pipeline includes data preprocessing, feature engineering, model training, evaluation, and model saving.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Objective](#objective)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Machine Learning Models](#machine-learning-models)
- [Project Workflow](#project-workflow)
- [Evaluation Metrics](#evaluation-metrics)
- [Project Structure](#project-structure)
- [Results](#results)
- [How to Run](#how-to-run-the-project)
- [Future Improvements](#future-improvements)
- [Author](#author)

---

## Problem Statement

Customer churn is a major challenge for telecom companies. Acquiring new customers is significantly more expensive than retaining existing ones. Predicting churn helps businesses identify customers who are likely to leave and take proactive steps to retain them.

Machine learning techniques can analyze customer behavior and identify patterns that indicate potential churn.

---

## Objective

The main objectives of this project are:

- Build an end-to-end machine learning pipeline for churn prediction  
- Perform data preprocessing and feature engineering  
- Train classification models for churn prediction  
- Evaluate model performance using standard metrics  
- Save the trained model for future predictions  

---

## Dataset

The dataset used in this project contains telecom customer information including demographics, services subscribed, and billing information.

Typical dataset features include:

- Customer demographics
- Service subscriptions
- Account information
- Billing details
- Contract type
- Tenure
- Monthly charges
- Total charges

Target Variable:

Churn

```
0 → Customer stays
1 → Customer leaves
```

The dataset allows the model to learn patterns that indicate whether a customer will churn.

---

## Technologies Used

### Programming
Python

### Data Science Libraries
Pandas  
NumPy  
Matplotlib  
Seaborn  

### Machine Learning
Scikit-learn

### Tools
Git  
GitHub  
Joblib  

---

## Machine Learning Models

This project uses classification algorithms to predict churn:

- Logistic Regression

Logistic Regression serves as a strong baseline model for binary classification problems such as churn prediction.

---

## Project Workflow

The machine learning pipeline follows these steps:

1. Data Loading  
2. Exploratory Data Analysis (EDA)  
3. Data Cleaning and Preprocessing  
4. Feature Encoding  
5. Train-Test Split  
6. Model Training  
7. Model Evaluation  
8. Model Saving  

---

## Evaluation Metrics

To measure model performance, the following metrics are used:

- Accuracy  
- Precision  
- Recall  
- F1 Score  
- Confusion Matrix  
- Classification Report  

These metrics provide insight into how well the model identifies churn customers.

---

## Project Structure

```
customer-churn-ml-pipeline/
│
├── data/                    # Dataset folder
│
├── models/                  # Saved trained models
│
├── src/                     # Source code
│   ├── data_preprocessing.py
│   ├── train_model.py
│   └── evaluate_model.py
│
├── notebooks/               # Future experimentation notebooks
│
├── requirements.txt         # Project dependencies
│
└── README.md                # Project documentation
```

---

## Results

The machine learning model successfully predicts telecom customer churn based on customer behavior and account-related features.

The trained model can help telecom companies identify customers who are likely to leave and take preventive actions to improve customer retention.

---

## How to Run the Project

### Install dependencies

pip install -r requirements.txt

### Run data preprocessing

python src/data_preprocessing.py

### Train the model

python src/train_model.py

### Evaluate the model

python src/evaluate_model.py

---

## Future Improvements

Possible improvements for this project include:

- Using ensemble models such as Random Forest and XGBoost  
- Hyperparameter tuning using GridSearchCV  
- Feature importance analysis  
- Model performance visualization  
- Deploying the model using FastAPI or Flask  

---

## Author

Bhushan Patil  
AI / Machine Learning Engineer  
Pune, Maharashtra, India