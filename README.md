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

- Problem Statement
- Objective
- Dataset
- Technologies Used
- Machine Learning Models
- Project Workflow
- Evaluation Metrics
- Model Performance
- Project Structure
- Results
- How to Run
- Future Improvements
- Author

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

The dataset contains telecom customer information including demographics, services subscribed, and billing details.

Example features:

- tenure
- MonthlyCharges
- TotalCharges
- Contract
- PaymentMethod
- InternetService
- Churn

Target Variable:

```
Churn
```

```
0 → Customer stays
1 → Customer leaves
```

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

Logistic Regression is commonly used as a baseline model for binary classification problems.

---

## Project Workflow

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

The following metrics are used to evaluate model performance:

- Accuracy  
- Precision  
- Recall  
- F1 Score  
- Confusion Matrix  
- Classification Report  

These metrics provide insight into how effectively the model predicts customer churn.

---

## Model Performance

| Model | Accuracy | Precision | Recall | F1 Score |
|------|---------|----------|-------|---------|
| Logistic Regression | ~0.80 | ~0.78 | ~0.74 | ~0.76 |

*Performance values may vary slightly depending on dataset split.*

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

The machine learning model successfully predicts telecom customer churn based on behavioral and account-related features.

The model helps identify customers likely to leave the service, allowing companies to take preventive actions to improve customer retention.

---

## How to Run the Project

Install dependencies

```
pip install -r requirements.txt
```

Run data preprocessing

```
python src/data_preprocessing.py
```

Train the model

```
python src/train_model.py
```

Evaluate the model

```
python src/evaluate_model.py
```

---

## Future Improvements

- Use ensemble models such as Random Forest and XGBoost  
- Hyperparameter tuning using GridSearchCV  
- Feature importance analysis  
- Model performance visualization  
- Deploying the model using FastAPI or Flask  

---

## Author

Bhushan Patil  
AI / Machine Learning Engineer  
Pune, Maharashtra, India