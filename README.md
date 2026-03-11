# Customer Churn Prediction ML Pipeline

End-to-End Machine Learning Pipeline for Predicting Telecom Customer Churn.

This project builds an end-to-end machine learning pipeline to predict whether a telecom customer is likely to churn.

The model is trained on the Telco Customer Churn dataset and demonstrates the complete machine learning workflow from data preprocessing to model deployment preparation.


---

## Objective

To build a complete machine learning pipeline that predicts customer churn using historical telecom customer data.

---

## Dataset

Telco Customer Churn Dataset

Records: 7043 customers  
Features: 21 attributes

Key features include:

• Customer demographics  
• Account information  
• Services subscribed  
• Billing details  
• Contract type  

Target Variable:

`Churn` → Whether the customer left the service.

---

## Tech Stack

Python  
Pandas  
NumPy  
Scikit-learn  
Matplotlib  
Seaborn  
Joblib  
Git & GitHub  

---

## Project Workflow

1. Data Loading  
2. Exploratory Data Analysis (EDA)  
3. Data Cleaning  
4. Feature Encoding  
5. Train-Test Split  
6. Model Training  
7. Model Evaluation  
8. Model Saving  

---

## Model Used

Logistic Regression

---

## Result

Model Accuracy: **~82%**

The model successfully predicts customer churn using telecom service usage and billing patterns.

---

## Project Structure
customer-churn-ml-pipeline/
│
├── data/ # Raw dataset
├── models/ # Saved trained model
├── src/ # Source code
│ ├── data_preprocessing.py
│ ├── train_model.py
│ └── evaluate_model.py
│
├── notebooks/ # Experiment notebooks
├── requirements.txt
└── README.md


---

## How to Run

Install dependencies
pip install -r requirements.txt

Run preprocessing
python src/data_preprocessing.py


Train model
python src/train_model.py


Evaluate model
python src/evaluate_model.py


---

## Future Improvements

• Hyperparameter tuning  
• Feature scaling  
• Trying advanced models (Random Forest, XGBoost)  
• Deploying the model using Flask or FastAPI  

---

## Author

Bhushan Patil  
AI / Machine Learning Engineer  