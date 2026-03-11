# Customer Churn ML Pipeline

This project predicts whether a telecom customer is likely to churn using machine learning.

## Objective
To build an end-to-end machine learning pipeline for customer churn prediction using the Telco Customer Churn dataset.

## Tech Stack
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib

## Project Workflow
1. Data loading
2. Exploratory Data Analysis (EDA)
3. Data cleaning
4. Feature encoding
5. Train-test split
6. Model training
7. Model evaluation
8. Model saving

## Model Used
- Logistic Regression

## Result
- Achieved approximately 82% accuracy on the test set

## Project Structure
customer-churn-ml-pipeline/
- data/ → raw dataset
- src/ → preprocessing, training, evaluation code
- models/ → saved machine learning model
- notebooks/ → future experiments
- README.md → project documentation

## How to Run
```bash
python src/data_preprocessing.py
python src/train_model.py
python src/evaluate_model.py