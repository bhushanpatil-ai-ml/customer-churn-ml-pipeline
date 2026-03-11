import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    df = df.drop("customerID", axis=1)
    return df


def encode_data(df: pd.DataFrame) -> pd.DataFrame:
    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})
    df = pd.get_dummies(df, drop_first=True)
    return df


def main() -> None:
    file_path = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"

    df = load_data(file_path)
    df = clean_data(df)
    df = encode_data(df)

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("Model Accuracy:", accuracy)

    joblib.dump(model, "models/logistic_regression_model.pkl")
    print("Model saved successfully in models/logistic_regression_model.pkl")


if __name__ == "__main__":
    main()