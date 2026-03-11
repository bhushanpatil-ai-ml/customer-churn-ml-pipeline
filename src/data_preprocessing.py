import pandas as pd


def load_data(file_path: str) -> pd.DataFrame:
    """Load churn dataset from CSV file."""
    df = pd.read_csv(file_path)
    return df


def clean_data(df):
    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Fill missing TotalCharges values with median
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # Drop customerID (not useful for prediction)
    df.drop("customerID", axis=1, inplace=True)

    return df


def encode_data(df):
    # Convert target column Churn into 0 and 1
    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

    # One-hot encode all categorical columns except target
    df = pd.get_dummies(df, drop_first=True)

    return df


def main() -> None:
    file_path = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    df = load_data(file_path)

    print("Dataset loaded successfully!")
    print("\nFirst 5 rows:")
    print(df.head())

    print("\nDataset shape:")
    print(df.shape)

    print("\nDataset info:")
    print(df.info())

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nBasic statistics:")
    print(df.describe())

    print("\nCleaning dataset...")
    df = clean_data(df)

    print("\nCleaned dataset shape:")
    print(df.shape)

    print("\nEncoding categorical features...")
    df = encode_data(df)

    print("\nEncoded dataset shape:")
    print(df.shape)

    print("\nFirst 5 rows after encoding:")
    print(df.head())


if __name__ == "__main__":
    main()