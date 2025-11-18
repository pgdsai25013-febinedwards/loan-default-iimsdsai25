import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib

DATA_PATH = "C:\\Users\\evely\\OneDrive\\IIM\\chandrika\\Loan Default dataset\\Loan Default Dataset.csv"
SAVE_DIR = "C:\\Users\\evely\\OneDrive\\IIM\\chandrika\\Loan Default dataset\\loan_project\\data\\processed"

os.makedirs(SAVE_DIR, exist_ok=True)

def main():
    df = pd.read_csv(DATA_PATH)

    numeric_cols = [
        "person_age", "person_income", "person_emp_length",
        "loan_amnt", "loan_int_rate", "loan_percent_income",
        "cb_person_cred_hist_length"
    ]

    categorical_cols = [
        "person_home_ownership", "loan_intent",
        "loan_grade", "cb_person_default_on_file"
    ]

    X = df[numeric_cols + categorical_cols]
    y = df["loan_status"]

    # Handle numeric missing values
    for col in numeric_cols:
        X[col] = X[col].fillna(X[col].median())

    # Preprocessor
    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(drop="first"), categorical_cols),
        ("num", StandardScaler(), numeric_cols)
    ])

    # Fit + Transform
    X_processed = preprocessor.fit_transform(X)

    # ------------------------------
    # SAVE FEATURE NAMES (IMPORTANT!)
    # ------------------------------
    ohe = preprocessor.named_transformers_["cat"]
    cat_names = ohe.get_feature_names_out(categorical_cols)
    num_names = numeric_cols

    feature_names = list(num_names) + list(cat_names)

    np.save(f"{SAVE_DIR}/feature_names.npy", feature_names)
    # ------------------------------

    # Save preprocessor
    joblib.dump(preprocessor, f"{SAVE_DIR}/preprocessor.pkl")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Save data
    np.save(f"{SAVE_DIR}/X_train.npy", X_train)
    np.save(f"{SAVE_DIR}/X_test.npy", X_test)
    np.save(f"{SAVE_DIR}/y_train.npy", y_train)
    np.save(f"{SAVE_DIR}/y_test.npy", y_test)

    print("Preprocessing completed!")


if __name__ == "__main__":
    main()
