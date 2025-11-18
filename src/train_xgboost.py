import numpy as np
import os
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib

DATA_DIR = r"C:\Users\evely\OneDrive\IIM\chandrika\Loan Default dataset\loan_project\data\processed"
MODEL_DIR = r"C:\Users\evely\OneDrive\IIM\chandrika\Loan Default dataset\loan_project\models"

os.makedirs(MODEL_DIR, exist_ok=True)

# Load data
X_train = np.load(f"{DATA_DIR}\\X_train.npy")
y_train = np.load(f"{DATA_DIR}\\y_train.npy")
X_test = np.load(f"{DATA_DIR}\\X_test.npy")
y_test = np.load(f"{DATA_DIR}\\y_test.npy")

# Train XGBoost
model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric='logloss',
    scale_pos_weight=3  # handle imbalance
)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Evaluation
print("\n=== XGBoost Performance ===")
print("AUC:", roc_auc_score(y_test, y_prob))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model
joblib.dump(model, f"{MODEL_DIR}\\xgboost_model.pkl")
print("\nXGBoost model saved!")
