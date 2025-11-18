import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib

DATA_DIR = r"C:\Users\evely\OneDrive\IIM\chandrika\Loan Default dataset\loan_project\data\processed"
MODEL_DIR = r"C:\Users\evely\OneDrive\IIM\chandrika\Loan Default dataset\loan_project\models"

os.makedirs(MODEL_DIR, exist_ok=True)

# Load data
X_train = np.load(f"{DATA_DIR}\\X_train.npy")
y_train = np.load(f"{DATA_DIR}\\y_train.npy")
X_test = np.load(f"{DATA_DIR}\\X_test.npy")
y_test = np.load(f"{DATA_DIR}\\y_test.npy")

# Train Logistic Regression
model = LogisticRegression(max_iter=500, class_weight="balanced")
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Evaluation
print("\n=== Logistic Regression Performance ===")
print("AUC:", roc_auc_score(y_test, y_prob))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model
joblib.dump(model, f"{MODEL_DIR}\\logreg_model.pkl")
print("\nLogistic Regression model saved!")
