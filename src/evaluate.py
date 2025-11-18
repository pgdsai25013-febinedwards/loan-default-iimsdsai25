import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

DATA_DIR = "C:\\Users\\evely\\OneDrive\\IIM\\chandrika\\Loan Default dataset\\loan_project\\data\\processed"
MODEL_PATH = "C:\\Users\\evely\\OneDrive\\IIM\\chandrika\\Loan Default dataset\\loan_project\\models\\best_model.h5"

# Load data
X_test = np.load(f"{DATA_DIR}\\X_test.npy")
y_test = np.load(f"{DATA_DIR}\\y_test.npy")

# Load model
model = load_model(MODEL_PATH)

# Prediction
y_prob = model.predict(X_test).flatten()
y_pred = (y_prob >= 0.5).astype(int)

print("=== CONFUSION MATRIX ===")
print(confusion_matrix(y_test, y_pred))

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred))

print("\n=== ROC-AUC SCORE ===")
print(roc_auc_score(y_test, y_prob))
