import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import roc_auc_score

DATA_DIR = r"C:\Users\evely\OneDrive\IIM\chandrika\Loan Default dataset\loan_project\data\processed"
MODEL_PATH = r"C:\Users\evely\OneDrive\IIM\chandrika\Loan Default dataset\loan_project\models\best_model.h5"

# Load data
X_test = np.load(f"{DATA_DIR}\\X_test.npy")
y_test = np.load(f"{DATA_DIR}\\y_test.npy")
feature_names = np.load(f"{DATA_DIR}\\feature_names.npy", allow_pickle=True)

# Load model
model = load_model(MODEL_PATH)

# Baseline model performance
baseline_pred = model.predict(X_test, verbose=0)
baseline_auc = roc_auc_score(y_test, baseline_pred)

print(f"Baseline AUC: {baseline_auc:.4f}\n")

importances = []

# Loop over each feature and permute
for i in range(X_test.shape[1]):
    X_permuted = X_test.copy()
    np.random.shuffle(X_permuted[:, i])   # shuffle ONE feature
    
    perm_pred = model.predict(X_permuted, verbose=0)
    perm_auc = roc_auc_score(y_test, perm_pred)
    
    importance = baseline_auc - perm_auc
    importances.append(importance)

# Convert to array
importances = np.array(importances)

# Sort by importance
sorted_idx = np.argsort(importances)[::-1]

print("=== FEATURE IMPORTANCE (Permutation) ===\n")
for idx in sorted_idx:
    print(f"{feature_names[idx]} : {importances[idx]:.4f}")
