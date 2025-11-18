import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import joblib
from tensorflow.keras.models import load_model

DATA_DIR = r"C:\Users\evely\OneDrive\IIM\chandrika\Loan Default dataset\loan_project\data\processed"
MODEL_DIR = r"C:\Users\evely\OneDrive\IIM\chandrika\Loan Default dataset\loan_project\models"

# Load data
X_test = np.load(f"{DATA_DIR}\\X_test.npy")
y_test = np.load(f"{DATA_DIR}\\y_test.npy")

# Load models
logreg = joblib.load(f"{MODEL_DIR}\\logreg_model.pkl")
xgb = joblib.load(f"{MODEL_DIR}\\xgboost_model.pkl")
nn = load_model(f"{MODEL_DIR}\\best_model.h5")

# Predict probabilities
logreg_prob = logreg.predict_proba(X_test)[:, 1]
xgb_prob = xgb.predict_proba(X_test)[:, 1]
nn_prob = nn.predict(X_test, verbose=0)

# Compute ROC
fpr_lr, tpr_lr, _ = roc_curve(y_test, logreg_prob)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_prob)
fpr_nn, tpr_nn, _ = roc_curve(y_test, nn_prob)

auc_lr = roc_auc_score(y_test, logreg_prob)
auc_xgb = roc_auc_score(y_test, xgb_prob)
auc_nn = roc_auc_score(y_test, nn_prob)

# Plot ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC={auc_lr:.3f})")
plt.plot(fpr_xgb, tpr_xgb, label=f"XGBoost (AUC={auc_xgb:.3f})")
plt.plot(fpr_nn, tpr_nn, label=f"Neural Network (AUC={auc_nn:.3f})")
plt.plot([0,1],[0,1],'k--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison: LR vs ANN vs XGBoost")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(f"{MODEL_DIR}\\roc_comparison.png")
plt.show()

print("\nROC curve saved as roc_comparison.png")
