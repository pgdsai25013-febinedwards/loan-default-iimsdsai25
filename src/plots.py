import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
from tensorflow.keras.models import load_model
import seaborn as sns

DATA_DIR = "C:\\Users\\evely\\OneDrive\\IIM\\chandrika\\Loan Default dataset\\loan_project\\data\\processed"
MODEL_PATH = "C:\\Users\\evely\\OneDrive\\IIM\\chandrika\\Loan Default dataset\\loan_project\\models\\best_model.h5"

X_test = np.load(f"{DATA_DIR}\\X_test.npy")
y_test = np.load(f"{DATA_DIR}\\y_test.npy")

model = load_model(MODEL_PATH)
y_prob = model.predict(X_test).flatten()
y_pred = (y_prob >= 0.5).astype(int)

# ===== CONFUSION MATRIX =====
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ===== ROC CURVE =====
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0,1], [0,1], '--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()
