import shap
import numpy as np
from tensorflow.keras.models import load_model

# Paths
DATA_DIR = r"C:\Users\evely\OneDrive\IIM\chandrika\Loan Default dataset\loan_project\data\processed"
MODEL_PATH = r"C:\Users\evely\OneDrive\IIM\chandrika\Loan Default dataset\loan_project\models\best_model.h5"

# Load data
X_test = np.load(f"{DATA_DIR}\\X_test.npy")
feature_names = np.load(f"{DATA_DIR}\\feature_names.npy", allow_pickle=True)

print("X_test shape:", X_test.shape)
print("Feature names:", len(feature_names))

# Load model
model = load_model(MODEL_PATH)

# --- SHAP FIX: Use GradientExplainer ---
explainer = shap.GradientExplainer(model, X_test[:200])     # small background

# Compute SHAP values for 300 rows
shap_values = explainer.shap_values(X_test[:300])

print("SHAP values shape:", np.array(shap_values).shape)

# Correct shape for summary plot
vals = shap_values[0] if isinstance(shap_values, list) else shap_values

# Summary plot
shap.summary_plot(vals, X_test[:300], feature_names=feature_names)