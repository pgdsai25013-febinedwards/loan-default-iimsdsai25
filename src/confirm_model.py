import numpy as np
from tensorflow.keras.models import load_model

DATA_DIR = r"C:\Users\evely\OneDrive\IIM\chandrika\Loan Default dataset\loan_project\data\processed"
MODEL_PATH = r"C:\Users\evely\OneDrive\IIM\chandrika\Loan Default dataset\loan_project\models\best_model.h5"

model = load_model(MODEL_PATH)
X_test = np.load(f"{DATA_DIR}\\X_test.npy")

print("Predict single row:", model.predict(X_test[:1], verbose=0).shape)
print("Predict 200 rows:", model.predict(X_test[:200], verbose=0).shape)
