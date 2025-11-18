from tensorflow.keras.models import load_model

MODEL_PATH = r"C:\Users\evely\OneDrive\IIM\chandrika\Loan Default dataset\loan_project\models\best_model.h5"

model = load_model(MODEL_PATH)

print("Model input shape:", model.input_shape)
print("Model summary:")
model.summary()
