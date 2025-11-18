import numpy as np
from model import build_model
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os

# Load processed data
DATA_DIR = "C:\\Users\\evely\\OneDrive\\IIM\\chandrika\\Loan Default dataset\\loan_project\\data\\processed"
MODEL_DIR = "C:\\Users\\evely\\OneDrive\\IIM\\chandrika\\Loan Default dataset\\loan_project\\models"

os.makedirs(MODEL_DIR, exist_ok=True)

X_train = np.load(f"{DATA_DIR}\\X_train.npy")
y_train = np.load(f"{DATA_DIR}\\y_train.npy")
X_test = np.load(f"{DATA_DIR}\\X_test.npy")
y_test = np.load(f"{DATA_DIR}\\y_test.npy")

# Build model
model = build_model(X_train.shape[1])

# Handle class imbalance
cw = class_weight.compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
cw = {i: cw[i] for i in range(len(cw))}

# Callbacks
callbacks = [
    EarlyStopping(patience=8, restore_best_weights=True),
    ModelCheckpoint(f"{MODEL_DIR}\\best_model.h5", save_best_only=True)
]

# Train
history = model.fit(
    X_train, y_train,
    validation_split=0.15,
    epochs=100,
    batch_size=128,
    callbacks=callbacks,
    class_weight=cw,
    verbose=1
)

model.save(f"{MODEL_DIR}\\final_model.h5")

print("Training complete!")
