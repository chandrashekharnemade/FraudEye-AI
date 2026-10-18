
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

df = pd.read_csv("data/raw/creditcard.csv")

X = df.drop(columns=["Class", "Time"])
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = joblib.load("models/fraudeye_random_forest.joblib")

# Use a validation split to choose the threshold
X_fit, X_val, y_fit, y_val = train_test_split(
    X_train, y_train,
    test_size=0.25,
    random_state=42,
    stratify=y_train
)

# Train a fresh model on the fitting portion
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=150,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)
model.fit(X_fit, y_fit)

val_probs = model.predict_proba(X_val)[:, 1]

print("Validation threshold comparison:")
for threshold in [0.10, 0.20, 0.30, 0.40, 0.50]:
    pred = (val_probs >= threshold).astype(int)

    print(
        f"Threshold={threshold:.2f} | "
        f"Precision={precision_score(y_val, pred, zero_division=0):.4f} | "
        f"Recall={recall_score(y_val, pred, zero_division=0):.4f} | "
        f"F1={f1_score(y_val, pred, zero_division=0):.4f}"
    )
