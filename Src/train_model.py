
from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    average_precision_score,
    confusion_matrix
)

# Load dataset
df = pd.read_csv("data/raw/creditcard.csv")

X = df.drop(columns=["Class", "Time"])
y = df["Class"]

# Keep test data separate from training data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train baseline model
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    )
)

model.fit(X_train, y_train)

# Evaluate on held-out test data
pred = model.predict(X_test)
proba = model.predict_proba(X_test)[:, 1]

print("Confusion Matrix:")
print(confusion_matrix(y_test, pred))

print("\nClassification Report:")
print(classification_report(y_test, pred, zero_division=0))

print("PR-AUC (Average Precision):",
      average_precision_score(y_test, proba))

# Save trained model
Path("models").mkdir(exist_ok=True)
joblib.dump(model, "models/fraudeye_baseline.joblib")

print("\nModel saved to models/fraudeye_baseline.joblib")
