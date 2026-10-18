
from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score
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

model = RandomForestClassifier(
    n_estimators=150,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

probs = model.predict_proba(X_test)[:, 1]
threshold = 0.20
preds = (probs >= threshold).astype(int)

print("Threshold:", threshold)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, preds))

print("\nClassification Report:")
print(classification_report(y_test, preds, zero_division=0))

print("PR-AUC:", average_precision_score(y_test, probs))

Path("models").mkdir(exist_ok=True)
joblib.dump(model, "models/fraudeye_random_forest.joblib")
joblib.dump({"threshold": threshold}, "models/fraudeye_config.joblib")

print("\nFinal model and threshold saved.")
