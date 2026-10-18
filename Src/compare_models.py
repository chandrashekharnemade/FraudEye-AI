
from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score
)

# Load dataset
df = pd.read_csv("data/raw/creditcard.csv")

X = df.drop(columns=["Class", "Time"])
y = df["Class"]

# Same split for a fair comparison
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

models = {
    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42
        )
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=150,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    )
}

results = []

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    results.append({
        "Model": name,
        "Precision": precision_score(
            y_test, pred, zero_division=0
        ),
        "Recall": recall_score(
            y_test, pred, zero_division=0
        ),
        "F1-score": f1_score(
            y_test, pred, zero_division=0
        ),
        "PR-AUC": average_precision_score(y_test, proba)
    })

    if name == "Random Forest":
        Path("models").mkdir(exist_ok=True)
        joblib.dump(model, "models/fraudeye_random_forest.joblib")

result_df = pd.DataFrame(results)
print("\n=== MODEL COMPARISON ===")
print(result_df.round(4).to_string(index=False))

Path("results").mkdir(exist_ok=True)
result_df.to_csv("results/model_comparison.csv", index=False)

print("\nSaved: results/model_comparison.csv")
print("Saved: models/fraudeye_random_forest.joblib")