
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
)

from torch_geometric.nn import SAGEConv


# --------------------------------------------------
# FraudEye AI - GraphSAGE prototype
# Feature-similarity graph, NOT real banking relations
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "Raw" / "creditcard.csv"
MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "results"

MODEL_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

SEED = 42
MAX_LEGIT = 20000
K_NEIGHBORS = 5
EPOCHS = 30
PATIENCE = 6

np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 55)
print("FraudEye AI - GraphSAGE Training")
print("=" * 55)
print("Device:", device)

# 1. Load data
df = pd.read_csv(DATA_PATH)

fraud = df[df["Class"] == 1]
legit = df[df["Class"] == 0]

# Keep every fraud row and sample legitimate rows.
# This makes a manageable prototype dataset.
legit_sample = legit.sample(
    n=min(MAX_LEGIT, len(legit)),
    random_state=SEED,
)

data = pd.concat([legit_sample, fraud], ignore_index=True)
data = data.sample(frac=1, random_state=SEED).reset_index(drop=True)

feature_cols = [
    c for c in data.columns
    if c not in ["Class", "Time"]
]

X = data[feature_cols].to_numpy(dtype=np.float32)
y = data["Class"].to_numpy(dtype=np.int64)

print("Prototype rows:", len(data))
print("Fraud rows:", int(y.sum()))
print("Legitimate rows:", int((y == 0).sum()))
print("Features:", len(feature_cols))

# 2. Split before fitting the scaler
indices = np.arange(len(y))

train_idx, temp_idx = train_test_split(
    indices,
    test_size=0.30,
    random_state=SEED,
    stratify=y,
)

val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=0.50,
    random_state=SEED,
    stratify=y[temp_idx],
)

scaler = StandardScaler()
X[train_idx] = scaler.fit_transform(X[train_idx]).astype(np.float32)
X[temp_idx] = scaler.transform(X[temp_idx]).astype(np.float32)

# 3. Build a feature-similarity graph
# Edges use features only; labels are not used to create edges.
print("Building similarity graph...")

knn = NearestNeighbors(
    n_neighbors=K_NEIGHBORS + 1,
    metric="euclidean",
    algorithm="auto",
    n_jobs=-1,
)

knn.fit(X)
neighbors = knn.kneighbors(X, return_distance=False)

src = np.repeat(np.arange(len(X)), K_NEIGHBORS)
dst = neighbors[:, 1:].reshape(-1)

# Make edges bidirectional and add self-loops.
src_all = np.concatenate([src, dst, np.arange(len(X))])
dst_all = np.concatenate([dst, src, np.arange(len(X))])

edge_index = torch.tensor(
    np.stack([src_all, dst_all]),
    dtype=torch.long,
    device=device,
)

x_tensor = torch.tensor(X, dtype=torch.float32, device=device)
y_tensor = torch.tensor(y, dtype=torch.float32, device=device)

train_mask = torch.zeros(len(y), dtype=torch.bool, device=device)
val_mask = torch.zeros(len(y), dtype=torch.bool, device=device)
test_mask = torch.zeros(len(y), dtype=torch.bool, device=device)

train_mask[train_idx] = True
val_mask[val_idx] = True
test_mask[test_idx] = True

print("Graph nodes:", len(X))
print("Directed edges:", edge_index.shape[1])
print("Train / validation / test:", len(train_idx), len(val_idx), len(test_idx))

# 4. Define GraphSAGE
class FraudGraphSAGE(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.output = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        return self.output(x).squeeze(-1)


model = FraudGraphSAGE(len(feature_cols)).to(device)

# Account for class imbalance in the training split.
train_labels = y[train_idx]
positive = max(int(train_labels.sum()), 1)
negative = int(len(train_labels) - positive)

pos_weight = torch.tensor(
    [negative / positive],
    dtype=torch.float32,
    device=device,
)

criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-4)

# 5. Train; use validation PR-AUC for early stopping.
best_val_ap = -1.0
best_state = None
wait = 0
start = time.perf_counter()

for epoch in range(1, EPOCHS + 1):
    model.train()
    optimizer.zero_grad()

    logits = model(x_tensor, edge_index)
    loss = criterion(logits[train_mask], y_tensor[train_mask])

    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_prob = torch.sigmoid(model(x_tensor, edge_index)[val_mask])
        val_true = y_tensor[val_mask].cpu().numpy().astype(int)
        val_scores = val_prob.cpu().numpy()

    val_ap = average_precision_score(val_true, val_scores)

    print(
        f"Epoch {epoch:02d} | "
        f"Loss {loss.item():.4f} | "
        f"Validation PR-AUC {val_ap:.4f}"
    )

    if val_ap > best_val_ap:
        best_val_ap = val_ap
        best_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
        wait = 0
    else:
        wait += 1

    if wait >= PATIENCE:
        print("Early stopping.")
        break

if best_state is None:
    raise RuntimeError("Training did not produce a model checkpoint.")

model.load_state_dict(best_state)
model.eval()

# 6. Evaluate once on the held-out test split.
with torch.no_grad():
    test_logits = model(x_tensor, edge_index)[test_mask]
    test_prob = torch.sigmoid(test_logits).cpu().numpy()

test_true = y[test_mask.cpu().numpy()]
threshold = 0.5
test_pred = (test_prob >= threshold).astype(int)

precision = precision_score(test_true, test_pred, zero_division=0)
recall = recall_score(test_true, test_pred, zero_division=0)
f1 = f1_score(test_true, test_pred, zero_division=0)
pr_auc = average_precision_score(test_true, test_prob)
cm = confusion_matrix(test_true, test_pred, labels=[0, 1])

elapsed = time.perf_counter() - start

print("\n" + "=" * 55)
print("TEST RESULTS - GraphSAGE")
print("=" * 55)
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"PR-AUC:    {pr_auc:.4f}")
print("Confusion matrix [[TN, FP], [FN, TP]]:")
print(cm)
print(f"Training + evaluation time: {elapsed:.1f} seconds")

# 7. Save a separate checkpoint and result file.
model_path = MODEL_DIR / "fraudeye_graphsage_prototype.pt"
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "input_dim": len(feature_cols),
        "hidden_dim": 32,
        "feature_cols": feature_cols,
        "threshold": threshold,
        "graph_type": "feature_similarity_knn",
        "note": "Prototype graph, not real banking relationships",
    },
    model_path,
)

metrics = {
    "model": "GraphSAGE prototype",
    "graph_type": "feature_similarity_knn",
    "sample_rows": int(len(data)),
    "fraud_rows": int(y.sum()),
    "test_rows": int(len(test_idx)),
    "threshold": threshold,
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "pr_auc": float(pr_auc),
    "training_seconds": float(elapsed),
    "device": str(device),
}

metrics_path = RESULT_DIR / "graphsage_prototype_metrics.json"
metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

print("\nSaved model:", model_path)
print("Saved metrics:", metrics_path)
print("GraphSAGE prototype completed.")
