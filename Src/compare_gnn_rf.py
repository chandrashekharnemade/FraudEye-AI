
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
)
from torch_geometric.nn import SAGEConv

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "Raw" / "creditcard.csv"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

SEED = 42
MAX_LEGIT = 20000
K = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

np.random.seed(SEED)
torch.manual_seed(SEED)

# Load the same sampled dataset and split
df = pd.read_csv(DATA)
fraud = df[df["Class"] == 1]
legit = df[df["Class"] == 0].sample(
    n=min(MAX_LEGIT, int((df["Class"] == 0).sum())),
    random_state=SEED,
)
data = pd.concat([legit, fraud], ignore_index=True)
data = data.sample(frac=1, random_state=SEED).reset_index(drop=True)

features = [c for c in data.columns if c not in ["Class", "Time"]]
X = data[features].to_numpy(dtype=np.float32)
y = data["Class"].to_numpy(dtype=np.int64)
idx = np.arange(len(y))

train_idx, temp_idx = train_test_split(
    idx, test_size=0.30, random_state=SEED, stratify=y
)
val_idx, test_idx = train_test_split(
    temp_idx, test_size=0.50, random_state=SEED,
    stratify=y[temp_idx],
)

scaler = StandardScaler()
X[train_idx] = scaler.fit_transform(X[train_idx]).astype(np.float32)
X[temp_idx] = scaler.transform(X[temp_idx]).astype(np.float32)

test_y = y[test_idx]

# Random Forest baseline
print("Training Random Forest...")
rf = RandomForestClassifier(
    n_estimators=150,
    class_weight="balanced_subsample",
    random_state=SEED,
    n_jobs=-1,
)
rf.fit(X[train_idx], y[train_idx])
rf_prob = rf.predict_proba(X[test_idx])[:, 1]

# Feature-similarity graph
print("Building graph...")
knn = NearestNeighbors(
    n_neighbors=K + 1, metric="euclidean", n_jobs=-1
)
knn.fit(X)
neighbors = knn.kneighbors(X, return_distance=False)

src = np.repeat(np.arange(len(X)), K)
dst = neighbors[:, 1:].reshape(-1)
src_all = np.concatenate([src, dst, np.arange(len(X))])
dst_all = np.concatenate([dst, src, np.arange(len(X))])

edge_index = torch.tensor(
    np.stack([src_all, dst_all]), dtype=torch.long, device=DEVICE
)
x = torch.tensor(X, dtype=torch.float32, device=DEVICE)
labels = torch.tensor(y, dtype=torch.float32, device=DEVICE)

train_mask = torch.zeros(len(y), dtype=torch.bool, device=DEVICE)
val_mask = torch.zeros(len(y), dtype=torch.bool, device=DEVICE)
train_mask[train_idx] = True
val_mask[val_idx] = True


class GraphSAGE(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.conv1 = SAGEConv(input_dim, 32)
        self.conv2 = SAGEConv(32, 32)
        self.out = torch.nn.Linear(32, 1)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        return self.out(x).squeeze(-1)


print("Training GraphSAGE on", DEVICE)
gnn = GraphSAGE(len(features)).to(DEVICE)

train_y = y[train_idx]
pos = max(int(train_y.sum()), 1)
neg = len(train_y) - pos
pos_weight = torch.tensor(
    [neg / pos], dtype=torch.float32, device=DEVICE
)
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(
    gnn.parameters(), lr=0.003, weight_decay=1e-4
)

best_ap = -1.0
best_state = None
wait = 0

for epoch in range(1, 31):
    gnn.train()
    optimizer.zero_grad()

    logits = gnn(x, edge_index)
    loss = criterion(logits[train_mask], labels[train_mask])
    loss.backward()
    optimizer.step()

    gnn.eval()
    with torch.no_grad():
        all_prob = torch.sigmoid(gnn(x, edge_index)).cpu().numpy()

    val_prob = all_prob[val_idx]
    val_y = y[val_idx]
    val_ap = average_precision_score(val_y, val_prob)

    print(
        f"Epoch {epoch:02d}: loss={loss.item():.4f}, "
        f"val_PR_AUC={val_ap:.4f}"
    )

    if val_ap > best_ap:
        best_ap = val_ap
        best_state = {
            k: v.detach().cpu().clone()
            for k, v in gnn.state_dict().items()
        }
        wait = 0
    else:
        wait += 1

    if wait >= 6:
        print("Early stopping.")
        break

if best_state is None:
    raise RuntimeError("No valid GraphSAGE checkpoint was produced.")

gnn.load_state_dict(best_state)
gnn.eval()

with torch.no_grad():
    all_prob = torch.sigmoid(gnn(x, edge_index)).cpu().numpy()

gnn_prob = all_prob[test_idx]

assert len(test_y) == len(rf_prob) == len(gnn_prob)

# Same held-out rows and threshold for both models
rows = []
for name, prob in [
    ("Random Forest", rf_prob),
    ("GraphSAGE", gnn_prob),
]:
    pred = (prob >= 0.5).astype(int)
    rows.append({
        "Model": name,
        "Threshold": 0.5,
        "Precision": precision_score(test_y, pred, zero_division=0),
        "Recall": recall_score(test_y, pred, zero_division=0),
        "F1-score": f1_score(test_y, pred, zero_division=0),
        "PR-AUC": average_precision_score(test_y, prob),
        "Test rows": len(test_y),
        "Test frauds": int(test_y.sum()),
    })

result = pd.DataFrame(rows)
out = RESULTS / "gnn_vs_random_forest.csv"
result.to_csv(out, index=False)

print("\nFAIR-SPLIT COMPARISON")
print(result.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print("\nSaved:", out)
print("Device:", DEVICE)
print(
    "Note: feature-similarity graph prototype; "
    "not real banking relationships."
)
