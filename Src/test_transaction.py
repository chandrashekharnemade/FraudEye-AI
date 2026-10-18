
import pandas as pd

df = pd.read_csv("data/raw/creditcard.csv")

# Select one known fraudulent transaction
row = df[df["Class"] == 1].iloc[0]

print("Actual class:", row["Class"])
print("Transaction amount:", row["Amount"])

for i in range(1, 29):
    print(f"V{i}: {row[f'V{i}']:.6f}")
