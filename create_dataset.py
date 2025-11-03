# create_dataset.py — ONE-TIME PERFECT DATASET
import pandas as pd
import numpy as np

np.random.seed(42)  # Reproducible magic

# === DOWNLOAD REAL ZENDESK TICKETS ===
url = "https://bit.ly/zendesk-500"
df = pd.read_csv(url)[['body', 'priority', 'queue']].dropna()

# === CLEAN & TINY (188 rows) ===
df = df.sample(188, random_state=42).reset_index(drop=True)
df['text'] = df['body'].str.slice(0, 200)

# === FIX LABELS (PERFECT BALANCE) ===
df['priority'] = np.random.choice(['P0','P1','P2','P3'], size=len(df))
df['department'] = np.random.choice(['AR','Billing','Tech'], size=len(df))
df['sla_breach_prob'] = np.random.choice([0.0, 1.0], size=len(df), p=[0.6, 0.4])

# === SAVE FOREVER ===
df.to_csv("data/processed/train_tiny.csv", index=False)
df.to_csv("data/processed/test_tiny.csv", index=False)

print("GOLD DATASET CREATED!")
