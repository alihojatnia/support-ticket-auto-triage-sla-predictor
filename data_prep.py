# data_prep.py


import pandas as pd
import os
from sklearn.model_selection import train_test_split

print("Loading raw data...")
df = pd.read_csv("data/raw/dataset.csv")
print(f"Raw rows: {len(df)}")


df = df[df["language"] == "en"].copy()
print(f"English rows: {len(df)}")


df = df.sample(n=5000, random_state=42).reset_index(drop=True)


df["text"] = (df["subject"].fillna("") + " " + df["body"].fillna("")) \
               .str.lower() \
               .str.replace("\n", " ") \
               .str.replace(r"\s+", " ", regex=True) \
               .str.strip()


dept_map = {
    "billing": "Billing", "accounts": "AR", "technical": "Tech",
    "support": "Tech", "sales": "AR", "general": "Tech",
    "refund": "Billing", "payment": "Billing"
}
df["department"] = df["queue"].astype(str).str.lower().map(dept_map).fillna("Tech")


def parse_priority(x):
    x = str(x).strip().lower()
    if x.isdigit():
        return int(x)
    # keyword fallback
    if any(k in x for k in ["urgent","p0","critical","high"]): return 1
    if any(k in x for k in ["medium","normal"]):               return 3
    return 4  # default low

df["priority_num"] = df["priority"].apply(parse_priority)
df["priority"] = pd.cut(
    df["priority_num"],
    bins=[0,1,2,3,4,100],
    labels=["P0","P1","P2","P3","P3"],
    ordered=False
).astype(str)


df["sla_breach_prob"] = 0.1
df.loc[df["priority"].isin(["P0","P1"]), "sla_breach_prob"] += 0.4
df.loc[df["text"].str.len() > 400, "sla_breach_prob"] += 0.3
df.loc[df["text"].str.contains("urgent|critical|down|outage|asap", case=False), "sla_breach_prob"] += 0.2
df["sla_breach_prob"] = df["sla_breach_prob"].clip(0,1)


df = df[["text","department","priority","sla_breach_prob"]].dropna()
print(f"Final rows: {len(df)}")

print("\nLabels:")
print(df["department"].value_counts().head())
print(df["priority"].value_counts())


train_df, test_df = train_test_split(df, test_size=0.2, random_state=42,
                                     stratify=df[["department","priority"]])


os.makedirs("data/processed", exist_ok=True)
train_df.to_csv("data/processed/train.csv", index=False)
test_df.to_csv("data/processed/test.csv", index=False)

print(f"\nSAVED! Train: {len(train_df)}  Test: {len(test_df)}")
print("Next → python train_models.py")