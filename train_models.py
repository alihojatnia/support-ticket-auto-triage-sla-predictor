# train_models.py — CLEAN, FINAL, UNBREAKABLE 2025
import os
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
import joblib

# ================================
# 1. CREATE FOLDERS
# ================================
os.makedirs("models", exist_ok=True)

# ================================
# 2. LOAD TINY PERFECT DATA
# ================================
url = "https://bit.ly/zendesk-tiny-compatible"
df = pd.read_csv(url)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

train_df = df.iloc[:150]
test_df  = df.iloc[150:]

print(f"Loaded {len(train_df)} train | {len(test_df)} test tickets")

# ================================
# 3. TOKENIZER
# ================================
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
tokenizer.model_max_length = 64

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=64)

# ================================
# 4. TRAIN CLASSIFIER (Priority / Department)
# ================================
def train_classifier(task, col, num_labels):
    print(f"\nTraining {task.upper()}...")

    train_ds = Dataset.from_pandas(train_df[["text", col]]).map(tokenize, batched=True)
    test_ds  = Dataset.from_pandas(test_df[["text", col]]).map(tokenize, batched=True)

    train_ds = train_ds.class_encode_column(col).rename_column(col, "labels")
    test_ds  = test_ds.class_encode_column(col).rename_column(col, "labels")

    # BULLETPROOF TORCH FORMAT
    train_ds = train_ds.with_format("torch", columns=["input_ids", "attention_mask", "labels"], dtype=torch.long)
    test_ds  = test_ds.with_format("torch", columns=["input_ids", "attention_mask", "labels"], dtype=torch.long)

    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=num_labels)

    args = TrainingArguments(
        output_dir=f"models/{task}",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        learning_rate=5e-5,
        warmup_steps=10,
        weight_decay=0.01,
        load_best_model_at_end=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=5,
        report_to=[],
        seed=42,
    )

    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=test_ds)
    trainer.train()

    preds = trainer.predict(test_ds).predictions.argmax(-1)
    f1 = f1_score(test_ds["labels"], preds, average="weighted")
    print(f"{task.upper()} F1: {f1:.3f}")

    model.save_pretrained(f"models/{task}")
    tokenizer.save_pretrained(f"models/{task}")
    return f1

# ================================
# 5. RUN BOTH CLASSIFIERS
# ================================
f1_priority = train_classifier("priority", "priority", 4)
f1_dept     = train_classifier("department", "department", 3)

# ================================
# 6. SLA: Logistic on BERT Features
# ================================
print("\nTraining SLA...")
base = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=1)

def get_features(df):
    ds = Dataset.from_pandas(df[["text"]]).map(tokenize, batched=True)
    ds.set_format("torch", columns=["input_ids", "attention_mask"])
    feats = []
    for i in range(0, len(ds), 32):
        batch = {k: ds[i:i+32][k] for k in ["input_ids", "attention_mask"]}
        with torch.no_grad():
            hidden = base.base_model(**batch).last_hidden_state[:, 0, :].detach().cpu().numpy()
        feats.append(hidden)
    return np.vstack(feats)

X_train = get_features(train_df)
X_test  = get_features(test_df)
y_train = (train_df["sla_breach_prob"] > 0.5).astype(int)

reg = LogisticRegression(max_iter=1000)
reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)
mae = np.mean(np.abs(y_pred - (test_df["sla_breach_prob"] > 0.5).astype(int)))
print(f"SLA MAE: {mae:.3f}")

joblib.dump(reg, "models/sla_regressor.pkl")
base.save_pretrained("models/sla_features")
tokenizer.save_pretrained("models/sla_features")

# ================================
# 7. SAVE METRICS
# ================================
with open("metrics.txt", "w") as f:
    f.write("Priority F1,Department F1,SLA MAE\n")
    f.write(f"{f1_priority:.3f},{f1_dept:.3f},{mae:.3f}")

print("\nVICTORY! Models ready → streamlit run app.py")