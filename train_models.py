from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.metrics import f1_score, mean_absolute_error
from sklearn.linear_model import LogisticRegression
import pandas as pd, torch, numpy as np, os, joblib

# SETTINGS
MAX_LEN = 64
BATCH = 4
EPOCHS = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {DEVICE}")

os.makedirs("models", exist_ok=True)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=MAX_LEN)

# LOAD
train_df = pd.read_csv("data/processed/train.csv")
test_df  = pd.read_csv("data/processed/test.csv")
print(f"Loaded {len(train_df):,} train | {len(test_df):,} test")

# TRAIN ONE TASK
def train_classifier(task, col):
    print(f"\nTraining {task.upper()}...")
    train_ds = Dataset.from_pandas(train_df[["text", col]]).map(tokenize, batched=True)
    test_ds  = Dataset.from_pandas(test_df[["text", col]]).map(tokenize, batched=True)

    # Encode & rename
    train_ds = train_ds.class_encode_column(col).rename_column(col, "labels")
    test_ds  = test_ds.class_encode_column(col).rename_column(col, "labels")

    # FINAL FIX: keep ALL columns, set torch format ONCE
    train_ds.set_format("torch")
    test_ds.set_format("torch")

    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=train_ds.features["labels"].num_classes
    )

    args = TrainingArguments(
        f"models/{task}",
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=BATCH,
        per_device_eval_batch_size=BATCH,
        num_train_epochs=EPOCHS,
        load_best_model_at_end=True,
        fp16=(DEVICE=="cuda"),
        logging_steps=50,
        report_to=[],
        remove_unused_columns=False,   # ← KEEPS input_ids
    )

    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=test_ds)
    trainer.train()

    preds = trainer.predict(test_ds).predictions.argmax(-1)

    f1 = f1_score(test_ds["labels"], preds, average="weighted")
    print(f"{task.upper()} F1: {f1:.3f}")

    model.save_pretrained(f"models/{task}")
    tokenizer.save_pretrained(f"models/{task}")
    return f1

# RUN
f1_p = train_classifier("priority", "priority")
f1_d = train_classifier("department", "department")

# SLA
print("\nTraining SLA...")
base = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=1).to(DEVICE)
base.eval()

def feats(df):
    ds = Dataset.from_pandas(df[["text"]]).map(tokenize, batched=True).set_format("torch")
    f = []
    with torch.no_grad():
        for i in range(0, len(ds), 32):
            batch = {k: ds[i:i+32][k].to(DEVICE) for k in ["input_ids", "attention_mask"]}
            hidden = base.base_model(**batch).last_hidden_state[:,0,:].cpu().numpy()
            f.append(hidden)
    return np.vstack(f)

X_tr = feats(train_df)
X_te = feats(test_df)
reg = LogisticRegression(max_iter=10).fit(X_tr, (train_df["sla_breach_prob"]>0.5).astype(int))
mae = mean_absolute_error(test_df["sla_breach_prob"], reg.predict_proba(X_te)[:,1])
print(f"SLA MAE: {mae:.3f}")

joblib.dump(reg, "models/saa_regressor.pkl")
base.save_pretrained("models/sla_features")
tokenizer.save_pretrained("models/sla_features")

# METRICS
with open("metrics.txt", "w") as f:
    f.write("Priority F1,Department F1,SLA MAE\n")
    f.write(f"{f1_p:.3f},{f1_d:.3f},{mae:.3f}")

print("\nDONE! Check metrics.txt → streamlit run app.py")