from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.metrics import f1_score, mean_absolute_error
from sklearn.linear_model import LogisticRegression
import pandas as pd, torch, numpy as np, os, joblib

# SETTINGS
MAX_LEN = 128
BATCH = 8
EPOCHS = 2
DEVICE = "cpu"
print("Training on CPU — 3 minutes total")

os.makedirs("models", exist_ok=True)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(b):
    return tokenizer(b["text"], truncation=True, padding="max_length", max_length=MAX_LEN, return_tensors="pt")

train_df = pd.read_csv("data/processed/train.csv")
test_df  = pd.read_csv("data/processed/test.csv")
print(f"Loaded {len(train_df):,} train | {len(test_df):,} test")

def train_classifier(task, col):
    print(f"\nTraining {task.upper()}...")
    train_ds = Dataset.from_pandas(train_df[["text", col]]).map(tokenize, batched=True)
    test_ds  = Dataset.from_pandas(test_df[["text", col]]).map(tokenize, batched=True)

    train_ds = train_ds.class_encode_column(col).rename_column(col, "labels")
    test_ds  = test_ds.class_encode_column(col).rename_column(col, "labels")

    # FINAL FIX: keep input_ids + correct dtype
    train_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    test_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    # Convert labels Long → Float
    train_ds = train_ds.map(lambda x: {"labels": x["labels"].float()})
    test_ds  = test_ds.map(lambda x: {"labels": x["labels"].float()})

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
        logging_steps=50,
        report_to=[],
        remove_unused_columns=False,
    )

    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=test_ds)
    trainer.train()

    preds = trainer.predict(test_ds).predictions.argmax(-1)
    f1 = f1_score([l.item() for l in test_ds["labels"]], preds, average="weighted")
    print(f"{task.upper()} F1: {f1:.3f}")

    model.save_pretrained(f"models/{task}")
    tokenizer.save_pretrained(f"models/{task}")
    return f1

f1_p = train_classifier("priority", "priority")
f1_d = train_classifier("department", "department")

print("\nTraining SLA...")
base = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=1)
base.eval()

def feats(df):
    ds = Dataset.from_pandas(df[["text"]]).map(tokenize, batched=True).set_format("torch")
    f = []
    with torch.no_grad():
        for i in range(0, len(ds), 32):
            batch = {k: v.squeeze(1) if k != "labels" else v for k, v in ds[i:i+32].items()}
            batch = {k: v.to(DEVICE) for k, v in batch.items() if k in ["input_ids", "attention_mask"]}
            hidden = base.base_model(**batch).last_hidden_state[:,0,:].cpu().numpy()
            f.append(hidden)
    return np.vstack(f)

X_tr = feats(train_df)
X_te = feats(test_df)
reg = LogisticRegression(max_iter=1000).fit(X_tr, (train_df["sla_breach_prob"]>0.5).astype(int))
mae = mean_absolute_error(test_df["sla_breach_prob"], reg.predict_proba(X_te)[:,1])
print(f"SLA MAE: {mae:.3f}")

joblib.dump(reg, "models/sla_regressor.pkl")
base.save_pretrained("models/sla_features")
tokenizer.save_pretrained("models/sla_features")

with open("metrics.txt","w") as f:
    f.write("Priority F1,Department F1,SLA MAE\n")
    f.write(f"{f1_p:.3f},{f1_d:.3f},{mae:.3f}")

print("\nDONE! Run: streamlit run app.py")