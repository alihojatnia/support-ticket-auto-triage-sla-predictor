from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.metrics import f1_score, mean_absolute_error
from sklearn.linear_model import LogisticRegression
import pandas as pd, torch, numpy as np, os, joblib

MAX_LEN = 64
BATCH   = 4
EPOCHS  = 1
DEVICE  = "cpu"

print("Training on CPU – 90 seconds total")
os.makedirs("models", exist_ok=True)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=MAX_LEN)

train_df = pd.read_csv("data/processed/train.csv")
test_df  = pd.read_csv("data/processed/test.csv")
print(f"Loaded {len(train_df):,} train | {len(test_df):,} test")

def train_classifier(task: str, col: str):
    print(f"\nTraining {task.upper()}...")
    tr = Dataset.from_pandas(train_df[["text", col]]).map(tokenize, batched=True)
    te = Dataset.from_pandas(test_df[["text", col]]).map(tokenize, batched=True)

    tr = tr.class_encode_column(col).rename_column(col, "labels")
    te = te.class_encode_column(col).rename_column(col, "labels")

    # KEEP INPUTS + FORCE LABELS TO FLOAT32
    tr.set_format("torch", columns=["input_ids","attention_mask","labels"])
    te.set_format("torch", columns=["input_ids","attention_mask","labels"])

    # ONE-LINE MAGIC FIX
    tr = tr.map(lambda x: {"labels": x["labels"].to(torch.float32)}, batched=False)
    te = te.map(lambda x: {"labels": x["labels"].to(torch.float32)}, batched=False)

    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=tr.features["labels"].num_classes
    )

    args = TrainingArguments(
        f"models/{task}",
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=BATCH,
        per_device_eval_batch_size=BATCH,
        num_train_epochs=EPOCHS,
        load_best_model_at_end=True,
        logging_steps=100,
        report_to=[],
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )

    trainer = Trainer(model=model, args=args, train_dataset=tr, eval_dataset=te)
    trainer.train()

    # CLEAN F1
    out = trainer.predict(te)
    f1 = f1_score(out.label_ids, out.predictions.argmax(-1), average="weighted")
    print(f"{task.upper()} F1: {f1:.3f}")

    model.save_pretrained(f"models/{task}")
    tokenizer.save_pretrained(f"models/{task}")
    return f1

f1_p = train_classifier("priority", "priority")
f1_d = train_classifier("department", "department")

# SLA
print("\nTraining SLA breach predictor...")
base = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=1).to(DEVICE)
base.eval()

def feats(df):
    ds = Dataset.from_pandas(df[["text"]]).map(tokenize, batched=True).set_format("torch")
    f = []
    with torch.no_grad():
        for i in range(0, len(ds), 32):
            batch = {k: ds[i:i+32][k].to(DEVICE) for k in ["input_ids","attention_mask"]}
            h = base.base_model(**batch).last_hidden_state[:,0,:].cpu().numpy()
            f.append(h)
    return np.vstack(f)

X_tr, X_te = feats(train_df), feats(test_df)
reg = LogisticRegression(max_iter=1000).fit(X_tr, (train_df["sla_breach_prob"]>0.5).astype(int))
mae = mean_absolute_error(test_df["sla_breach_prob"], reg.predict_proba(X_te)[:,1])
print(f"SLA MAE: {mae:.3f}")

joblib.dump(reg, "models/sla_regressor.pkl")
base.save_pretrained("models/sla_features")
tokenizer.save_pretrained("models/sla_features")

with open("metrics.txt","w") as f:
    f.write("Priority F1,Department F1,SLA MAE\n")
    f.write(f"{f1_p:.3f},{f1_d:.3f},{mae:.3f}")

print("\nVICTORY! Run!")