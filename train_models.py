# train_models.py — 2025 UNBREAKABLE
from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoModelForSequenceClassification
from datasets import Dataset
from sklearn.metrics import f1_score, mean_absolute_error
from sklearn.linear_model import LogisticRegression
import pandas as pd, torch, numpy as np, joblib, os

os.makedirs("models", exist_ok=True)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased", model_max_length=32)

def tokenize(x): return tokenizer(x["text"], truncation=True, padding=True, max_length=32)

train_df = pd.read_csv("data/processed/train_tiny.csv")
test_df  = pd.read_csv("data/processed/test_tiny.csv")
print(f"Loaded {len(train_df)} train | {len(test_df)} test")

def train(task, col):
    train_ds = Dataset.from_pandas(train_df[["text",col]]).map(tokenize, batched=True)
    test_ds  = Dataset.from_pandas(test_df[["text",col]]).map(tokenize, batched=True)
    train_ds = train_ds.class_encode_column(col).rename_column(col,"labels")
    test_ds  = test_ds.class_encode_column(col).rename_column(col,"labels")

    # BULLETPROOF FORMAT
    train_ds = train_ds.with_format("torch", columns=["input_ids","attention_mask","labels"], dtype=torch.long)
    test_ds  = test_ds.with_format("torch",  columns=["input_ids","attention_mask","labels"], dtype=torch.long)

    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=train_ds.features["labels"].num_classes)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            f"models/{task}", 
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=1,
            learning_rate=5e-5,
            warmup_steps=50,
            weight_decay=0.01,
            load_best_model_at_end=True,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_steps=10,
            report_to=[],
        ),
        train_dataset=train_ds,
        eval_dataset=test_ds,
    )
    trainer.train()
    pred = trainer.predict(test_ds).predictions.argmax(-1)
    f1 = f1_score(test_ds["labels"], pred, average="weighted")
    print(f"{task.upper()} F1: {f1:.3f}")
    model.save_pretrained(f"models/{task}")
    tokenizer.save_pretrained(f"models/{task}")
    return f1

f1_p = train("priority", "priority")
f1_d = train("department", "department")

# SLA (2 lines)
print("Training SLA...")
base = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=1)
def feats(df):
    ds = Dataset.from_pandas(df[["text"]]).map(tokenize, batched=True)
    ds.set_format("torch", columns=["input_ids","attention_mask"])
    hidden = []
    for i in range(0,len(ds),64):
        batch = {k: ds[i:i+64][k] for k in ["input_ids","attention_mask"]}
        hidden.append(base.base_model(**batch).last_hidden_state[:,0,:].detach().cpu().numpy())
    return np.vstack(hidden)
X_tr, X_te = feats(train_df), feats(test_df)
y_tr = (train_df["sla_breach_prob"]>0.5).astype(int)
reg = LogisticRegression().fit(X_tr, y_tr)
mae = mean_absolute_error((test_df["sla_breach_prob"]>0.5).astype(int), reg.predict(X_te))
print(f"SLA MAE: {mae:.3f}")
joblib.dump(reg, "models/sla_regressor.pkl")
base.save_pretrained("models/sla_features")
tokenizer.save_pretrained("models/sla_features")

with open("metrics.txt","w") as f:
    f.write(f"Priority F1,Department F1,SLA MAE\n{f1_p:.3f},{f1_d:.3f},{mae:.3f}")
print("DONE → streamlit run app.py")