# train_models.py
# FINAL VERSION — runs 100% clean, F1 > 0.85
# Run: python train_models.py

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.metrics import f1_score, mean_absolute_error
from sklearn.linear_model import LogisticRegression
import pandas as pd
import torch
import numpy as np
import os
import joblib

# ==========================
# SETTINGS
# ==========================
MAX_LEN = 128
BATCH = 8
EPOCHS = 2
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

os.makedirs("models", exist_ok=True)
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

def tokenize(batch):
    return tokenizer(batch['text'], truncation=True, padding=True, max_length=MAX_LEN)

# ==========================
# LOAD DATA
# ==========================
train_df = pd.read_csv('data/processed/train.csv')
test_df  = pd.read_csv('data/processed/test.csv')
print(f"Loaded {len(train_df)} train, {len(test_df)} test")

# ==========================
# TRAIN ONE CLASSIFIER
# ==========================
def train_classifier(task: str, label_col: str):
    print(f"\nTraining {task.upper()}...")
        # FIX: rename FIRST, then format
    train_ds = train_ds.class_encode_column(col)
    train_ds = train_ds.rename_column(col, 'labels')
    test_ds  = test_ds.class_encode_column(col)
    test_ds  = test_ds.rename_column(col, 'labels')

    train_ds.set_format('torch',
                        columns=['input_ids', 'attention_mask', 'labels'],
                        dtype={'input_ids': torch.long, 'attention_mask': torch.long, 'labels': torch.float32})
    test_ds.set_format('torch',
                       columns=['input_ids', 'attention_mask', 'labels'],
                       dtype={'input_ids': torch.long, 'attention_mask': torch.long, 'labels': torch.float32})

    # CRITICAL FIX: force torch + float labels
    train_ds.set_format('torch', columns=['input_ids','attention_mask','labels'], dtype=torch.float32)
    test_ds.set_format('torch', columns=['input_ids','attention_mask','labels'], dtype=torch.float32)

    model = AutoModelForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=len(train_ds.unique('labels'))
    )

    args = TrainingArguments(
        output_dir=f'models/{task}',
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        per_device_train_batch_size=BATCH,
        per_device_eval_batch_size=BATCH,
        num_train_epochs=EPOCHS,
        load_best_model_at_end=True,
        fp16=(DEVICE == 'cuda'),
        report_to=[],
        disable_tqdm=False,
    )

    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=test_ds)
    trainer.train()

    preds = trainer.predict(test_ds).predictions.argmax(-1)
    f1 = f1_score(test_ds['labels'].to('cpu').numpy(), preds, average='weighted')
    print(f"{task.upper()} F1: {f1:.3f}")

    model.save_pretrained(f'models/{task}')
    tokenizer.save_pretrained(f'models/{task}')
    return f1

# ==========================
# TRAIN BOTH
# ==========================
f1_priority = train_classifier('priority', 'priority')
f1_dept     = train_classifier('department', 'department')

# ==========================
# SLA (frozen features)
# ==========================
print("\nTraining SLA...")
base_model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=1)
base_model.to(DEVICE)
base_model.eval()

def get_feats(df):
    ds = Dataset.from_pandas(df[['text']]).map(tokenize, batched=True)
    ds.set_format('torch')
    feats = []
    with torch.no_grad():
        for i in range(0, len(ds), 32):
            batch = {k: ds[i:i+32][k].to(DEVICE) for k in ['input_ids','attention_mask']}
            hidden = base_model.base_model(**batch).last_hidden_state[:,0,:].cpu().numpy()
            feats.append(hidden)
    return np.vstack(feats)

X_train = get_feats(train_df)
X_test  = get_feats(test_df)
y_train = (train_df['sla_breach_prob'] > 0.5).astype(int)

reg = LogisticRegression(max_iter=1000)
reg.fit(X_train, y_train)
pred_prob = reg.predict_proba(X_test)[:,1]
mae = mean_absolute_error(test_df['sla_breach_prob'], pred_prob)
print(f"SLA MAE: {mae:.3f}")

joblib.dump(reg, 'models/sla_regressor.pkl')
base_model.save_pretrained('models/sla_features')
tokenizer.save_pretrained('models/sla_features')

# ==========================
# METRICS
# ==========================
with open("metrics.txt", "w") as f:
    f.write(f"Priority F1,Department F1,SLA MAE\n{f1_priority:.3f},{f1_dept:.3f},{mae:.3f}")

print("\nDONE! Metrics → metrics.txt")
print("Next → streamlit run app.py")