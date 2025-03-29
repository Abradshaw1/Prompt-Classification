import os
import torch
import pandas as pd
import logging
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from peft import get_peft_model, LoraConfig, TaskType
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from transformers.trainer_callback import TrainerCallback

# ------------------- Setup -------------------
os.makedirs("./results", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("./results/training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DeBERTa LoRA Fine-Tuning")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

MODEL_NAME = "microsoft/deberta-v3-large"
INPUT_FILE = "/home/abradsha/Prompt-Classification/data/outputs/validation_and_cleaning/step2_cleaned_data.csv"

# ------------------- Load Data -------------------
full_df = pd.read_csv(INPUT_FILE)
full_df = full_df[["Prompt", "Malicious (0/1)", "Source"]].dropna()
full_df["Malicious (0/1)"] = full_df["Malicious (0/1)"].astype(int)

# Split by source
manual_df = full_df[full_df["Source"].str.lower() == "manual"]
synthetic_df = full_df[full_df["Source"].str.lower() != "manual"]

# Split 15% of manual into training+validation
manual_train_val_df, manual_test_df = train_test_split(
    manual_df,
    test_size=0.85,
    stratify=manual_df["Malicious (0/1)"],
    random_state=42
)

combined_df = pd.concat([synthetic_df, manual_train_val_df], ignore_index=True)

logger.info(f"Loaded full dataset with {len(full_df)} rows.")
logger.info(f"Manual (held-out test) rows: {len(manual_test_df)} | Synthetic + Manual-train rows: {len(combined_df)}")

# Train/Val split on combined
train_df, val_df = train_test_split(
    combined_df,
    test_size=0.2,
    stratify=combined_df["Malicious (0/1)"],
    random_state=42
)

# Log distributions
for name, subset in zip(["Train", "Validation", "Test (Manual)"], [train_df, val_df, manual_test_df]):
    counts = subset["Malicious (0/1)"].value_counts().to_dict()
    logger.info(
        f"{name} set distribution:\n"
        f"  Non-malicious (0): {counts.get(0, 0)}\n"
        f"  Malicious     (1): {counts.get(1, 0)}\n"
        f"  Total:             {len(subset)}"
    )

# ------------------- Tokenization -------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
logger.info("Tokenizer loaded successfully.")

def preprocess_function(examples):
    return tokenizer(examples["Prompt"], truncation=True, padding='max_length', max_length=256)

train_dataset = Dataset.from_pandas(train_df.rename(columns={"Malicious (0/1)": "label"}))
val_dataset = Dataset.from_pandas(val_df.rename(columns={"Malicious (0/1)": "label"}))

train_dataset = train_dataset.map(preprocess_function, batched=True)
val_dataset = val_dataset.map(preprocess_function, batched=True)
logger.info("Tokenization and dataset formatting complete.")

# ------------------- Model + LoRA -------------------
base_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# Add dropout to classifier head
classifier_dropout = torch.nn.Dropout(0.2)
base_model.classifier = torch.nn.Sequential(
    base_model.classifier,
    classifier_dropout
)
base_model = base_model.to(device)

lora_config = LoraConfig(task_type=TaskType.SEQ_CLS, r=8, lora_alpha=16, lora_dropout=0.1)
model = get_peft_model(base_model, lora_config).to(device)
logger.info("DeBERTa LoRA model initialized and ready.")

# ------------------- Training Arguments -------------------
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=100,
    weight_decay=0.01,
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    logging_dir="./deberta_logs",
    logging_strategy="steps",
    logging_steps=100,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=True,
    label_smoothing_factor=0.1,
    report_to="none",
    disable_tqdm=False
)

# ------------------- Metrics -------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred

    # Convert to tensors
    logits = torch.tensor(logits)
    labels = torch.tensor(labels)

    # Log stats on logits
    logger.info(f"Logits mean: {logits.float().mean().item():.4f} | std: {logits.float().std().item():.4f}")

    # Compute predictions and probabilities
    probs = torch.nn.functional.softmax(logits, dim=-1)[:, 1].numpy()
    preds = torch.argmax(logits, dim=-1).numpy()
    labels = labels.numpy()

    # Log some samples
    logger.info(f"Sample preds: {preds[:10]}")
    logger.info(f"Sample labels: {labels[:10]}")
    logger.info(f"Computing metrics on {len(labels)} samples | Positives: {(labels == 1).sum()} | Negatives: {(labels == 0).sum()}")

    # Compute metrics
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
    acc = accuracy_score(labels, preds)
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = 0.0  # fallback for edge cases

    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "auc": auc,
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
    }

# ------------------- Custom Callback -------------------
class TrainLoggingCallback(TrainerCallback):
    def __init__(self, train_eval_interval=0.25):
        self.train_eval_interval = train_eval_interval
        self.last_logged_train_epoch = -1

    def on_log(self, args, state, control, logs=None, **kwargs):
        current_epoch = state.epoch or 0.0
        epoch_marker = round(current_epoch / self.train_eval_interval) * self.train_eval_interval

        if epoch_marker > self.last_logged_train_epoch:
            self.last_logged_train_epoch = epoch_marker

            logger.info(f"\n[Train Evaluation at Epoch {epoch_marker:.2f}] Running training metrics...")
            metrics = trainer.evaluate(eval_dataset=train_dataset, metric_key_prefix="train")

            tp = metrics.get("train_true_positives", 0)
            fp = metrics.get("train_false_positives", 0)
            tn = metrics.get("train_true_negatives", 0)
            fn = metrics.get("train_false_negatives", 0)
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

            logger.info(f"[Train Epoch {epoch_marker:.2f}] Accuracy: {metrics['train_accuracy']:.4f} | "
                        f"F1: {metrics['train_f1']:.4f} | Precision: {metrics['train_precision']:.4f} | "
                        f"Recall: {metrics['train_recall']:.4f} | AUC: {metrics['train_auc']:.4f} | "
                        f"TPR: {tpr:.4f} | TNR: {tnr:.4f} | FPR: {fpr:.4f} | FNR: {fnr:.4f}")

        # if logs is not None:
        #     if "loss" in logs and "learning_rate" in logs:
        #         logger.info(f"[Train Step {state.global_step}] Loss: {logs['loss']:.4f} | LR: {logs['learning_rate']:.8f}")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None and 'eval_accuracy' in metrics:
            epoch = state.epoch or metrics.get("epoch", "N/A")
            tp = metrics.get("eval_true_positives", 0)
            fp = metrics.get("eval_false_positives", 0)
            tn = metrics.get("eval_true_negatives", 0)
            fn = metrics.get("eval_false_negatives", 0)
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
            logger.info(f"\n[Eval Epoch {epoch}] Accuracy: {metrics['eval_accuracy']:.4f} | F1: {metrics['eval_f1']:.4f} | Precision: {metrics['eval_precision']:.4f} | Recall: {metrics['eval_recall']:.4f} | AUC: {metrics['eval_auc']:.4f} | TPR: {tpr:.4f} | TNR: {tnr:.4f} | FPR: {fpr:.4f} | FNR: {fnr:.4f}")
            logger.info(f"[Eval Epoch {epoch}] Runtime: {metrics['eval_runtime']:.2f}s | Samples/sec: {metrics['eval_samples_per_second']:.2f} | Steps/sec: {metrics['eval_steps_per_second']:.2f}")
            if all(k in metrics for k in ["eval_loss", "eval_auc"]):
                logger.info(f"[Eval Epoch {epoch}] Loss: {metrics['eval_loss']:.4f} | AUC: {metrics['eval_auc']:.4f}")

# ------------------- Trainer -------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[
        EarlyStoppingCallback(early_stopping_patience=5),
        TrainLoggingCallback(train_eval_interval=0.25)  # Adjustable if needed
    ],
)

# ------------------- Train -------------------
logger.info("Starting training...")
trainer.train()

# ------------------- Final Test Evaluation (Manual Set) -------------------
logger.info("\n[Final Evaluation on Held-Out Manual Set] Running test metrics...")

# Preprocess and convert manual set
manual_dataset = Dataset.from_pandas(manual_df.rename(columns={"Malicious (0/1)": "label"}))
manual_dataset = manual_dataset.map(preprocess_function, batched=True)

test_metrics = trainer.evaluate(eval_dataset=manual_dataset, metric_key_prefix="test")

tp = test_metrics.get("test_true_positives", 0)
fp = test_metrics.get("test_false_positives", 0)
tn = test_metrics.get("test_true_negatives", 0)
fn = test_metrics.get("test_false_negatives", 0)
tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

logger.info(f"[Test Manual Set] Accuracy: {test_metrics['test_accuracy']:.4f} | "
            f"F1: {test_metrics['test_f1']:.4f} | Precision: {test_metrics['test_precision']:.4f} | "
            f"Recall: {test_metrics['test_recall']:.4f} | AUC: {test_metrics['test_auc']:.4f} | "
            f"TPR: {tpr:.4f} | TNR: {tnr:.4f} | FPR: {fpr:.4f} | FNR: {fnr:.4f}")
logger.info(f"[Test Manual Set] Runtime: {test_metrics['test_runtime']:.2f}s | "
            f"Samples/sec: {test_metrics['test_samples_per_second']:.2f} | Steps/sec: {test_metrics['test_steps_per_second']:.2f}")
if all(k in test_metrics for k in ["test_loss", "test_auc"]):
    logger.info(f"[Test Manual Set] Loss: {test_metrics['test_loss']:.4f} | AUC: {test_metrics['test_auc']:.4f}")


# ------------------- Save -------------------
model_path = "./deberta_lora_classifier"
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)
logger.info(f"Model saved to {model_path}")
