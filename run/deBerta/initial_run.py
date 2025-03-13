"""initial_run"""

import logging
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import Dataset
import torch.nn.functional as F
from peft import get_peft_model, LoraConfig, TaskType

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DeBERTa Initial Runs")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")
df = pd.read_csv("./data/FINAL_validated_prompts_with_similarity_deBERTa.csv")
logger.info(f"Loaded dataset with {len(df)} rows.")

MODEL_NAME_1 = "microsoft/deberta-v3-small"
MODEL_NAME_2 = "microsoft/deberta-v3-base"
MODEL_NAME_3 = "microsoft/deberta-v3-large"


def preprocess_function(examples):
    logger.info("Preparing data...")
    return tokenizer(examples["Prompt"],
                     truncation=True,
                     padding='max_length',
                     max_length=256) \
        .to(device)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


def trainer():
    logger.info("Training model...")

    training_args = TrainingArguments(
        output_dir="./results",  # Stores checkpoints per epoch
        eval_strategy="epoch",
        save_strategy="epoch",  # Saves after every epoch
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=1,
        weight_decay=0.01,
        logging_dir="./deberta_logs",  # Logs will be saved here
        logging_strategy="steps",  # Log metrics every few steps
        logging_steps=50,  # Adjust this based on dataset size
        save_total_limit=3,  # Keeps only the last 3 checkpoints
        load_best_model_at_end=True,
        report_to="none"  # Disable WandB (if using)
    )

    return Trainer(
            model=model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=val_data,
            compute_metrics=compute_metrics
            )


def predict(data, trainer):
    logger.info("Evaluating model...")
    predictions = trainer.predict(data)
    logits = predictions.predictions
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1)[:, 1]  # Get positive class probability
    labels = np.array(predictions.label_ids)
    return labels, probs.numpy()


base_model = AutoModelForSequenceClassification \
    .from_pretrained(MODEL_NAME_1, num_labels=2) \
    .to(device)

lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,  # Low-rank factor
    lora_alpha=16,  # Scaling factor
    lora_dropout=0.1,
)

# Apply LoRA
model = get_peft_model(base_model, lora_config).to(device)
logger.info("DeBERTa LoRA model initialized.")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_1, use_fast=False)

df["Malicious (0/1)"] = df["Malicious (0/1)"].astype(int)
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["Prompt"].tolist(),
    df["Malicious (0/1)"].tolist(),
    test_size=0.2,
    random_state=42
    )

train_data = Dataset.from_dict({"Prompt": train_texts, "label": train_labels})
val_data = Dataset.from_dict({"Prompt": val_texts, "label": val_labels})
train_data = train_data.map(preprocess_function, batched=True)
val_data = val_data.map(preprocess_function, batched=True)


trainer = trainer()
trainer.train()

model_path = "./deberta_lora_classifier"
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)
logger.info(f"Model saved to {model_path}")

labels, probs = predict(val_data, trainer)

fpr, tpr, _ = roc_curve(labels, probs)
roc_auc = auc(fpr, tpr)

cm = confusion_matrix(val_labels, labels)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")  # Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.savefig("roc_curve.png", dpi=300, bbox_inches="tight")


# Plot confusion matrix

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["0", "1"], yticklabels=["0", "1"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
