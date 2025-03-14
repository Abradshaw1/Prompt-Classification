"""initial_run"""

import logging
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import Dataset
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


def predict(data, trainer):
    logger.info("Evaluating model...")
    predictions = trainer.predict(data)
    logits = predictions.predictions
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1)[:, 1]  # Get positive class probability
    labels = np.array(predictions.label_ids)
    return labels, probs.numpy()


model = AutoModelForSequenceClassification \
    .from_pretrained(MODEL_NAME_1, num_labels=2) \
    .to(device)
logger.info("DeBERTa model initialized.")

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
plt.savefig("roc_curve_none.png", dpi=300, bbox_inches="tight")


# Plot confusion matrix

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["0", "1"], yticklabels=["0", "1"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix_none.png", dpi=300, bbox_inches="tight")
