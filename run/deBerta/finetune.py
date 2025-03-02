import os
import torch
import pandas as pd
import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DeBERTa LoRA Fine-Tuning")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")
df = pd.read_csv("../../data/FINAL_validated_prompts_with_similarity.csv")
logger.info(f"Loaded dataset with {len(df)} rows.")

# we may want to cosnider a big boy bert model, since we have enough trnaing data mabyye deBERTa-large
MODEL_NAME = "protectai/deberta-v3-small-prompt-injection-v2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess_function(examples):
    return tokenizer(examples["Prompt"], truncation=True, padding='max_length', max_length=256)

df["Malicious (0/1)"] = df["Malicious (0/1)"].astype(int)
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["Prompt"].tolist(), df["Malicious (0/1)"].tolist(), test_size=0.2, random_state=42
)

train_data = Dataset.from_dict({"Prompt": train_texts, "label": train_labels})
val_data = Dataset.from_dict({"Prompt": val_texts, "label": val_labels})
train_data = train_data.map(preprocess_function, batched=True)
val_data = val_data.map(preprocess_function, batched=True)

base_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(device)

# LoRA config
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,  # Low-rank factor
    lora_alpha=16,  # Scaling factor
    lora_dropout=0.1,
)

# Apply LoRA
model = get_peft_model(base_model, lora_config).to(device)
logger.info("DeBERTa LoRA model initialized.")

# search the space here
training_args = TrainingArguments(
    output_dir="./results",  # Stores checkpoints per epoch
    evaluation_strategy="epoch",
    save_strategy="epoch",  # Saves after every epoch
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir="./deberta_logs",  # Logs will be saved here
    logging_strategy="steps",  # Log metrics every few steps
    logging_steps=50,  # Adjust this based on dataset size
    save_total_limit=3,  # Keeps only the last 3 checkpoints
    load_best_model_at_end=True,
    report_to="none",  # Disable WandB (if using)
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    compute_metrics=compute_metrics,
)

logger.info("Starting Training...")
trainer.train()
model_path = "./deberta_lora_classifier"
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)
logger.info(f"Model saved to {model_path}")
