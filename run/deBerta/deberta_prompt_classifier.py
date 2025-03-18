"""DeBERTa-based prompt classification model with LoRA fine-tuning."""

import logging
from typing import Tuple, Dict, Any
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import Dataset
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, precision_score,
    recall_score, f1_score, roc_auc_score
)
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments
)
from peft import get_peft_model, LoraConfig, TaskType

# Constants
MODEL_NAME = "microsoft/deberta-v3-small"
MAX_LENGTH = 256
BATCH_SIZE = 8
NUM_EPOCHS = 1
WEIGHT_DECAY = 0.01
MODEL_SAVE_PATH = "./deberta_lora_classifier"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DeBERTa Initial Runs")

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")


class PromptClassifier:
    def __init__(self):
        """Initialize the classifier with model, tokenizer, and LoRA config."""
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME, use_fast=False
        )
        self.base_model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=2
        ).to(device)

        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
        )
        self.model = get_peft_model(self.base_model, lora_config).to(device)
        logger.info("DeBERTa LoRA model initialized.")

    def preprocess_data(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess the input data using the tokenizer."""
        return self.tokenizer(
            examples["Prompt"],
            truncation=True,
            padding='max_length',
            max_length=MAX_LENGTH
        ).to(device)

    def prepare_datasets(self, df: pd.DataFrame) -> Tuple[Dataset, Dataset]:
        """Prepare training and validation datasets."""
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            df["Prompt"].tolist(),
            df["Malicious (0/1)"].tolist(),
            test_size=0.2,
            random_state=42
        )

        train_data = Dataset.from_dict({
            "Prompt": train_texts,
            "label": train_labels
        })
        val_data = Dataset.from_dict({
            "Prompt": val_texts,
            "label": val_labels
        })

        train_data = train_data.map(self.preprocess_data, batched=True)
        val_data = val_data.map(self.preprocess_data, batched=True)

        return train_data, val_data

    def compute_metrics(
        self, eval_pred: Tuple[np.ndarray, np.ndarray]
    ) -> Dict[str, float]:
        """Compute evaluation metrics."""
        logits, labels = eval_pred
        predictions = torch.argmax(torch.tensor(logits), dim=-1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='binary'
        )
        acc = accuracy_score(labels, predictions)
        return {
            "accuracy": acc,
            "f1": f1,
            "precision": precision,
            "recall": recall
        }

    def train(self, train_data: Dataset, val_data: Dataset) -> None:
        """Train the model."""
        training_args = TrainingArguments(
            output_dir="./results",
            eval_strategy="epoch",
            save_strategy="epoch",
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            num_train_epochs=NUM_EPOCHS,
            weight_decay=WEIGHT_DECAY,
            logging_dir="./deberta_logs",
            logging_strategy="steps",
            logging_steps=50,
            save_total_limit=3,
            load_best_model_at_end=True,
            report_to="none"
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=val_data,
            compute_metrics=self.compute_metrics
        )

        trainer.train()
        self.save_model()

    def save_model(self) -> None:
        """Save the trained model and tokenizer."""
        self.model.save_pretrained(MODEL_SAVE_PATH)
        self.tokenizer.save_pretrained(MODEL_SAVE_PATH)
        logger.info(f"Model saved to {MODEL_SAVE_PATH}")

    def evaluate(
        self, val_data: Dataset
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate the model and return predictions."""
        test_trainer = Trainer(self.model)
        predictions = test_trainer.predict(val_data)

        logits = torch.tensor(predictions.predictions)
        probs = torch.nn.functional.softmax(logits, dim=-1).numpy()
        pred_labels = np.argmax(probs, axis=-1)

        return predictions.label_ids, pred_labels, probs[:, 1]

    def plot_metrics(
        self, true_labels: np.ndarray, pred_labels: np.ndarray,
        probs: np.ndarray
    ) -> None:
        """Plot ROC curve and confusion matrix."""
        # ROC Curve
        fpr, tpr, _ = roc_curve(true_labels, probs)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(
            fpr, tpr, color="blue", lw=2,
            label=f"ROC Curve (AUC = {roc_auc:.2f})"
        )
        plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (ROC) Curve")
        plt.legend(loc="lower right")
        plt.savefig(
            "roc_curve_baseline_full.png",
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()

        # Confusion Matrix
        cm = confusion_matrix(true_labels, pred_labels)
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["0", "1"], yticklabels=["0", "1"]
        )
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")
        plt.savefig(
            "confusion_matrix_baseline_full.png",
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()


def main():
    """Main execution function."""
    # Load data
    df = pd.read_csv(
        "./data/FINAL_validated_prompts_with_similarity_deBERTa.csv"
    )
    df["Malicious (0/1)"] = df["Malicious (0/1)"].astype(int)
    logger.info(f"Loaded dataset with {len(df)} rows.")

    # Initialize and train classifier
    classifier = PromptClassifier()
    train_data, val_data = classifier.prepare_datasets(df)
    classifier.train(train_data, val_data)

    # Evaluate and plot results
    true_labels, pred_labels, probs = classifier.evaluate(val_data)

    # Print metrics
    print(f"Accuracy: {accuracy_score(true_labels, pred_labels):.4f}")
    print(f"Precision: {precision_score(true_labels, pred_labels):.4f}")
    print(f"Recall: {recall_score(true_labels, pred_labels):.4f}")
    print(f"F1 Score: {f1_score(true_labels, pred_labels):.4f}")
    print(f"AUC Score: {roc_auc_score(true_labels, probs):.4f}")

    # Plot metrics
    classifier.plot_metrics(true_labels, pred_labels, probs)


if __name__ == "__main__":
    main()
