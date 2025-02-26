from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import logging
import pandas as pd
from datasets import Dataset
from transformers import TrainingArguments
from transformers import Trainer

INPUT_FILE = "../data/outputs/generated_prompts.csv"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

logger.info("Checking system hardware...")
gpu_available = torch.cuda.is_available()
num_gpus = torch.cuda.device_count()
gpu_name = torch.cuda.get_device_name(0) if gpu_available else "None"

logger.info(f"GPU Available: {gpu_available}")
logger.info(f"Number of GPUs: {num_gpus}")
logger.info(f"Using GPU: {gpu_name}")

device = "cuda" if gpu_available else "cpu"

logger.info("Loading dataset...")
df = pd.read_csv(INPUT_FILE)

# Convert DataFrame to Hugging Face Dataset
dataset = Dataset.from_pandas(df)

logger.info("Loading model...")
MODEL_NAME = "protectai/deberta-v3-small-prompt-injection-v2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    return tokenizer(examples["Prompt"], padding="max_length", truncation=True)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],  # Use "train" split or slice the data accordingly
    eval_dataset=tokenized_datasets["test"],    # Use "test" split or slice the data accordingly
)

classifier = pipeline(
  "text-classification",
  model=model,
  tokenizer=tokenizer,
  truncation=True,
  max_length=512,
  device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)

print(classifier("Your prompt injection is here"))
