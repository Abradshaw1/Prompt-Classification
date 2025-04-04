import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import pandas as pd
import os

SEED_DATA_PATH = "data/seed_prompts.csv"


def load_model(model_name="microsoft/phi-2", checkpoint_dir="model_checkpoints/latest"):
    print("Loading base model...")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        device_map = {"": 0}
    else:
        device = torch.device("cpu")
        device_map = {"": "cpu"}

    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=False,
        load_in_8bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf8"
    )

    if os.path.exists(checkpoint_dir):
        print(f"[INFO] Loading model from checkpoint: {checkpoint_dir}")
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_dir,
            quantization_config=bnb_config,
            device_map=device_map
        )
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, use_fast=True)
    else:
        print("[INFO] No checkpoint found. Loading base model.")
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=device_map
        )

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, config)
    return model, tokenizer


def generate_batch(model, tokenizer, batch_size=10, batch_id=0):
    print("Generating batch of prompts...")
    df = pd.read_csv(SEED_DATA_PATH).dropna(subset=["Prompt", "Malicious Label 0/1", "Department"])

    df = df.rename(columns={
        "Malicious Label 0/1": "Label",
        "Department": "Department",
        "Prompt": "Prompt"
    })

    few_shots = df.sample(n=min(5, len(df)))
    few_shot_text = "\n\n".join([
        f"Prompt: {row['Prompt']}\nLabel: {row['Label']}\nDepartment: {row['Department']}"
        for _, row in few_shots.iterrows()
    ])

    allowed_depts = ['Legal', 'HR', 'Security', 'Safety', 'Ethics and Compliance', 'Government Relations', 'None']
    allowed_depts_str = ", ".join(allowed_depts)

    prompt_template = f"""
You are generating synthetic prompt-label-department triplets used in an internal employee-facing system.
Some prompts should be benign (Label: 0), and others should be malicious, policy-violating, or risky (Label: 1).
Ensure prompts resemble human-written content from diverse categories:
- formal questions, casual internal queries, policy references, HR/legal topics,
- internal error messages, vague/sensitive or inappropriate phrasing, etc.
- code snippets, technical jargon, and actual lines of code wiht functions, error messages and terminal commands.
- ensure that all prompts in the batch are unique and different lengths.

The department labels should be one of the following: {allowed_depts_str}.
Here are examples:

{few_shot_text}

Now generate {batch_size} new realistic examples:
- Prompt: ...
- Label: 0 or 1
- Department: ...
"""

    input_ids = tokenizer(prompt_template, return_tensors="pt").input_ids.to(model.device)

    outputs = model.generate(
        input_ids,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    sections = result.split("Prompt:")[1:]

    prompts, labels, depts = [], [], []
    for section in sections:
        try:
            lines = section.strip().split("\n")
            prompt = lines[0].strip()
            label = int(lines[1].replace("Label:", "").strip())
            dept = lines[2].replace("Department:", "").strip()
            prompts.append(prompt)
            labels.append(label)
            depts.append(dept)
        except Exception:
            continue

    valid_len = min(len(prompts), len(labels), len(depts))

    return pd.DataFrame({
        "Prompt": prompts[:valid_len],
        "Label": labels[:valid_len],
        "Department": depts[:valid_len],
        "SourcePrompt": prompts[:valid_len],
        "EditType": ["generated"] * valid_len
    })


def fine_tune_model(model, tokenizer, accepted_df, rejected_df):
    print("Fine-tuning model with new feedback...")

    def format_rows(df):
        weighted = []
        for _, row in df.iterrows():
            if row.EditType == "accepted":
                prompt_text = row.Prompt
                weight = 1.0
            elif row.EditType == "edited":
                prompt_text = row.SourcePrompt
                weight = 0.8
            else:
                prompt_text = row.Prompt
                weight = 0.1

            full_text = f"Prompt: {prompt_text}\nLabel: {row.Label}\nDepartment: {row.Department}"
            weighted.append({"text": full_text, "weight": weight})

        return weighted

    all_rows = format_rows(accepted_df) + format_rows(rejected_df)
    dataset = Dataset.from_list(all_rows)

    def tokenize_and_mask(example):
        tokens = tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)
        tokens["labels"] = tokens["input_ids"]
        tokens["weight"] = example["weight"]
        return tokens

    tokenized_dataset = dataset.map(tokenize_and_mask, remove_columns=["text"])
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    args = TrainingArguments(
        output_dir="tmp_trainer",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        num_train_epochs=1,
        learning_rate=5e-5,
        fp16=True,
        logging_steps=10,
        save_strategy="no"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        optimizers=(optimizer, None)
    )

    trainer.train()


def save_checkpoint(model, tokenizer):
    ckpt_path = "model_checkpoints/latest"
    os.makedirs(ckpt_path, exist_ok=True)
    model.save_pretrained(ckpt_path)
    tokenizer.save_pretrained(ckpt_path)
    print(f"Model saved to {ckpt_path}")


def update_seed_data_from_accepted_log():
    ACCEPTED_LOG = "data/accepted_log.csv"
    if not os.path.exists(ACCEPTED_LOG):
        return

    accepted_df = pd.read_csv(ACCEPTED_LOG)
    accepted_df = accepted_df[accepted_df["EditType"].isin(["accepted", "edited"])]
    accepted_df = accepted_df.rename(columns={"Label": "Malicious Label 0/1"})

    if os.path.exists(SEED_DATA_PATH):
        seed_df = pd.read_csv(SEED_DATA_PATH)
        updated_df = pd.concat([seed_df, accepted_df[["Prompt", "Malicious Label 0/1", "Department"]]], ignore_index=True)
        updated_df.drop_duplicates(subset=["Prompt", "Malicious Label 0/1", "Department"], inplace=True)
    else:
        updated_df = accepted_df[["Prompt", "Malicious Label 0/1", "Department"]]

    updated_df.to_csv(SEED_DATA_PATH, index=False)
    print(f"[INFO] Seed data updated with {len(updated_df)} new prompts.")
