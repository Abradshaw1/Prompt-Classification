import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import pandas as pd
import difflib
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

def is_similar(a, b, threshold=0.9):
    return difflib.SequenceMatcher(None, a, b).ratio() > threshold

def get_all_previous_prompts(batch_dir="data/batches"):
    all_prompts = set()
    if not os.path.exists(batch_dir):
        return all_prompts
    for fname in os.listdir(batch_dir):
        if fname.startswith("batch_") and fname.endswith(".csv"):
            df = pd.read_csv(os.path.join(batch_dir, fname))
            if "Prompt" in df.columns:
                all_prompts.update(df["Prompt"].dropna().tolist())
    return all_prompts



def generate_batch(model, tokenizer, batch_size=20, batch_id=0):
    print("Generating batch of prompts...")
    df = pd.read_csv(SEED_DATA_PATH).dropna(subset=["Prompt", "Malicious Label 0/1", "Department"])
    df = df.rename(columns={"Malicious Label 0/1": "Label"})

    # Sliding window: use last 30 seed prompts, shuffled
    recent_few_shots = df.tail(30).sample(frac=1).reset_index(drop=True)
    few_shot_text = "\n\n".join([
        f"Prompt: {row['Prompt']}\nLabel: {row['Label']}\nDepartment: {row['Department']}"
        for _, row in recent_few_shots.iterrows()
    ])
    print(f"[FEW-SHOT] Loaded {len(recent_few_shots)} seed examples for in-context prompting.")
    print("[FEW-SHOT] First seed example:\n")
    print(f"Prompt: {recent_few_shots.iloc[0]['Prompt']}\nLabel: {recent_few_shots.iloc[0]['Label']}\nDepartment: {recent_few_shots.iloc[0]['Department']}\n")



    allowed_depts = ['Legal', 'HR', 'Security', 'Safety', 'Ethics and Compliance', 'Government Relations', 'None']
    allowed_depts_str = ", ".join(allowed_depts)

    prompt_template = f"""
You are the world's best linguist trying to generating the most human prompt-label-department triplets used in an internal employee-facing system.
Some prompts should be benign (Label: 0), and others should be malicious, policy-violating, or risky (Label: 1).
Ensure prompts resemble human-written content from diverse categories:
- formal questions, casual internal queries, policy references, HR/legal topics,
- internal error messages, vague/sensitive or inappropriate phrasing, etc.
- code snippets, technical jargon, and actual lines of code wiht functions, error messages and terminal commands.
- questions about the Parsons company, its policies, or its products.
- ensure that all prompts in the batch are unique and different lengths.

The department labels should be one of the following: {allowed_depts_str}.
Here are examples:

{few_shot_text}

Now generate {batch_size} new realistic examples:
- Prompt: ...
- Label: 0 or 1
- Department: ...
"""


    prior_prompts = get_all_previous_prompts()

    # Iteratively generate until we hit a batch of unique prompts
    batch_prompts = []
    attempts = 0
    MAX_ATTEMPTS = 10
    while len(batch_prompts) < batch_size and attempts < MAX_ATTEMPTS:
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

        for section in sections:
            try:
                lines = section.strip().split("\n")
                prompt = lines[0].strip()
                label = int(lines[1].replace("Label:", "").strip())
                dept = lines[2].replace("Department:", "").strip()

                # Skip if prompt too similar to previous ones
                if any(is_similar(prompt, old) for old in prior_prompts.union(p["Prompt"] for p in batch_prompts)):
                    print(f"[SKIP] Duplicate prompt skipped: {prompt[:60]}...")
                    continue

                batch_prompts.append({
                    "Prompt": prompt,
                    "Label": label,
                    "Department": dept,
                    "SourcePrompt": prompt,
                    "EditType": "generated"
                })

                if len(batch_prompts) == batch_size:
                    break
            except Exception:
                continue

        attempts += 1
        if attempts == MAX_ATTEMPTS:
            print(f"[WARNING] Only collected {len(batch_prompts)}/{batch_size} unique prompts after {MAX_ATTEMPTS} tries.")

    print(f"[BATCH COMPLETE] Collected {len(batch_prompts)} prompts out of requested {batch_size}.")

    return pd.DataFrame(batch_prompts)


    # input_ids = tokenizer(prompt_template, return_tensors="pt").input_ids.to(model.device)

    # outputs = model.generate(
    #     input_ids,
    #     max_new_tokens=512,
    #     do_sample=True,
    #     temperature=0.8,
    #     top_p=0.9,
    #     pad_token_id=tokenizer.eos_token_id
    # )

    # result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # sections = result.split("Prompt:")[1:]

    # prompts, labels, depts = [], [], []
    # for section in sections:
    #     try:
    #         lines = section.strip().split("\n")
    #         prompt = lines[0].strip()
    #         label = int(lines[1].replace("Label:", "").strip())
    #         dept = lines[2].replace("Department:", "").strip()
    #         prompts.append(prompt)
    #         labels.append(label)
    #         depts.append(dept)
    #     except Exception:
    #         continue

    # valid_len = min(len(prompts), len(labels), len(depts))

    # return pd.DataFrame({
    #     "Prompt": prompts[:valid_len],
    #     "Label": labels[:valid_len],
    #     "Department": depts[:valid_len],
    #     "SourcePrompt": prompts[:valid_len],
    #     "EditType": ["generated"] * valid_len
    # })


def fine_tune_model(model, tokenizer, accepted_df, rejected_df, current_batch_id, window_size=3):
    print("Fine-tuning model with new feedback (last N batches)...")

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

    # Slice accepted/rejected data for only the last N batches
    accepted_recent = accepted_df[accepted_df.BatchID >= current_batch_id - window_size]
    print(f"[TRAINING DATA] Using accepted prompts from BatchID >= {current_batch_id - window_size}")
    print(f"[TRAINING DATA] Total accepted rows: {len(accepted_recent)}")
    rejected_recent = rejected_df[rejected_df.BatchID >= current_batch_id - window_size] if not rejected_df.empty else pd.DataFrame()
    print(f"[TRAINING DATA] Total rejected rows: {len(rejected_recent)}")

    all_rows = format_rows(accepted_recent) + format_rows(rejected_recent)
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

    # Only use new prompts from the most recent batch
    if "BatchID" not in accepted_df.columns:
        print("[WARN] BatchID column not found.")
        return

    latest_batch_id = accepted_df["BatchID"].max()
    new_prompts = accepted_df[accepted_df["BatchID"] == latest_batch_id][["Prompt", "Malicious Label 0/1", "Department"]]
    print(f"[SEED UPDATE] Latest batch ID: {latest_batch_id}")
    print(f"[SEED UPDATE] Appending {len(new_prompts)} prompts from Batch {latest_batch_id} to seed data.")


    # Append-only: do not deduplicate or overwrite
    if os.path.exists(SEED_DATA_PATH):
        seed_df = pd.read_csv(SEED_DATA_PATH)
        seed_df = pd.concat([seed_df, new_prompts], ignore_index=True)
    else:
        seed_df = new_prompts

    seed_df.to_csv(SEED_DATA_PATH, index=False)
    print(f"[INFO] Appended {len(new_prompts)} prompts from batch {latest_batch_id} to seed data.")
