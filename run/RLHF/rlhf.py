from model_manager import load_model, generate_batch, fine_tune_model, save_checkpoint, update_seed_data_from_accepted_log
import pandas as pd
import os
import time

BATCH_DIR = "data/batches"
ACCEPTED_LOG = "data/accepted_log.csv"
REJECTED_LOG = "data/rejected_log.csv"

def initialize_logs():
    os.makedirs("data", exist_ok=True)
    os.makedirs(BATCH_DIR, exist_ok=True)
    if not os.path.exists(ACCEPTED_LOG):
        pd.DataFrame(columns=["Prompt", "Label", "Department", "SourcePrompt", "EditType", "BatchID"]).to_csv(ACCEPTED_LOG, index=False)
    if not os.path.exists(REJECTED_LOG):
        pd.DataFrame(columns=["Prompt", "Label", "Department", "SourcePrompt", "EditType", "BatchID"]).to_csv(REJECTED_LOG, index=False)

def run_pipeline():
    initialize_logs()
    model, tokenizer = load_model()

    batch_id = 1
    while True:
        # === STEP 1: Generate new batch ===
        print(f"\n[INFO] Generating batch {batch_id}...")
        batch_df = generate_batch(model, tokenizer, batch_size=10, batch_id=batch_id)
        batch_path = f"{BATCH_DIR}/batch_{batch_id}.csv"
        batch_df.to_csv(batch_path, index=False)
        print(f"[INFO] Batch saved to: {batch_path}")
        print("[INFO] Awaiting Streamlit UI feedback...")

        # === STEP 2: Wait until Streamlit logs feedback for current batch ===
        while True:
            if not os.path.exists(ACCEPTED_LOG):
                time.sleep(2)
                continue

            accepted_df = pd.read_csv(ACCEPTED_LOG)
            if not accepted_df.empty and accepted_df["BatchID"].astype(int).max() >= batch_id:
                break

            print("[WAITING] Waiting for feedback on batch", batch_id)
            time.sleep(3)

        # === STEP 3: Load full cumulative accepted + rejected logs ===
        accepted_df = pd.read_csv(ACCEPTED_LOG)
        rejected_df = pd.read_csv(REJECTED_LOG) if os.path.exists(REJECTED_LOG) else pd.DataFrame()

        if not accepted_df.empty or not rejected_df.empty:
            print(f"[TRAINING] Fine-tuning on cumulative feedback...")
            fine_tune_model(model, tokenizer, accepted_df, rejected_df)
            save_checkpoint(model, tokenizer)
            update_seed_data_from_accepted_log()
        else:
            print(f"[SKIP] No feedback found for batch {batch_id}. Skipping fine-tuning.")

        batch_id += 1

if __name__ == "__main__":
    run_pipeline()
