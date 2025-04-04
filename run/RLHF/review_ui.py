import streamlit as st
import pandas as pd
import os
import time

BATCH_DIR = "data/batches"
ACCEPTED_LOG = "data/accepted_log.csv"
REJECTED_LOG = "data/rejected_log.csv"
SESSION_EXPORT_PATH = "data/user_session_prompts.csv"

st.set_page_config(page_title="RLHF Prompt Review", layout="wide")
st.title("RLHF Prompt Review")

# == Get latest batch ==
def get_latest_batch():
    if not os.path.exists(BATCH_DIR):
        st.error("No batch directory found.")
        st.stop()
    batch_files = sorted([f for f in os.listdir(BATCH_DIR) if f.startswith("batch_") and f.endswith(".csv")])
    if not batch_files:
        st.warning("No batches to review yet.")
        st.stop()
    latest = batch_files[-1]
    return os.path.join(BATCH_DIR, latest), int(latest.split("_")[1].split(".")[0])

batch_path, batch_id = get_latest_batch()
proposed_df = pd.read_csv(batch_path)

if "last_batch_id" not in st.session_state:
    st.session_state["last_batch_id"] = batch_id

if batch_id != st.session_state["last_batch_id"]:
    for k in list(st.session_state.keys()):
        if k.startswith("accept_") or k.startswith("reject_"):
            del st.session_state[k]
    st.session_state["last_batch_id"] = batch_id

st.info(f"Reviewing Batch {batch_id} → `{batch_path}`")

accepted, rejected = [], []

st.markdown("---")

for idx, row in proposed_df.iterrows():
    with st.expander(f"Prompt #{idx+1}", expanded=True):
        prompt = st.text_area("Prompt", value=row["Prompt"], key=f"prompt_{idx}")
        label = st.selectbox("Label", [0, 1], index=row["Label"], key=f"label_{idx}")
        dept = st.text_input("Department", value=row["Department"], key=f"dept_{idx}")

        col1, col2 = st.columns([1, 1])
        with col1:
            accept = st.checkbox("Accept", key=f"accept_{idx}")
        with col2:
            reject = st.checkbox("Reject", key=f"reject_{idx}")

        if accept and not reject:
            edit_type = "edited" if prompt != row["Prompt"] or int(label) != row["Label"] or dept != row["Department"] else "accepted"
            accepted.append({
                "Prompt": prompt,
                "Label": int(label),
                "Department": dept,
                "SourcePrompt": row["Prompt"],
                "EditType": edit_type,
                "BatchID": batch_id
            })
        elif reject and not accept:
            rejected.append({
                "Prompt": prompt,
                "Label": int(label),
                "Department": dept,
                "SourcePrompt": row["Prompt"],
                "EditType": "rejected",
                "BatchID": batch_id
            })

# === Save & Trigger Training ===
if st.button("Save Feedback and Trigger Training"):
    accepted_df = pd.DataFrame(accepted)
    rejected_df = pd.DataFrame(rejected)

    with st.spinner("Fine-tuning model on current batch..."):
        if not accepted_df.empty:
            accepted_df.to_csv(ACCEPTED_LOG, mode="a", index=False, header=not os.path.exists(ACCEPTED_LOG))
        if not rejected_df.empty:
            rejected_df.to_csv(REJECTED_LOG, mode="a", index=False, header=not os.path.exists(REJECTED_LOG))

        # Wait for backend to complete training and advance batch
        next_batch_ready = False
        while not next_batch_ready:
            batch_files = sorted([f for f in os.listdir(BATCH_DIR) if f.startswith("batch_")])
            if len(batch_files) > batch_id:
                next_batch_ready = True
            else:
                time.sleep(3)

    st.success("Training complete! Reloading next batch...")

# === Export session prompts ===
if st.button("Save All Prompts From Session"):
    # Load prior session if exists
    session_df = pd.DataFrame()
    if os.path.exists(ACCEPTED_LOG):
        session_df = pd.read_csv(ACCEPTED_LOG)
        session_df = session_df[session_df["EditType"].isin(["accepted", "edited"])]

    # Append current session's in-memory selections
    current_df = pd.DataFrame(accepted + rejected)  # both accepted and rejected, formatted same as log
    combined = pd.concat([session_df, current_df], ignore_index=True)

    # Drop duplicates in case anything overlaps
    combined.drop_duplicates(subset=["Prompt", "Label", "Department", "EditType", "BatchID"], inplace=True)
    combined.to_csv(SESSION_EXPORT_PATH, index=False)
    st.success(f"Session prompts exported to `{SESSION_EXPORT_PATH}`.")
