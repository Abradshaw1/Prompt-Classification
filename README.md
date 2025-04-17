test commit
# Malicious Prompt-Classification
## Overview
This project aims to build a synthetic dataset using policy documents in order to finetune and create a malicious prompt for few-shot prompt classification. The pipeline follows the structure:
1. Extracts compliance-related words and phrases from **company policy documents**.
2. Assigns **department labels** to extracted terms using **Legal-BERT**, **TF-IDF + spaCy**, and **SBERT**.
3. Generates synthetic **malicious and non-malicious prompts** based on extracted phrases.
4. Validates the generated prompts using **Legal-BERT** similarity scoring.
5. Removes nonsense prompts and changes generated labels with **rule-based classification**.

## Pipeline

### 1. Preprocessing Policy Documents (preprocess_policy_docs.py)
**Goal:** Extract clean text from policy documents by removing metadata, disclaimers, and irrelevant content.

#### **Methods:**
- **PyMuPDF (`fitz`)**: Extracts text from PDFs.
- **Regular expressions (`re`)**: Removes metadata like "Page X of Y," effective dates, classification labels, and other non-relevant sections.

#### **Output:**
- Cleaned text files stored in `data/cleaned_policy_documents/`.

### 2. Extracting Malicious Terms
Extract keywords and phrases from policy documents.

Methods
- **Named Entity Recognition (NER) with Legal-BERT**: Extracts named entities related to HR, Legal, Ethics, and Compliance.
- **TF-IDF & spaCy Dependency Parsing**: Selects the most informative words from extracted phrases.
- **SBERT Similarity Matching**: Assigns department labels to extracted words based on similarity.

**Output:**
- `data/outputs/word_label_bank.csv`

**Structure of Word/Phrase Label Bank:**
```markdown
| Index | Department  | Word            | Phrase                                   | Flag (0/1) |
|-------|-------------|-----------------|------------------------------------------|------------|
| 1     | Legal       | insider-trading | Illegal insider trading knowledge        | 1          |
| 2     | HR          | harassment      | Unlawful workplace harassment report     | 1          |
| 3     | Security    | phishing        | Email phishing attempt                   | 1          |
```

---

### 3. Generating Synthetic Prompts (flan_t5_prompt_gen.py)
Generate both malicious and non-malicious prompts based on the extracted word-label pairs.

Methods:
- **FLAN-T5**: Zero-shot & fine-tunable prompt generation.
- **User-defined parameters:**
  - Department: (HR, Legal, Security, etc.)
  - Number of prompts to generate
  - Malicious vs. Non-Malicious Ratio

**Output:**
- `data/outputs/generated_prompts.csv`

**Structure of dataset:**
```markdown
| Prompt ID | Prompt                                       | Classification (0/1)| Department | Confidence Score | Source (Manual/Generated) |
|-----------|----------------------------------------------|---------------------|------------|------------------|---------------------------|
| 1         | Can I report a coworker for harassment?      | 1                   | HR         | 0.97             | Manual                    |
| 2         | Can I file a fake complaint against my boss? | 1                   | HR         | 0.92             | Generated                 |
| 3         | How do I optimize machine learning models?   | 0                   | None       | 0.99             | Manual                    |
```

---

### 4. Validating Generated Prompts (final_similarity_scoring.py, clean_add_validate_synthetic_prompts.ipynb)
**Goal:** Ensure synthetic prompts align with real flagged queries.

**Methods:**
- **Legal-BERT similarity scoring:**
  - If similarity < 0.00, discard.
- **Malicious vs. Non-malicious Label Confirmation**
  - If prompt contains malicious phrases and was labeled 0 non-malicious, relabel as 1 malicious.

**Output:**
- `data/outputs/validation_and_cleaning/step2_cleaned_data.csv`
- `data/outputs/validation_and_cleaning/subset_relabeled_prompts_with_rule.csv`
**Structure of final/optional dataset:**
```markdown
| Prompt ID | Prompt                                       | Classification (0/1)| Department | Confidence Score | Source (Manual/Generated) | Similarity Score |
|-----------|----------------------------------------------|---------------------|------------|------------------|---------------------------|------------------|
| 1         | Can I report a coworker for harassment?      | 1                   | HR         | 0.97             | Manual                    | 1.00             |
| 2         | Can I file a fake complaint against my boss? | 1                   | HR         | 0.92             | Generated                 | 0.85             |
| 3         | How do I optimize machine learning models?   | 0                   | None       | 0.99             | Manual                    | 1.00             |
```

---

## Python Scripts in models/

| Script                  | Purpose                                         |
|-------------------------|-------------------------------------------------|
| `preprocess_policy_documents.py`|Strip headings, alphanumeric chars and clean |
| `build_word_bank.py`   | Extracts compliance terms using TF-IDF + Regex.  |
| `flan_t5_prompt_gen.py` | Runs FLAN-T5 to generate synthetic prompts.     |
| `final_similarity_scoring.py` | Computes Legal-BERT similarity between prompts and finalized dataset. |
| `clean_add_validate_synthetic_prompts.ipynb` | Removes nonsense prompts and validates prompt labels. |


---

## Usage

### Step 1: Install Environment
```sh
conda env create -f environment.yml
conda activate malicious_prompt_env
python -m spacy download en_core_web_sm (download)
```
### Step 2: Add Policy Documents
```sh
add files to policy documents folder
```
### Step 3: Clean Policy Documents
```sh
python preprocess_policy_documents.py
```
### Step 4: Generate the Word/Phrase-Label Bank
```sh
python build_word_bank.py
```
**Output:** `data/outputs/word_label_bank.csv`

### Step 5: Generate Synthetic Prompts
```sh
python models/flan_t5_prompt_gen.py
```
**Output:** `data/outputs/generated_prompts.csv`

### Step 6: Validate Prompts with Legal-BERT
```sh
python build_final_ds.py --input data/outputs/generated_prompts.csv
```
**Output:** `data/outputs/final_dataset.csv`

### Step 7: Create Prompt Embeddings with MPNet
```sh

```
**Output:** `data/outputs/FINAL_validated_prompts_with_similarity_MPnet.csv`

### Step 8: Clean data by similarity score and prompt label.
```sh

```
**Output:** `data/outputs/validation_and_cleaning/step2_cleaned_data.csv`

---

## Repository Structure
```bash
malicious_prompt_classification
│── data/
│   ├── policy_documents/            # Raw policy documents (INPUT)
│   ├── cleaned_policy_documents/    # Processed text (OUTPUT)
│   ├── outputs/                     # Stores generated CSV files
│   │   ├── word_label_bank.csv      # Extracted words & phrases
│   │   ├── generated_prompts.csv    # Generated prompts
│   │   ├── final_dataset.csv         # Final validated dataset
│
│── run/
│   ├── preprocess_policy_docs.py    # Cleans and extracts text from PDFs
│   ├── build_word_bank.py           # Identifies key phrases & assigns labels
│   ├── flan_t5_prompt_gen.py       # Generates synthetic prompts
│   ├── build_final_ds.py           # Legal-BERT similarity checker for final data set
│
│── notebooks/
│   ├── 1_scrape_documents.ipynb     # Interactive notebook for document processing
│
│── environment.yml                  # Conda environment setup
│── README.md                        # Project documentation
```

## RLHF: Reinforcement Learning from Human Feedback

This module implements a lightweight RLHF loop that fine-tunes an open-source language model to iteratively improve prompt classification and generation. The goal is to generate realistic internal employee prompts—some benign, others risky or policy-violating—and refine the model using structured human feedback.

### Model & Setup

- **Base model**: `microsoft/phi-2` (quantized with bitsandbytes)
- **Training adapter**: LoRA via `peft`
- **Quantization**: 8-bit (fp16 compute)
- **Frameworks**: PyTorch, HuggingFace Transformers, Streamlit (UI)

### File Overview

| Script/File              | Purpose                                                                 |
|--------------------------|-------------------------------------------------------------------------|
| `model_manager.py`       | Loads the base model, generates prompt batches, performs weighted fine-tuning, and manages checkpoints. |
| `rlhf.py`                | Orchestrates the full feedback loop: batch generation → feedback waiting → model fine-tuning → checkpoint saving. |
| `review_ui.py`           | Streamlit-based human-in-the-loop UI for accepting/rejecting/editing prompt-label pairs. |
| `data/seed_prompts.csv`  | Initial seed prompts used for few-shot generation. Updated after every round with accepted/edited feedback. |
| `data/accepted_log.csv`  | Stores cumulative accepted prompts from all batches.                    |
| `data/rejected_log.csv`  | Stores cumulative rejected prompts from all batches.                    |
| `data/batches/`          | Contains `batch_1.csv`, `batch_2.csv`, etc. Each holds a round of generated prompts for review. |
| `model_checkpoints/`     | Stores saved model/tokenizer weights after each batch-level fine-tuning pass. |

### Running the RLHF Pipeline

1. **Open the Streamlit UI**: This launches streamlit UI as a local host
```bash
   streamlit run review_ui.py
```

2. **Launch the backend loop in a separate terminal**: This will generate prompt batches, wait for feedback, and trigger fine-tuning.
```bash
   python rlhf.py
```
3. **Refresh the UI page**

4. **The user can now review prompts**:
- Accept or reject each prompt
- Edit the label or department field if necessary
- Click Save Feedback and Trigger Training

5. **Model Training**
- Fine-tuning will automatically begin once feedback is saved
- The model is trained cumulatively on all accepted and rejected feedback across all prior batches
- After training, the updated model is saved to model_checkpoints/

### Notes on Generation & Training

- Prompt generation is conditioned on a small sample of few-shot examples pulled from the evolving `seed_prompts.csv`
- All generated departments are constrained to the following:
```bash
  Legal, HR, Security, Safety, Ethics and Compliance, Government Relations, None
```
- The model is fine-tuned cumulatively on all accepted and rejected prompts from prior batches
- Prompts are weighted during training to simulate reward-based feedback:
  - **Accepted prompts** receive a full reward weight of `1.0`
  - **Edited prompts** are slightly downweighted (`0.8`)
  - **Rejected prompts** are retained as negative signals with a low weight (`0.1`)
- This mimics **Reinforcement Learning from Human Feedback (RLHF)** without requiring reward models or PPO
- Training uses the **AdamW** optimizer and runs entirely on **GPU** (if available)

