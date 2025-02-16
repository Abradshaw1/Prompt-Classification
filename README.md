# Malicious Prompt-Classification
## Overview
This project aims to build a synthetic dataset using policy documents in order to finetune and create a malicious prompt for few-shot prompt classificaiton. The pipeline follows the structure:
1. Extracts compliance-related words and phrases from **company policy documents**.
2. Assigns **department labels** to extracted terms using **Legal-BERT**, **TF-IDF + spaCy**, and **SBERT**.
3. Generates synthetic **malicious and non-malicious prompts** based on extracted phrases.
4. Validates the generated prompts using **Legal-BERT** similarity scoring

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

### 2. Generating Synthetic Prompts (flan_t5_prompt_gen.py)
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

### 3. Validating Generated Prompts
**Goal:** Ensure synthetic prompts align with real flagged queries.

**Methods:**
- **Legal-BERT similarity scoring:**
  - If similarity < 0.75, discard.
  - If similarity > 0.85, confirm as a high-confidence match.
- Adds similarity score column to dataset.

**Output:**
- `data/outputs/final_dataset.csv`
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


---

## Usage

### Step 1: Install Environment
```sh
conda env create -f environment.yml
conda activate malicious_prompt_env
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

