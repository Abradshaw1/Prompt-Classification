# Malicious Prompt-Classification
## Overview
This project automates malicious prompt classification and synthetic prompt generation for ParsonsGPT. The pipeline extracts compliance-related words and phrases from company policy documents, generates flagged and non-flagged prompts, and validates them using Legal-BERT against existing labeled prompts.

### 1. Extracting Malicious Terms (1_scrape_documents.ipynb)
Extract keywords and phrases from policy documents.

Methods
- **TF-IDF + Regex** → Extracts compliance-related terms.
- **spaCy NER** → Detects HR, Legal, and Ethics-related entities.
- **Legal-BERT** → Assigns department labels and flagging status based on extracted words.

**Output:**
- `data/outputs/word_label_bank.csv`

**Structure of Word/Phrase Label Bank:**
```markdown
| Index | Department  | Word            | Phrase                                   | Flag (0/1) |
|-------|-------------|-----------------|------------------------------------------|------------|
| 1     | Legal       | insider trading | Illegal insider trading knowledge        | 1          |
| 2     | HR          | harassment      | Unlawful workplace harassment report     | 1          |
| 3     | Security    | phishing        | Email phishing attempt                   | 1          |
```

---

### 2. Generating Synthetic Prompts (2_generate_prompts.ipynb)
Generate both malicious and non-malicious prompts based on the extracted word-label pairs.

**Methods:**
- **FLAN-T5** → Zero-shot & fine-tunable prompt generation.
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

### 3. Validating Generated Prompts (`3_validate_prompts.ipynb`)
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

## Python Scripts in `models/`

| Script                  | Purpose                                          |
|-------------------------|--------------------------------------------------|
| `bert_similarity.py`    | Computes Legal-BERT similarity between prompts. |
| `flan_t5_prompt_gen.py` | Runs FLAN-T5 to generate synthetic prompts.     |
| `tfidf_extraction.py`   | Extracts compliance terms using TF-IDF + Regex. |

### Running Scripts
Scripts can be executed in the background if needed:
```sh
python models/tfidf_extraction.py
python models/flan_t5_prompt_gen.py --num_prompts 100 --label HR --malicious_ratio 0.5
python models/bert_similarity.py --input data/outputs/generated_prompts.csv
```

---

## Usage

### Step 1: Install Environment
```sh
conda env create -f environment.yml
conda activate malicious_prompt_env
```

### Step 2: Generate the Word/Phrase-Label Bank
```sh
python models/tfidf_extraction.py
```
**Output:** `data/outputs/word_label_bank.csv`

### Step 3: Generate Synthetic Prompts
```sh
python models/flan_t5_prompt_gen.py --num_prompts 100 --label HR --malicious_ratio 0.5
```
**Output:** `data/outputs/generated_prompts.csv`

### Step 4: Validate Prompts with Legal-BERT
```sh
python models/bert_similarity.py --input data/outputs/generated_prompts.csv
```
**Output:** `data/outputs/final_dataset.csv`

---

---

## Final Repository Structure
```bash
malicious_prompt_classification
│── data/
│   ├── policy_documents/   # User-uploaded policy documents (INPUT ONLY)
│   ├── outputs/            # Stores all generated CSV files
│   │   ├── word_label_bank.csv  # Extracted words & phrases
│   │   ├── generated_prompts.csv  # GPT-generated prompts
│   │   ├── final_dataset.csv  # Final validated dataset
│
│── models/
│   ├── bert_similarity.py  # Legal-BERT similarity checker
│   ├── flan_t5_prompt_gen.py   # FLAN-T5-based prompt generation
│   ├── tfidf_extraction.py # TF-IDF + Regex-based term extraction
│
│── notebooks/
│   ├── 1_scrape_documents.ipynb  # Extract word-label bank
│   ├── 2_generate_prompts.ipynb  # Generate flagged & non-flagged prompts
│   ├── 3_validate_prompts.ipynb  # Compare and validate generated prompts
│
│── environment.yml   # Conda environment file
│── README.md

