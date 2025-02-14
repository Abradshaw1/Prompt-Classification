import os
import pandas as pd
import re
import spacy
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
import torch
from pdfminer.high_level import extract_text
from sentence_transformers import SentenceTransformer

nlp = spacy.load("en_core_web_trf")

bert_model = "nlpaueb/legal-bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(bert_model)
ner_model = AutoModelForTokenClassification.from_pretrained(bert_model)
ner_pipeline = pipeline("ner", model=ner_model, tokenizer=tokenizer, aggregation_strategy="simple")

sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

COMPLIANCE_KEYWORDS = ["must", "shall", "prohibited", "forbidden", "not allowed", "required", "should", "violates", "illegal"]
LEGAL_ACTION_PHRASES = ["must comply", "shall ensure", "is prohibited from", "is required to", "not allowed to", "forbidden to"]

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[\n\r\t]", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_text_from_file(filepath):
    if filepath.endswith(".pdf"):
        text = extract_text(filepath)
    elif filepath.endswith(".txt"):
        with open(filepath, "r", encoding="utf-8") as file:
            text = file.read()
    else:
        return ""
    return preprocess_text(text)

# Extract Named Entities (Words) using Legal-BERT
def extract_words(content):
    words = set()
    ner_results = ner_pipeline(content)

    for entity in ner_results:
        words.add(entity["word"])  # Keep full entity words

    return list(words)

# Extract Compliance Clauses (Phrases) using Rule-Based Matching
def extract_phrases(content):
    phrases = []
    doc = nlp(content)

    for sent in doc.sents:
        sent_text = sent.text.strip()

        # Identify compliance-related clauses
        for token in sent:
            if token.text in COMPLIANCE_KEYWORDS and token.dep_ in ["aux", "ROOT", "acl"]:
                phrase = " ".join([child.text for child in token.subtree if child.text not in COMPLIANCE_KEYWORDS])

                # Validate phrase using SBERT similarity
                phrase_embedding = sbert_model.encode(phrase)
                max_similarity = max([
                    torch.nn.functional.cosine_similarity(
                        torch.tensor(phrase_embedding).unsqueeze(0),
                        torch.tensor(sbert_model.encode(legal_phrase)).unsqueeze(0)
                    ).item()
                    for legal_phrase in LEGAL_ACTION_PHRASES
                ])

                if max_similarity > 0.75:
                    phrases.append(phrase)

    return phrases

# Assign Department Labels using SBERT Similarity
def assign_labels(words, phrases, department_labels):
    updated_entries = []
    label_embeddings = {label: sbert_model.encode(label) for label in department_labels}

    for word, phrase in tqdm(zip(words, phrases), total=min(len(words), len(phrases)), desc="Assigning Labels"):
        text_to_classify = word if word else phrase
        term_embedding = torch.tensor(sbert_model.encode(text_to_classify)).unsqueeze(0)
        best_label = None
        best_score = -1

        for label, label_emb in label_embeddings.items():
            similarity = torch.nn.functional.cosine_similarity(term_embedding, torch.tensor(label_emb).unsqueeze(0), dim=-1)

            if similarity.item() > best_score:
                best_score = similarity.item()
                best_label = label

        updated_entries.append([best_label, word, phrase, 1])  # Store in correct format

    return updated_entries

# Run Extraction & Save Results
def main():
    doc_folder = "data/policy_documents/"
    output_file = "data/outputs/word_label_bank.csv"
    department_labels = ["HR", "Legal", "Security", "IT", "Ethics and Compliance", "Government Relations"]
    extracted_data = []

    files = [f for f in os.listdir(doc_folder) if f.endswith(".txt") or f.endswith(".pdf")]

    for filename in tqdm(files, desc="Processing Documents"):
        filepath = os.path.join(doc_folder, filename)
        content = extract_text_from_file(filepath)
        words = extract_words(content)  # Get Named Entities
        phrases = extract_phrases(content)  # Get Compliance Clauses
        num_pairs = max(len(words), len(phrases))
        words = words[:num_pairs] + [""] * (num_pairs - len(words))  # Pad words if needed
        phrases = phrases[:num_pairs] + [""] * (num_pairs - len(phrases))  # Pad phrases if needed

        labeled_data = assign_labels(words, phrases, department_labels)
        extracted_data.extend(labeled_data)
    df = pd.DataFrame(extracted_data, columns=["Department", "Word", "Phrase", "Flag"])
    df.to_csv(output_file, index=False)
    print(f"Extracted terms saved to {output_file}")

if __name__ == "__main__":
    main()
