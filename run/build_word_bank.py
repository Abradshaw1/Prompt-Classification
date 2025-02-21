import os
import logging
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
from tabulate import tabulate 

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

DOC_FOLDER = "/home/abradsha/Prompt-Classification/data/cleaned_policy_documents/"
OUTPUT_FILE = "/home/abradsha/Prompt-Classification/data/outputs/word_label_bank.csv"
DEPARTMENT_LABELS = ["HR", "Legal", "Security", "Ethics and Compliance", "Government Relations"]
logger.info("Loading models...")
LEGAL_BERT_MODEL = "nlpaueb/legal-bert-base-uncased"
SENTENCE_BERT_MODEL = "all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(LEGAL_BERT_MODEL)
ner_model = AutoModelForTokenClassification.from_pretrained(LEGAL_BERT_MODEL).to(device)
ner_pipeline = pipeline("ner", model=ner_model, tokenizer=tokenizer, aggregation_strategy="simple", device=0 if gpu_available else -1)
sbert_model = SentenceTransformer(SENTENCE_BERT_MODEL, device=device)
nlp = spacy.load("en_core_web_trf")
logger.info("Models loaded successfully.")


def read_cleaned_text_files():
    logger.info("Reading cleaned text files...")
    documents = {}
    files = [f for f in os.listdir(DOC_FOLDER) if f.endswith(".txt")]

    for i, filename in enumerate(files, start=1):
        filepath = os.path.join(DOC_FOLDER, filename)
        with open(filepath, "r", encoding="utf-8") as file:
            documents[filename] = file.read()
        logger.info(f"Batch {i} completed: {filename}")

    logger.info(f"Total documents read: {len(documents)}")
    return documents


def extract_named_entities(text):
    """Extracts full multi-word named entities using Legal-BERT."""
    phrases = []
    ner_results = ner_pipeline(text)

    for entity in ner_results:
        word = entity["word"].replace("##", "").strip()
        if word:
            phrases.append(word)

    full_entities = []
    current_phrase = []

    for i, word in enumerate(phrases):
        if i > 0 and ner_results[i - 1]["end"] == ner_results[i]["start"]:
            current_phrase.append(word)
        else:
            if current_phrase:
                full_entities.append(" ".join(current_phrase))
            current_phrase = [word]

    if current_phrase:
        full_entities.append(" ".join(current_phrase))

    return list(set(full_entities))


def extract_most_informative_word(phrase):
    doc = nlp(phrase)
    
    # Filter out stopwords, punctuation, and numbers
    words = [token.text.lower() for token in doc if token.is_alpha and not token.is_stop]
    
    if not words:
        return None  # Skip if phrase contains no meaningful words

    # Use TF-IDF to rank words
    vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
    try:
        tfidf_matrix = vectorizer.fit_transform([" ".join(words)])
        feature_names = vectorizer.get_feature_names_out()
        avg_scores = tfidf_matrix.mean(axis=0).A1
        ranked_words = sorted(zip(feature_names, avg_scores), key=lambda x: x[1], reverse=True)

        if ranked_words:
            return ranked_words[0][0]  # Return highest TF-IDF scoring word
    except ValueError:
        pass 

    # Fallback: return the first noun or verb in the phrase
    for token in doc:
        if token.pos_ in ["NOUN", "VERB", "ADJ"]:
            return token.text.lower()

    return words[0]  # Last fallback: return first valid word


def assign_labels(words):
    labeled_entries = {}
    label_embeddings = {label: sbert_model.encode(label) for label in DEPARTMENT_LABELS}

    for word in words:
        word_embedding = torch.tensor(sbert_model.encode(word)).unsqueeze(0)
        best_label = max(DEPARTMENT_LABELS, key=lambda label: torch.nn.functional.cosine_similarity(
            word_embedding, torch.tensor(label_embeddings[label]).unsqueeze(0), dim=-1).item())
        labeled_entries[word] = best_label

    return labeled_entries

def main():
    logger.info("Starting main pipeline...")
    extracted_data = []
    documents = read_cleaned_text_files()
    phrase_dict = {}

    for filename, content in documents.items():
        logger.info(f"Processing document: {filename}")
        legal_phrases = extract_named_entities(content)

        for phrase in legal_phrases:
            key_word = extract_most_informative_word(phrase)
            if key_word:
                phrase_dict[key_word] = phrase  # Store in dictionary

        logger.info("\n=== Sample of TF-IDF Extracted Key-Value Pairs ===")
        sample_pairs = list(phrase_dict.items())[:10]  # Show the first 10 items
        sample_table = [[key, sample_pairs[key]] for key in range(min(10, len(sample_pairs)))]
        logger.info("\n" + tabulate(sample_table, headers=["Malicious Word", "Full Phrase"], tablefmt="grid"))


        # logger.info("\n=== TF-IDF Extracted Key-Value Pairs ===")
        # table = [[key, phrase_dict[key]] for key in phrase_dict]
        # #logger.info("\n" + tabulate(table, headers=["Malicious Word", "Full Phrase"], tablefmt="grid"))

        word_labels = assign_labels(phrase_dict.keys())

        logger.info("\n=== SBERT Assigned Labels ===")
        labeled_table = [[word_labels[key], key, phrase_dict[key]] for key in phrase_dict]
        # Log label assignments, summarizing counts instead of showing every pair
        label_counts = pd.Series([word_labels[key] for key in phrase_dict]).value_counts()
        logger.info("\n" + tabulate(label_counts.reset_index(), headers=["Department", "Count"], tablefmt="grid"))


        # logger.info("\n=== SBERT Assigned Labels ===")
        # labeled_table = [[word_labels[key], key, phrase_dict[key]] for key in phrase_dict]
        #logger.info("\n" + tabulate(labeled_table, headers=["Department", "Malicious Word", "Phrase"], tablefmt="grid"))

        for key_word, phrase in phrase_dict.items():
            extracted_data.append([word_labels[key_word], key_word, phrase, 1])
            
    df = pd.DataFrame(extracted_data, columns=["Department", "Malicious Word", "Phrase", "Flag"])
    label_counts = df["Department"].value_counts()
    label_summary = tabulate(label_counts.reset_index(), headers=["Department", "Count"], tablefmt="grid")
    
    logger.info("\n=== Final Label Counts Before Saving ===")
    logger.info("\n" + label_summary)
    df.to_csv(OUTPUT_FILE, index=False)
    logger.info(f"Final extracted data saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
