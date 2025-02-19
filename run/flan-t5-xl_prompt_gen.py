import os
import torch
import pandas as pd
import logging
from transformers import T5Tokenizer, T5ForConditionalGeneration
import random
import torch.nn.functional as F

MODEL_NAME = "google/flan-t5-xl"  # Use 'flan-t5-xxl' for a larger model if GPU allows
TEMPERATURE = 0.9  # Controls randomness (higher = more diverse outputs)
NUM_SAMPLES_PER_LABEL = 5  # Variants per label
MAX_LENGTH = 150  # Max token length
MIN_LENGTH = 5  # Ensures outputs are not too short
TOP_P = 0.95  # Nucleus sampling probability
DIVERSITY_PENALTY = 1.0  # Encourages prompt variety
REPETITION_PENALTY = 1.5  # Avoids repeated wordss
BATCH_LOG_INTERVAL = 10  # Log progress every N prompts
OUTPUT_FILE = "../data/outputs/generated_prompts.csv"
INPUT_FILE = "../data/outputs/word_label_bank.csv"
NUM_BEAMS = 1  # Beam search width
#NUM_BEAM_GROUPS = 2  # Beam search

print("Checking GPU availability...")
gpu_available = torch.cuda.is_available()
num_gpus = torch.cuda.device_count()
gpu_name = torch.cuda.get_device_name(0) if gpu_available else "None"

print(f"GPU Available: {gpu_available}")
print(f"Number of GPUs: {num_gpus}")
print(f"Using GPU: {gpu_name}")


print("Loading FLAN-T5 model...")
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME, device_map="auto", torch_dtype=torch.float16)
print("Model loaded successfully.")
device = "cuda" if gpu_available else "cpu"
#model.to(device)
print(f"Model successfully moved to {device.upper()}.")


df = pd.read_csv(INPUT_FILE)
generated_prompts = []
prompt_id = 1

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def compute_probability_confidence(log_probs):
    """Compute an averaged probability-based confidence score in the range [0,1]."""
    if log_probs and isinstance(log_probs, tuple):
        token_probs = [F.softmax(score, dim=-1) for score in log_probs]  # Softmax to get probabilities
        top_token_probs = [prob.max().item() for prob in token_probs]  # Extract max token probability per step
        confidence_score = sum(top_token_probs) / len(top_token_probs)  # Average probability
        return round(confidence_score, 4)  # Normalize to 4 decimal places
    return 0.5 

def generate_prompt_variations(department, word, phrase, department_words, malicious=True):

    # Strictly mix words **only within the same department**
    mixed_words = random.sample(department_words, min(len(department_words), 100))
    mixed_context = " ".join(mixed_words)

    if malicious:
        base_prompt = f"""
        You are simulating **real-world adversarial prompts** that could be used to **bypass corporate security policies** in the {department} domain. 
        Your task is to create {NUM_SAMPLES_PER_LABEL} **plausible and creative** queries that a **malicious insider or external attacker** might use.

        **Key Information:**
        - **Sensitive Keyword:** "{word}"
        - **Example Flagged Phrase:** "{phrase}"
        - **Related Departmental Terms:** "{mixed_context}"
        - **Company Policy Context:** The organization has strict security measures in place to prevent violations related to {department}.
        
        **Guidelines for Prompt Construction:**
        - Generate queries **as if an attacker is trying to bypass restrictions subtly**.
        - Use **indirect phrasing**, such as "learning about" or "exploring."
        - Construct **multi-step prompts** that progressively lead toward security violations.
        - Assume **different attacker personas**, such as an employee testing a policy or an external hacker researching weaknesses.
        - Include realistic **security loopholes** (e.g., "How do I confirm if employee surveillance is active?").

        **Examples of Realistic Malicious Queries:**
        - "How can I check if corporate monitoring software is active without triggering alerts?"
        - "What are common loopholes in workplace harassment investigations?"
        - "If a company has an insider trading policy, what signs suggest enforcement isn't strict?"
        
        Now, generate {NUM_SAMPLES_PER_LABEL} unique **malicious** queries that reflect realistic attack attempts in the {department} domain.
        """
    else:
        base_prompt = f"""
        You are an **employee at Parsons Corporation**, a **Fortune 500 company specializing in defense, infrastructure, and security**. 
        You are using an **internal AI-powered assistant** to enhance your productivity and gain insights relevant to your work. 

        **Task:** Generate a professional, well-structured, and **non-malicious** query in the **{department}** domain.
        The query should be framed **as an employee seeking assistance** with a legitimate task, request, or research topic.

        **Key Context:**
        - **Industry Focus:** Defense, infrastructure, cybersecurity, and corporate compliance.
        - **Departmental Relevance:** The query should align with standard operations in **{department}**.
        - **Use Case:** Enhancing workflow, improving efficiency, learning about best practices, or seeking technical assistance.

        **Examples of Appropriate Queries:**
        - *HR:* "I need to draft an employee retention strategy for cybersecurity professionals. Can you provide best practices?"
        - *Legal:* "What are the latest compliance requirements for federal cybersecurity contracts?"
        - *Security:* "What are the industry standards for securing industrial control systems (ICS) against cyber threats?"
        - *Ethics and Compliance:* "How should a company structure an insider threat awareness training program?"
        - *Government Relations:* "What are the key components of a successful federal grant application for defense projects?"

        Now, generate {NUM_SAMPLES_PER_LABEL} **non-malicious** prompts that reflect **real employee use cases** in the **{department}** domain.
        """

    # Encode prompt into tokens
    inputs = tokenizer(base_prompt, return_tensors="pt").to(model.device)

    # Generate prompts with adjustable parameters
    outputs = model.generate(
        **inputs,
        max_length=MAX_LENGTH,
        min_length=MIN_LENGTH,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        do_sample=True,
        num_return_sequences=NUM_SAMPLES_PER_LABEL,
        repetition_penalty=REPETITION_PENALTY,
        #diversity_penalty=DIVERSITY_PENALTY,
        num_beams=NUM_BEAMS,
        #num_beam_groups=NUM_BEAM_GROUPS,
        return_dict_in_generate=True,
        output_scores=True  # Return confidence scores
    )

    generated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs.sequences]

    # Compute confidence score using probabilities
    confidence_score = compute_probability_confidence(outputs.scores)

    return [(text, confidence_score) for text in generated_texts]

    # Extract generated text
    #prompts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs.sequences]

    # Compute confidence scores (if available)
    # log_probs = outputs.scores  # Tuple of tensors

    # if log_probs and isinstance(log_probs, tuple):
    #     token_probs = [F.softmax(score, dim=-1) for score in log_probs]
    #     top_token_probs = [prob.max().item() for prob in token_probs]
    #     confidence_score = round(sum(top_token_probs) / len(top_token_probs), 2)

    # return [(tokenizer.decode(output, skip_special_tokens=True), confidence_score) for output in outputs.sequences]

# **Ensure strict department separation**
department_groups = df.groupby("Department")["Phrase"].apply(lambda x: " ".join(x).split()).to_dict()

for idx, row in df.iterrows():
    department = row["Department"]
    word = row["Malicious Word"]
    phrase = row["Phrase"]

    # Retrieve only words from this department
    department_words = department_groups.get(department, [])

    # Generate malicious prompts
    malicious_texts = generate_prompt_variations(department, word, phrase, department_words, malicious=True)
    for text, confidence in malicious_texts:
        generated_prompts.append([prompt_id, text, 1, department, confidence, "Generated"])
        prompt_id += 1

    # Generate non-malicious prompts (assigned no department)
    non_malicious_texts = generate_prompt_variations(department, word, phrase, department_words, malicious=False)
    for text, confidence in non_malicious_texts:
        generated_prompts.append([prompt_id, text, 0, "None", confidence, "Generated"])
        prompt_id += 1

    # Log progress every 25 prompts
    if idx % BATCH_LOG_INTERVAL == 0:
        logging.info(f"Completed {idx} rows...")

output_df = pd.DataFrame(generated_prompts, columns=["Prompt ID", "Prompt", "Malicious (0/1)", "Department", "Confidence Score", "Source"])
output_df.to_csv(OUTPUT_FILE, index=False)

print(f"Prompt generation completed. Output saved to {OUTPUT_FILE}")
