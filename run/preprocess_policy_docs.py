import os
import fitz  # PyMuPDF for extracting text from PDFs
import re

def process_and_clean_pdfs(input_folder, output_folder):
    """
    Recursively processes all PDFs in the input folder, extracts text, applies preprocessing,
    and saves cleaned text files in the output folder while printing previews.

    Args:
        input_folder (str): Folder containing PDF files.
        output_folder (str): Folder where cleaned text files will be saved.
    """
    os.makedirs(output_folder, exist_ok=True) 

    for root, _, files in os.walk(input_folder): 
        for file in files:
            if file.endswith(".pdf"):
                pdf_path = os.path.join(root, file)
                print(f"\nProcessing: {file}")
                with fitz.open(pdf_path) as doc:
                    raw_text = " ".join([page.get_text("text") for page in doc])
                initial_word_count = len(raw_text.split())
                patterns_to_remove = [
                    r'Page\s+\d+\s+of\s+\d+',  # Page numbers
                    r'Rev\.\s*\d+',  # Revision labels
                    r'Effective Date:.*',  # Effective Date metadata
                    r'Info Classification:.*',  # Info Classification metadata
                    r'Approved by.*',  # Approval metadata
                    r'Table of Contents',  # Table of Contents headers
                    r'Corporate Policy',  # General corporate headers
                    r'Scope.*',  # Remove scope sections
                    r'References.*',  # Remove References sections
                    r'Company may revise.*',  # Remove disclaimers
                    r'\b(confidential|sensitive|public|general business)\b',  # Classification labels
                    r'[\*\!\?\(\)\[\]\{\},\.;:_/\\\&\@\#\%\|<>]',  # Remove punctuation & special characters
                    r'[-]',  # Remove hyphens
                    r'[\d]',  # Remove numbers

                ]

                for pattern in patterns_to_remove:
                    raw_text = re.sub(pattern, '', raw_text, flags=re.IGNORECASE)

                raw_text = raw_text.replace("-", " ")
                cleaned_lines = [line.strip() for line in raw_text.split("\n") if len(line.strip()) > 10]
                cleaned_text = " ".join(cleaned_lines)
                cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip().lower()
                final_word_count = len(cleaned_text.split())
                print(f"Preview: {cleaned_text[:500]}...\n")  # Show first 500 characters
                print(f"Initial Word Count: {initial_word_count}, Final Word Count: {final_word_count}")
                relative_path = os.path.relpath(root, input_folder)
                save_folder = os.path.join(output_folder, relative_path)
                os.makedirs(save_folder, exist_ok=True)

                output_file = os.path.join(save_folder, f"{os.path.splitext(file)[0]}.txt")
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(cleaned_text)

                print(f"Saved: {output_file}")

input_folder = "/home/abradsha/Prompt-Classification/data/policy_documents"  
output_folder = "/home/abradsha/Prompt-Classification/data/cleaned_policy_documents"  
process_and_clean_pdfs(input_folder, output_folder)
