import re
import os

def clean_text(text):
    text = re.sub(r'\n+', '\n', text)  # Collapse multiple newlines
    text = re.sub(r'\s+', ' ', text)   # Collapse whitespace
    text = re.sub(r'References.*', '', text, flags=re.I)  # Remove everything after "References"
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII chars
    return text.strip()

def clean_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        if filename.endswith('.txt'):
            input_path = os.path.join(input_folder, filename)
            with open(input_path, 'r', encoding='utf-8') as f:
                text = f.read()
            cleaned = clean_text(text)
            output_path = os.path.join(output_folder, filename)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(cleaned)

if __name__ == "__main__":
    input_folder = r"D:\Mihir\DEGREE\Activities\HECK\kharagpur omyown\parsed_texts"
    output_folder = r"D:\Mihir\DEGREE\Activities\HECK\kharagpur omyown\cleaned_texts"
    clean_folder(input_folder, output_folder)
