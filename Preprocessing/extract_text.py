import fitz
import os

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

input_dirs = {
    "reference": r"D:\Mihir\DEGREE\Activities\HECK\kharagpur omyown\drive dataset\Reference",
    "papers": r"D:\Mihir\DEGREE\Activities\HECK\kharagpur omyown\drive dataset\Papers"
}

output_folder = r"D:\Mihir\DEGREE\Activities\HECK\kharagpur omyown\parsed_texts"
os.makedirs(output_folder, exist_ok=True)

for root_label, base_dir in input_dirs.items():
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".pdf"):
                pdf_path = os.path.join(root, file)
                paper_id = file.replace(".pdf", "")
                text = extract_text_from_pdf(pdf_path)
                with open(os.path.join(output_folder, f"{paper_id}.txt"), 'w', encoding='utf-8') as f:
                    f.write(text)
