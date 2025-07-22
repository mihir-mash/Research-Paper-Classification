import os
import csv

reference_dir = "D:\Mihir\DEGREE\Activities\HECK\kharagpur omyown\drive dataset\Reference"
cleaned_texts_dir = "D:\Mihir\DEGREE\Activities\HECK\kharagpur omyown\cleaned_texts"
output_csv = "reference_labels.csv"

rows = []

# Handle Publishable (recursively)
publishable_root = os.path.join(reference_dir, "Publishable")
for root, _, files in os.walk(publishable_root):
    for filename in files:
        if filename.endswith(".pdf"):
            txt_filename = filename.replace(".pdf", ".txt")
            txt_path = os.path.join(cleaned_texts_dir, txt_filename)
            if os.path.exists(txt_path):
                rows.append({"filename": txt_filename, "label": "Publishable"})
            else:
                print(f"Warning: {txt_filename} not found in cleaned_texts.")

# Handle Non-Publishable (flat)
non_publishable_dir = os.path.join(reference_dir, "Non-Publishable")
if os.path.exists(non_publishable_dir):
    for filename in os.listdir(non_publishable_dir):
        if filename.endswith(".pdf"):
            txt_filename = filename.replace(".pdf", ".txt")
            txt_path = os.path.join(cleaned_texts_dir, txt_filename)
            if os.path.exists(txt_path):
                rows.append({"filename": txt_filename, "label": "Non-Publishable"})
            else:
                print(f"Warning: {txt_filename} not found in cleaned_texts.")

with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["filename", "label"])
    writer.writeheader()
    writer.writerows(rows)

print(f"reference_labels.csv created with {len(rows)} entries.")