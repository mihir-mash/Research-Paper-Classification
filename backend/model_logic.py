import joblib
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import os

_model_data = None

def get_model(path=None):
    base = os.path.dirname(__file__)           # backend/
    project_root = os.path.normpath(os.path.join(base, ".."))
    if path is None:
        candidates = [
            os.path.join(project_root, "models", "scibert_small_classifier.pkl")
        ]
        for c in candidates:
            if os.path.exists(c):
                path = c
                break
        if path is None:
            raise FileNotFoundError("Could not find classifier pickle. Searched: " + ", ".join(candidates))
    print("Loading model from", path)   
    return joblib.load(path)

def embed_text(text, model_name="allenai/scibert_scivocab_uncased", max_length=512):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :]
    return embedding[0].numpy().reshape(1, -1)

def predict_publishability(text):
    model_data = get_model()
    clf = model_data['classifier']
    scaler = model_data['scaler']
    emb = embed_text(text, model_data['model_name'], model_data['max_length'])
    scaled = scaler.transform(emb)
    pred = clf.predict(scaled)[0]
    print("DEBUG: Model raw prediction:", pred)  # <-- Add this line
    prob = clf.predict_proba(scaled)[0]
    return "publishable" if pred.lower().startswith("publish") else "non-publishable", float(max(prob))