import joblib
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import os

_model_data = None

def get_model(path=None):
    global _model_data
    if _model_data is None:
        if path is None:
            path = os.path.join("models", "scibert_small_classifier.pkl")
        _model_data = joblib.load(path)
    return _model_data

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
    prob = clf.predict_proba(scaled)[0]
    return "publishable" if pred == 1 else "non-publishable", float(max(prob))