import os
import joblib
import numpy as np
from models.scibert_embedder import SciBERTEmbedder as Task1Embedder
from .scibert_embedder import SciBERTEmbedder as Task2Embedder
from backend.api_rationale import generate_rationale_api  # Groq API

def cosine_similarity(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return np.dot(a, b)

# TASK 1: Publishability
def predict_publishability(text):
    model = joblib.load("models/scibert_small_classifier.pkl")
    embedder = Task1Embedder()
    emb = embedder.get_embeddings([text])
    X = model["scaler"].transform(emb)
    pred = model["classifier"].predict(X)[0]
    return pred  # "Publishable" or "Non-Publishable"

# TASK 2: Conference + Local Similarity Rationale
def predict_conference(text):
    model = joblib.load("model2/scibert_recommender.pkl")
    embedder = Task2Embedder()
    emb = embedder.get_embeddings([text])
    X = model["scaler"].transform(emb)

    # Probabilities for each conference
    probs = model["classifier"].predict_proba(X)[0]
    labels = model["classifier"].classes_
    confidence_scores = {label: round(float(prob * 100), 2) for label, prob in zip(labels, probs)}

    # Predicted conference
    best_idx = np.argmax(probs)
    pred_conf = labels[best_idx]

    return pred_conf, confidence_scores

def full_pipeline(text: str):
    """
    Full prediction pipeline:
    1. Predict publishability (local)
    2. Predict conference + confidence scores (local)
    3. Generate rationale (via Groq API)
    """
    try:
        # Local predictions
        publishable_pred = predict_publishability(text)
        conference_pred, confidence_scores = predict_conference(text)

        # Try to get rationale from Groq API, but don't break if it fails
        try:
            rationale = generate_rationale_api(text, conference_pred, confidence_scores)
        except Exception as api_err:
            rationale = f"Rationale unavailable: {api_err}"

        return {
            "publishable": publishable_pred == "Publishable",
            "conference": conference_pred,
            "percentages": confidence_scores,
            "rationale": rationale
        }

    except Exception as e:
        return {
            "error": str(e),
            "publishable": False,
            "rationale": "An error occurred. Please try again.",
            "conference": "",
            "percentages": {}
        }
