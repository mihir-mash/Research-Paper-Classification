import os
import joblib
import numpy as np
from model2.scibert_embedder import SciBERTEmbedder
from .api_rationale import generate_rationale_api  

def recommend_conference_with_probs(text):
    """
    Predict conference, return confidence scores, and generate rationale via API.
    """
    # 1. Load trained model
    model_path = os.path.join("model2", "scibert_recommender.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Conference model not found at {model_path}")
    model = joblib.load(model_path)

    # 2. Embed paper
    embedder = SciBERTEmbedder()
    emb = embedder.get_embeddings([text])
    X = model["scaler"].transform(emb)

    # 3. Predict + probabilities
    pred_label = model["classifier"].predict(X)[0]
    probs = model["classifier"].predict_proba(X)[0]
    labels = model["classifier"].classes_
    confidence_scores = {labels[i]: float(probs[i] * 100) for i in range(len(labels))}  # as %

    # 4. Generate rationale using API
    rationale = generate_rationale_api(text, pred_label, confidence_scores)

    return pred_label, confidence_scores, rationale
