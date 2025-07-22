import os
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from transformers import AutoTokenizer, AutoModel
import torch
from scibert_embedder import SciBERTEmbedder

LABEL_CSV = "conference_labels.csv"
TEXT_DIR = "../cleaned_texts"
SAVE_PATH = "scibert_recommender.pkl"

def load_data():
    df = pd.read_csv(LABEL_CSV)
    texts = []
    labels = []
    for _, row in df.iterrows():
        path = os.path.join(TEXT_DIR, row["filename"])
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    texts.append(content)
                    labels.append(row["label"])
    return texts, labels

def main():
    texts, labels = load_data()
    embedder = SciBERTEmbedder()
    embeddings = embedder.get_embeddings(texts)
    
    scaler = StandardScaler()
    X = scaler.fit_transform(embeddings)
    y = labels

    clf = LogisticRegression(max_iter=1000)
    scores = cross_val_score(clf, X, y, cv=StratifiedKFold(n_splits=2, shuffle=True, random_state=42))
    print(f"Cross-validated accuracy: {scores.mean():.4f}")

    clf.fit(X, y)

    joblib.dump({
        "classifier": clf,
        "scaler": scaler,
        "model_name": embedder.model_name,
        "max_length": embedder.max_length,
        "labels": sorted(set(labels))
    }, SAVE_PATH)

    print(f"Model saved to {SAVE_PATH}")

if __name__ == "__main__":
    main()