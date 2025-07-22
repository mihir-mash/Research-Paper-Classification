# backend/conference_recommender.py
import torch
import joblib
import os
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm

REFERENCE_TEXT_DIR = "cleaned_texts"
EMBED_CACHE_PATH = "reference_embeddings.npz"

def embed_text(text, model_name="allenai/scibert_scivocab_uncased", max_length=512):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :]
    return embedding[0].numpy().reshape(1, -1)

def load_reference_embeddings():
    if os.path.exists(EMBED_CACHE_PATH):
        print(f"Loading cached embeddings from {EMBED_CACHE_PATH}")
        data = np.load(EMBED_CACHE_PATH, allow_pickle=True)
        embeddings = data["embeddings"]
        names = data["names"]
        return embeddings, names.tolist()
    else:
        embeddings = []
        names = []
        for filename in tqdm(os.listdir(REFERENCE_TEXT_DIR)):
            if filename.endswith(".txt"):
                path = os.path.join(REFERENCE_TEXT_DIR, filename)
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                    emb = embed_text(content)
                    embeddings.append(emb)
                    names.append(filename.replace(".txt", ""))
        embeddings = np.vstack(embeddings)
        np.savez_compressed(EMBED_CACHE_PATH, embeddings=embeddings, names=np.array(names))
        print(f"Saved embeddings to {EMBED_CACHE_PATH}")
        return embeddings, names

REFERENCE_EMBS, PAPER_NAMES = load_reference_embeddings()

def recommend_conference(text):
    query_emb = embed_text(text)
    sims = cosine_similarity(query_emb, REFERENCE_EMBS)[0]
    top_indices = sims.argsort()[-3:][::-1]
    top_papers = [PAPER_NAMES[i] for i in top_indices]

    conference_counts = {}
    for name in top_papers:
        conf = name.split("_")[0]
        conference_counts[conf] = conference_counts.get(conf, 0) + 1

    best_conf = max(conference_counts.items(), key=lambda x: x[1])[0]
    rationale = f"The paper aligns with {best_conf} due to methodological and topical similarity with papers such as {', '.join(top_papers)}."

    return best_conf, rationale
