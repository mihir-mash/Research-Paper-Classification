from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from tqdm import tqdm

class SciBERTEmbedder:
    def __init__(self, model_name="allenai/scibert_scivocab_uncased", max_length=512):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def get_embeddings(self, texts, batch_size=8):
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size)):
            batch = texts[i:i + batch_size]
            tokens = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            tokens = {k: v.to(self.device) for k, v in tokens.items()}
            with torch.no_grad():
                outputs = self.model(**tokens)
                cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(cls_embeddings)
        return np.vstack(embeddings)
