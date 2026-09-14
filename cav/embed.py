import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from typing import List, Optional
import tqdm

from cav import config

class BertEmbedder:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(config.BERT_MODEL_NAME)
        self.model = AutoModelForMaskedLM.from_pretrained(config.BERT_MODEL_NAME, output_hidden_states=True)
        self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)

    def get_embeddings(self, texts: List[str], layer: int, cache_key: Optional[str] = None) -> np.ndarray:
        if cache_key:
            cache_path = config.EMBED_CACHE_DIR / f"{cache_key}_layer{layer}.npy"
            if cache_path.exists():
                return np.load(cache_path)

        embeddings = []
        for i in tqdm.tqdm(range(0, len(texts), config.BATCH_SIZE), desc=f"Embedding layer {layer}", leave=False):
            batch = texts[i : i + config.BATCH_SIZE]
            enc = self.tokenizer(
                batch, return_tensors='pt',
                truncation=True, max_length=config.MAX_SEQ_LEN, padding=True
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}

            with torch.no_grad():
                out = self.model(**enc)

            h = out.hidden_states[layer]            # (B, seq_len, 768)

            mask = enc['attention_mask'].unsqueeze(-1).float()   # (B, L, 1)
            pooled = (h * mask).sum(1) / mask.sum(1)             # (B, 768)
            embeddings.append(pooled.cpu().numpy())

        if embeddings:
            embeddings_arr = np.vstack(embeddings)
        else:
            embeddings_arr = np.empty((0, 768))
        
        if cache_key:
            np.save(cache_path, embeddings_arr)
            
        return embeddings_arr

_embedder = None
def get_bert_embeddings(texts: List[str], layer: int, cache_key: Optional[str] = None) -> np.ndarray:
    global _embedder
    if _embedder is None:
        _embedder = BertEmbedder()
    return _embedder.get_embeddings(texts, layer, cache_key)
