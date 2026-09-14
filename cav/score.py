import numpy as np
import json
from typing import List, Dict
from cav import config
from cav.embed import get_bert_embeddings

def score_transcript(transcript: str, probe_dict: dict, mode: str = 'raw') -> float:
    emb = get_bert_embeddings([transcript], layer=probe_dict['layer'])
    scaled = probe_dict['scaler'].transform(emb)
    raw = float(probe_dict['svm'].decision_function(scaled)[0])
    
    if mode == 'raw':
        return raw
    return float(1 / (1 + np.exp(-raw)))

def build_bottleneck_matrix(transcripts: List[str], probe_dicts: List[dict]) -> np.ndarray:
    N = len(transcripts)
    K = len(probe_dicts)
    matrix = np.zeros((N, K))
    
    # Group probes by layer to optimize embedding
    layer_to_probes = {}
    for i, p in enumerate(probe_dicts):
        layer = p['layer']
        if layer not in layer_to_probes:
            layer_to_probes[layer] = []
        layer_to_probes[layer].append((i, p))
        
    for layer, probes in layer_to_probes.items():
        print(f"Embedding {N} transcripts at layer {layer}...")
        # Since 'transcripts' changes between datasets (e.g. train/test), we don't cache globally with a static key here
        # unless we hash the texts. For now, since caching happens in train for positive/negative, we can omit cache_key 
        # or use a generic one if we are just scoring the whole dataset once.
        # It's safer to not cache this full dataset to avoid overwriting train vs test, 
        # or we could use hash of first transcript as part of key.
        cache_key = f"transcripts_{hash(transcripts[0])}" if transcripts else None
        emb = get_bert_embeddings(transcripts, layer=layer, cache_key=cache_key)
        
        for i, p in probes:
            scaled = p['scaler'].transform(emb)
            scores = p['svm'].decision_function(scaled)
            matrix[:, i] = scores
            
    # Save the matrix and names
    np.save(config.BOTTLENECK_OUT, matrix)
    names = [p['concept'] for p in probe_dicts]
    with config.BOTTLENECK_NAMES.open('w') as f:
        json.dump(names, f)
        
    print(f"Bottleneck matrix of shape {matrix.shape} saved to {config.BOTTLENECK_OUT}")
    return matrix
