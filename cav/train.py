import numpy as np
import pickle
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from cav import config
from cav.concepts import MINI_DATASETS
from cav.embed import get_bert_embeddings

def train_probe(concept_name: str) -> dict:
    dataset = MINI_DATASETS.get(concept_name)
    if not dataset:
        raise ValueError(f"Concept {concept_name} not found in MINI_DATASETS")
    
    pos_texts = dataset['positive']
    neg_texts = dataset['negative']
    
    print(f"\n--- Training probe for '{concept_name}' ---")
    
    y = np.array([1]*len(pos_texts) + [0]*len(neg_texts))
    
    # Layer sweep
    best_layer = config.PROBE_LAYER
    best_cv = 0.0
    layer_results = {}
    
    for layer in config.LAYER_SWEEP:
        pos_emb = get_bert_embeddings(pos_texts, layer=layer, cache_key=f"{concept_name}_pos")
        neg_emb = get_bert_embeddings(neg_texts, layer=layer, cache_key=f"{concept_name}_neg")
        X_l = np.vstack([pos_emb, neg_emb])
        
        scaler = StandardScaler()
        X_ls = scaler.fit_transform(X_l)
        
        svm_l = LinearSVC(C=config.SVM_C, max_iter=10000, dual=False)
        cv_scores = cross_val_score(svm_l, X_ls, y, cv=config.CV_FOLDS, scoring='accuracy')
        cv_mean = cv_scores.mean()
        layer_results[layer] = cv_mean
        print(f"  Layer {layer:2d}: CV = {cv_mean:.2f}")
        
        if cv_mean > best_cv:
            best_cv = cv_mean
            best_layer = layer
            
    print(f"  Selected Layer {best_layer} (CV = {best_cv:.2f})")
    
    # Train on best layer
    pos_emb = get_bert_embeddings(pos_texts, layer=best_layer, cache_key=f"{concept_name}_pos")
    neg_emb = get_bert_embeddings(neg_texts, layer=best_layer, cache_key=f"{concept_name}_neg")
    X = np.vstack([pos_emb, neg_emb])
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    svm = LinearSVC(C=config.SVM_C, max_iter=10000, dual=False)
    svm.fit(X_scaled, y)
    
    w = svm.coef_[0]
    cav_unit = w / np.linalg.norm(w)
    
    probe_dict = {
        "concept": concept_name,
        "svm": svm,
        "scaler": scaler,
        "layer": best_layer,
        "cav_unit": cav_unit,
        "cv_accuracy": best_cv,
        "n_positive": len(pos_texts),
        "n_negative": len(neg_texts),
        "mini_only": True
    }
    
    out_path = config.PROBE_OUT_DIR / f"{concept_name}.pkl"
    with out_path.open('wb') as f:
        pickle.dump(probe_dict, f)
        
    return probe_dict

def train_all_probes():
    probes = []
    for concept_name in MINI_DATASETS.keys():
        probe = train_probe(concept_name)
        probes.append(probe)
        
    print("\n--- Probe Training Summary ---")
    for p in probes:
        status = "OK" if p['cv_accuracy'] >= config.MIN_CV_ACCURACY else "FLAG"
        print(f"{p['concept']:25s} | Layer {p['layer']:2d} | CV: {p['cv_accuracy']:.2f} | {status}")
    return probes
