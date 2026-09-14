from cav import config
import json
import numpy as np
from pathlib import Path
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from cav.concepts import MINI_DATASETS

import sys
sys.path.append(str(Path(__file__).parent.parent))
from cav.embed import get_bert_embeddings

def generate_full_table():
    names_path = Path('outputs/cav_llm/bottleneck_names.json')
    if not names_path.exists():
        print(f"Error: {names_path} not found")
        return
        
    with open(names_path, 'r', encoding='utf-8') as f:
        concepts = json.load(f)
        
    datasets = MINI_DATASETS
        
    results = {}
    
    for concept in concepts:
        print(f"Processing concept: {concept} across 12 layers...")
        if concept not in datasets:
            print(f"Warning: {concept} not found in datasets")
            continue
            
        pos_texts = datasets[concept]['positive']
        neg_texts = datasets[concept]['negative']
        texts = pos_texts + neg_texts
        labels = np.array([1] * len(pos_texts) + [0] * len(neg_texts))
        y = np.array([1]*len(pos_texts) + [0]*len(neg_texts))
        results[concept] = {}
        for layer in range(1, 13):
            # This generates or loads from cache
            pos_emb = get_bert_embeddings(pos_texts, layer=layer, cache_key=f"{concept}_pos")
            neg_emb = get_bert_embeddings(neg_texts, layer=layer, cache_key=f"{concept}_neg")
            X_l = np.vstack([pos_emb, neg_emb])
            
            scaler = StandardScaler()
            X_ls = scaler.fit_transform(X_l)
            
            svm_l = LinearSVC(C=config.SVM_C, max_iter=10000, dual=False)
            cv_scores = cross_val_score(svm_l, X_ls, y, cv=config.CV_FOLDS, scoring='accuracy')
            cv_mean = cv_scores.mean()
                    
            results[concept][layer] = cv_mean
            
    # Save the intermediate probe results so generate_probing_table.py can use them
    probe_results_path = Path('outputs/cav_llm/probe_results.json')
    probe_results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(probe_results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
        
    print(f"Saved probe results to {probe_results_path.absolute()}")
            
    # Generate LaTeX table
    tex_lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Maximum probe accuracy across all 12 ModernBERT layers for each LLM-generated concept.}",
        "\\label{tab:full_probing_accuracy}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{lcccccccccccc}",
        "\\toprule",
        "\\textbf{Concept} & \\multicolumn{12}{c}{\\textbf{ModernBERT Layer}} \\\\",
        "\\cmidrule(lr){2-13}",
        "& \\textbf{1} & \\textbf{2} & \\textbf{3} & \\textbf{4} & \\textbf{5} & \\textbf{6} & \\textbf{7} & \\textbf{8} & \\textbf{9} & \\textbf{10} & \\textbf{11} & \\textbf{12} \\\\",
        "\\midrule"
    ]
    
    for concept in concepts:
        if concept not in results:
            continue
        concept_name = concept.replace('_', ' ').title()
        row = [concept_name]
        for layer in range(1, 13):
            acc = results[concept][layer]
            row.append(f"{acc:.2f}")
        tex_lines.append(" & ".join(row) + " \\\\")
        
    tex_lines.extend([
        "\\bottomrule",
        "\\end{tabular}%",
        "}",
        "\\end{table}"
    ])
    
    out_path = Path('full_probing_table.tex')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(tex_lines) + '\n')
        
    print(f"Full probing table saved to {out_path.absolute()}")

if __name__ == '__main__':
    generate_full_table()
