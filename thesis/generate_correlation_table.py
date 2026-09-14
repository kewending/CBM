import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr, mannwhitneyu
from cav.concepts import WEAK_LABEL_FUNCTIONS

def load_data():
    bottleneck_path = Path('outputs/cav_llm/bottleneck_matrix.npy')
    names_path = Path('outputs/cav_llm/bottleneck_names.json')
    train_path = Path('outputs/par_only/train.csv')
    test_path = Path('outputs/par_only/test.csv')
    
    if not bottleneck_path.exists():
        raise FileNotFoundError(f"{bottleneck_path} not found")
        
    bottleneck_matrix = np.load(bottleneck_path)
    
    with open(names_path, 'r', encoding='utf-8') as f:
        concept_names = json.load(f)
        
    train_df = pd.read_csv(train_path)
    train_df = train_df[train_df['par_transcript'].notna()].reset_index(drop=True)
    test_df = pd.read_csv(test_path)
    test_df = test_df[test_df['par_transcript'].notna()].reset_index(drop=True)
    
    df = pd.concat([train_df, test_df], ignore_index=True)
    labels = df['label'].values
    transcripts = df['par_transcript'].values
    
    return bottleneck_matrix, concept_names, labels, transcripts

def generate_table():
    out_path = Path('correlation_table.tex')
    
    bottleneck_matrix, concept_names, labels, transcripts = load_data()
    
    tex_lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Spearman correlation ($\\rho$) between LLM CAV scores and programmatic weak labels, and Dementia Discrimination Mann-Whitney U ($p$-value).}",
        "\\label{tab:correlation_and_discrimination}",
        "\\begin{tabular}{lcc}",
        "\\toprule",
        "\\textbf{Concept} & \\textbf{Spearman $\\rho$} & \\textbf{Mann-Whitney $p$} \\\\",
        "\\midrule"
    ]
    
    for i, concept in enumerate(concept_names):
        scores = bottleneck_matrix[:, i]
        
        # Spearman Rho
        weak_fn = WEAK_LABEL_FUNCTIONS.get(concept)
        rho_str = "N/A"
        if weak_fn:
            weak_scores = [weak_fn(t) for t in transcripts]
            if np.std(weak_scores) > 0 and np.std(scores) > 0:
                rho, pval = spearmanr(scores, weak_scores)
                rho_str = f"{rho:.2f}"
                
        # Mann-Whitney U
        scores_dem = [s for s, l in zip(scores, labels) if l == 1]
        scores_ctrl = [s for s, l in zip(scores, labels) if l == 0]
        
        mwu_str = "N/A"
        if scores_dem and scores_ctrl:
            stat, pval = mannwhitneyu(scores_dem, scores_ctrl)
            mwu_str = f"{pval:.4f}"
                
        concept_name = concept.replace('_', ' ').title()
        tex_lines.append(f"{concept_name} & {rho_str} & {mwu_str} \\\\")
        
    tex_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}"
    ])
    
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(tex_lines) + '\n')
        
    print(f"Correlation LaTeX table generated successfully at {out_path.absolute()}")

if __name__ == '__main__':
    generate_table()
