import numpy as np
from scipy.stats import spearmanr, mannwhitneyu
from cav import config
from cav.concepts import WEAK_LABEL_FUNCTIONS

def run_validation_report(probe_dicts, transcripts, true_labels):
    # Prepare bottleneck matrix to do cross-concept correlations
    from cav.score import build_bottleneck_matrix
    print("Building bottleneck matrix for validation report...")
    matrix = build_bottleneck_matrix(transcripts, probe_dicts) # (N, K)
    
    report_lines = ["# CAV Validation Report\n"]
    
    # 1. CV Accuracy Table
    report_lines.append("## 1. Cross-Validation Accuracy\n")
    report_lines.append("| Concept | Best Layer | CV Accuracy | Status |")
    report_lines.append("|---------|------------|-------------|--------|")
    for p in probe_dicts:
        status = "OK" if p['cv_accuracy'] >= config.MIN_CV_ACCURACY else "FLAG (Low CV)"
        report_lines.append(f"| {p['concept']} | {p['layer']} | {p['cv_accuracy']:.2f} | {status} |")
    report_lines.append("\n")
    
    # 2. Top-5 / Bottom-5 Inspection & Ground Truth Alignment & Dementia Discrimination
    report_lines.append("## 2. Concept Analysis\n")
    
    for i, p in enumerate(probe_dicts):
        concept = p['concept']
        scores = matrix[:, i]
        
        report_lines.append(f"### {concept}")
        report_lines.append(f"**Best Layer:** {p['layer']} | **CV Acc:** {p['cv_accuracy']:.2f}\n")
        
        # Spearman Rho
        weak_fn = WEAK_LABEL_FUNCTIONS.get(concept)
        if weak_fn:
            weak_scores = [weak_fn(t) for t in transcripts]
            # Handle cases where all weak scores might be the same (std=0) leading to NaN
            if np.std(weak_scores) > 0 and np.std(scores) > 0:
                rho, pval = spearmanr(scores, weak_scores)
                align_status = "Good" if rho > 0.50 else ("Acceptable" if rho > 0.30 else "Poor/Noisy")
                report_lines.append(f"- **Ground Truth Alignment (Spearman rho):** {rho:.2f} ({align_status})")
            else:
                report_lines.append(f"- **Ground Truth Alignment (Spearman rho):** N/A (zero variance in weak labels)")
        
        # Mann-Whitney U test between dementia (1) and control (0)
        scores_dem = [float(s) for s, l in zip(scores, true_labels) if str(l) == '1']
        scores_ctrl = [float(s) for s, l in zip(scores, true_labels) if str(l) == '0']
        if scores_dem and scores_ctrl:
            try:
                stat, pval = mannwhitneyu(scores_dem, scores_ctrl)
                disc_status = "Significant" if pval < 0.05 else "Not significant"
                report_lines.append(f"- **Dementia Discrimination (Mann-Whitney U):** p={pval:.4f} ({disc_status})")
            except Exception as e:
                report_lines.append(f"- **Dementia Discrimination (Mann-Whitney U):** Error ({e})")
        
        report_lines.append("\n**Top 5 Transcripts (Highest Concept Score):**\n")
        ranked_indices = np.argsort(scores)[::-1]
        for idx in ranked_indices[:5]:
            # sanitize transcript text for markdown
            t_snippet = transcripts[idx][:150].replace('\n', ' ')
            report_lines.append(f"- `[{scores[idx]:+.2f}]` {t_snippet}...")
            
        report_lines.append("\n**Bottom 5 Transcripts (Lowest Concept Score):**\n")
        for idx in ranked_indices[-5:]:
            t_snippet = transcripts[idx][:150].replace('\n', ' ')
            report_lines.append(f"- `[{scores[idx]:+.2f}]` {t_snippet}...")
        report_lines.append("\n---\n")

    # 3. Cross-Concept Correlation Matrix
    report_lines.append("## 3. Cross-Concept Correlation (Redundancy Check)\n")
    # np.corrcoef expects variables in rows, so we transpose matrix
    corr_matrix = np.corrcoef(matrix.T)
    redundant_pairs = []
    
    K = len(probe_dicts)
    for i in range(K):
        for j in range(i + 1, K):
            if abs(corr_matrix[i, j]) > 0.85:
                redundant_pairs.append((probe_dicts[i]['concept'], probe_dicts[j]['concept'], corr_matrix[i, j]))
                
    if redundant_pairs:
        report_lines.append("> [!WARNING]")
        report_lines.append("> **Potentially Redundant Concepts (|r| > 0.85)**\n")
        for c1, c2, r in redundant_pairs:
            report_lines.append(f"> - **{c1}** and **{c2}** (r = {r:.2f})")
        report_lines.append("\n")
    else:
        report_lines.append("No highly redundant concept pairs (|r| > 0.85) found.\n")
        
    report_content = "\n".join(report_lines)
    config.VALIDATION_REPORT.parent.mkdir(parents=True, exist_ok=True)
    with config.VALIDATION_REPORT.open('w') as f:
        f.write(report_content)
        
    print(f"\nValidation report saved to {config.VALIDATION_REPORT}")
