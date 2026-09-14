import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
from cav import config
from pathlib import Path

def run_experiment():
    if not config.BOTTLENECK_OUT.exists():
        print("Bottleneck matrix not found.")
        return
        
    bottleneck_matrix = np.load(config.BOTTLENECK_OUT)
    with open(config.BOTTLENECK_NAMES, 'r') as f:
        concept_names = json.load(f)
        
    train_df = pd.read_csv(config.TRAIN_CSV)
    train_df = train_df[train_df['par_transcript'].notna()].reset_index(drop=True)
    labels = train_df['label'].values
    
    num_train = len(train_df)
    X_train_raw = bottleneck_matrix[:num_train]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train_raw)
    
    C_values = np.logspace(-3, 1, 30)
    
    mean_aucs = []
    mean_accs = []
    active_concepts = []
    
    for C in C_values:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        aucs, accs, non_zeros = [], [], []
        
        for train_idx, val_idx in skf.split(X_scaled, labels):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = labels[train_idx], labels[val_idx]
            
            clf = LogisticRegression(penalty='l1', solver='saga', C=C, max_iter=5000, random_state=42)
            clf.fit(X_train, y_train)
            
            preds = clf.predict(X_val)
            probs = clf.predict_proba(X_val)[:, 1]
            
            aucs.append(roc_auc_score(y_val, probs))
            accs.append(accuracy_score(y_val, preds))
            non_zeros.append(np.sum(np.abs(clf.coef_[0]) > 1e-4))
            
        mean_aucs.append(np.mean(aucs))
        mean_accs.append(np.mean(accs))
        active_concepts.append(np.mean(non_zeros))
        
    # Plotting the trade-off
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Regularization Strength (C = 1/λ)', fontsize=12)
    ax1.set_xscale('log')
    ax1.set_ylabel('Validation ROC-AUC', color=color, fontsize=12)
    ax1.plot(C_values, mean_aucs, color=color, marker='o', label='ROC-AUC')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Active Concepts (Non-zero Weights)', color=color, fontsize=12)
    ax2.plot(C_values, active_concepts, color=color, marker='s', linestyle='--', label='Active Concepts')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('CBM Sparsity vs. Performance Trade-off (L1 Regularization Path)', fontsize=14)
    fig.tight_layout()
    out_path = Path('L1_Regularization_Path.pdf')
    plt.savefig(out_path, format='pdf', bbox_inches='tight')
    print(f"Saved plot to {out_path.absolute()}")
    
    # Print out optimal C
    optimal_idx = np.argmax(mean_aucs)
    print(f"Optimal C: {C_values[optimal_idx]:.4f}")
    print(f"Max ROC-AUC: {mean_aucs[optimal_idx]:.4f}")
    print(f"Active Concepts at Max AUC: {active_concepts[optimal_idx]:.1f}")
    
    results_df = pd.DataFrame({
        'C': C_values,
        'ROC_AUC': mean_aucs,
        'Accuracy': mean_accs,
        'Active_Concepts': active_concepts
    })
    csv_path = Path('outputs/cav_llm/l1_sparsity_results.csv')
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(csv_path, index=False)

if __name__ == "__main__":
    run_experiment()
