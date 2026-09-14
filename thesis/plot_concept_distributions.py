import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from pathlib import Path
from cav import config
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats

def run():
    if not config.BOTTLENECK_OUT.exists():
        print("Bottleneck matrix not found.")
        return
        
    bottleneck_matrix = np.load(config.BOTTLENECK_OUT)
    with open(config.BOTTLENECK_NAMES, 'r') as f:
        concept_names = json.load(f)
        
    train_df = pd.read_csv(config.TRAIN_CSV)
    test_df = pd.read_csv(config.TEST_CSV)
    df = pd.concat([train_df, test_df], ignore_index=True)
    df = df[df['par_transcript'].notna()].reset_index(drop=True)
    labels = df['label'].values
    
    # Standardize to match what the model sees
    scaler = StandardScaler()
    scaled_matrix = scaler.fit_transform(bottleneck_matrix)
    
    # Create a dataframe for plotting
    plot_df = pd.DataFrame(scaled_matrix, columns=concept_names)
    plot_df['Label'] = ['Dementia' if l == 1 else 'Control' for l in labels]
    
    # Plot all 10 concepts using matplotlib directly
    fig, axes = plt.subplots(5, 2, figsize=(15, 20))
    axes = axes.flatten()
    
    # Pruned vs Kept knowledge dynamically from the latest CBM training
    weights_path = Path("outputs/cav_llm/cbm_feature_weights.csv")
    if weights_path.exists():
        weights_df = pd.read_csv(weights_path)
        pruned = weights_df[weights_df['Status'] == 'Pruned']['Concept'].tolist()
    else:
        print("Warning: cbm_feature_weights.csv not found. Treating all concepts as retained. Run train_cbm.py first.")
        pruned = []
    
    for i, concept in enumerate(concept_names):
        ax = axes[i]
        
        # Density for Dementia
        dementia_data = plot_df[plot_df['Label'] == 'Dementia'][concept]
        density_d = stats.gaussian_kde(dementia_data)
        x_d = np.linspace(dementia_data.min() - 1, dementia_data.max() + 1, 200)
        ax.fill_between(x_d, density_d(x_d), alpha=0.5, color='tab:red', label='Dementia')
        
        # Density for Control
        control_data = plot_df[plot_df['Label'] == 'Control'][concept]
        density_c = stats.gaussian_kde(control_data)
        x_c = np.linspace(control_data.min() - 1, control_data.max() + 1, 200)
        ax.fill_between(x_c, density_c(x_c), alpha=0.5, color='tab:blue', label='Control')
        
        status = "PRUNED" if concept in pruned else "RETAINED"
        color = "red" if status == "PRUNED" else "green"
        ax.set_title(f"{concept} [{status}]", fontsize=14, fontweight='bold', color=color)
        ax.set_xlabel("Standardized Concept Score", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.legend()
        
    plt.tight_layout()
    out_path = Path('Concept_Distributions.pdf')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, format='pdf', bbox_inches='tight')
    print(f"Saved concept distributions to {out_path.absolute()}")

if __name__ == "__main__":
    run()
