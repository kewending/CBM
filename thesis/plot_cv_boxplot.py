import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re

def parse_results(filepath):
    data = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    current_model = None
    metrics = ["Accuracy", "Precision", "Recall", "Specificity", "F1 Score", "G-Mean", "ROC-AUC"]
    
    for line in lines:
        line = line.strip()
        if "--- ModernBERT Fold Results ---" in line:
            current_model = 'ModernBERT Baseline'
        elif "--- CBM (LLM CAVs) Fold Results ---" in line:
            current_model = 'CBM (LLM CAVs)'
        elif "--- Summary" in line:
            current_model = None
            
        if current_model and line.startswith("Fold"):
            for metric in metrics:
                # Capture the float value and optional % sign
                match = re.search(rf"{metric}:\s*([\d\.]+)(%?)", line)
                if match:
                    val = float(match.group(1))
                    has_pct = match.group(2) == '%'
                    if has_pct:
                        val /= 100.0
                    
                    data.append({'Model': current_model, 'Metric': metric, 'Score': val})
                
    return data

def plot_boxplot():
    results_path = Path('performance_results.txt')
    if not results_path.exists():
        print(f"Error: {results_path} not found.")
        return
        
    data = parse_results(results_path)
    if not data:
        print("No valid fold data found in the results file.")
        return
        
    df = pd.DataFrame(data)
    
    # Increased width to fit all 7 metrics comfortably
    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid")
    
    ax = sns.boxplot(x='Metric', y='Score', hue='Model', data=df, palette='Set2')
    
    # Using palette='dark:black' fixes the FutureWarning from Seaborn
    sns.stripplot(x='Metric', y='Score', hue='Model', data=df, dodge=True, 
                  jitter=True, palette='dark:black', alpha=0.6, ax=ax, legend=False)
    
    handles, labels = ax.get_legend_handles_labels()
    # We only want the first two handles (the boxplots, not the stripplots)
    if len(handles) >= 2:
        ax.legend(handles[:2], labels[:2], title='', loc='lower right')
    
    plt.title('5-Fold Test Set Performance Comparison')
    plt.ylabel('Score')
    plt.xlabel('')
    plt.ylim(0.5, 1.0)
    
    out_path = Path('Ch5_Boxplot.pdf')
    plt.savefig(out_path, format='pdf', bbox_inches='tight')
    print(f"Boxplot saved successfully to {out_path.absolute()}")

if __name__ == "__main__":
    plot_boxplot()
