import json
import os
import gc
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import torch

from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer

# Import utilities from the training script
from train_modernbert_cv import (
    build_tokenized_dataset,
    make_training_args,
    detect_precision,
    resolve_effective_max_length,
    compute_metrics
)

def eval_predictions(y_test, preds, probs):
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    roc = roc_auc_score(y_test, probs)
    
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    gmean = np.sqrt(rec * spec)
    
    return acc, prec, rec, spec, f1, gmean, roc

def eval_cbm(clf, X_test, y_test):
    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:, 1]
    return eval_predictions(y_test, preds, probs)

def generate_performance_table():
    bottleneck_path = Path('outputs/cav_llm/bottleneck_matrix.npy')
    train_path = Path('outputs/par_only/train.csv')
    test_path = Path('outputs/par_only/test.csv')
    out_path = Path('performance_table_1.tex')
    txt_out_path = Path('performance_results.txt')
    hyperparams_path = Path('outputs/modernbert_cv/best_hyperparameters.json')
    
    if not bottleneck_path.exists():
        print(f"Error: {bottleneck_path} not found.")
        return
        
    if not hyperparams_path.exists():
        print(f"Error: {hyperparams_path} not found. Please run train_modernbert_cv.py first.")
        return
        
    with open(hyperparams_path, 'r', encoding='utf-8') as f:
        best_config = json.load(f)
        
    print(f"Loaded BERT hyperparameters: {best_config}")
        
    bottleneck_matrix = np.load(bottleneck_path)
    
    train_df = pd.read_csv(train_path)
    train_df = train_df[train_df['par_transcript'].notna()].reset_index(drop=True)
    test_df = pd.read_csv(test_path)
    test_df = test_df[test_df['par_transcript'].notna()].reset_index(drop=True)
    
    num_train = len(train_df)
    
    X_train_cbm = bottleneck_matrix[:num_train]
    X_test_cbm = bottleneck_matrix[num_train:]
    
    y_train = train_df['label'].values
    y_test = test_df['label'].values
    
    # --- Setup ModernBERT Tokenizer & Datasets ---
    model_name = "answerdotai/ModernBERT-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    effective_max_length = resolve_effective_max_length(tokenizer, 512)
    precision = detect_precision()
    
    full_train_dataset = build_tokenized_dataset(train_df, tokenizer, "par_transcript", "label", effective_max_length, num_proc=1)
    test_dataset = build_tokenized_dataset(test_df, tokenizer, "par_transcript", "label", effective_max_length, num_proc=1)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    cbm_results = []
    bert_results = []
    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    print("Running 5-fold CV on training set, evaluating each fold on held-out test set...")
    for fold, (train_idx, _) in enumerate(skf.split(X_train_cbm, y_train)):
        print(f"\n===========================")
        print(f"   Processing Fold {fold + 1}/5")
        print(f"===========================\n")
        
        # --- CBM Fold ---
        print("Training CBM...")
        X_f_train_cbm = X_train_cbm[train_idx]
        y_f_train = y_train[train_idx]
        
        scaler_cbm = StandardScaler()
        X_f_train_cbm_s = scaler_cbm.fit_transform(X_f_train_cbm)
        X_test_cbm_s = scaler_cbm.transform(X_test_cbm)
        
        clf_cbm = LogisticRegressionCV(Cs=np.logspace(-3, 1, 20), penalty='l1', solver='saga', cv=5, scoring='roc_auc', max_iter=10000, random_state=42)
        clf_cbm.fit(X_f_train_cbm_s, y_f_train)
        cbm_metrics = eval_cbm(clf_cbm, X_test_cbm_s, y_test)
        cbm_results.append(cbm_metrics)
        print(f"CBM Fold {fold+1} Test AUC: {cbm_metrics[-1]:.3f}")
        
        # --- ModernBERT Fold ---
        print("\nTraining ModernBERT Baseline...")
        fold_train_dataset = full_train_dataset.select(train_idx.tolist())
        
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        
        training_args = make_training_args(
            output_dir=f"outputs/modernbert_cv/table_folds/fold_{fold+1}",
            train_batch_size=int(best_config["per_device_train_batch_size"]),
            learning_rate=float(best_config["learning_rate"]),
            num_train_epochs=float(best_config["num_train_epochs"]),
            weight_decay=0.01,
            warmup_ratio=0.1,
            gradient_accumulation_steps=1,
            bf16=precision.bf16,
            tf32=precision.tf32,
            seed=42 + fold,
            logging_steps=50,
            gradient_checkpointing=False,
            has_eval=False,
            save_checkpoints=False,
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=fold_train_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )
        
        trainer.train()
        
        # Evaluate ModernBERT on test set
        print("Evaluating ModernBERT on Test Set...")
        pred_out = trainer.predict(test_dataset)
        preds = np.argmax(pred_out.predictions, axis=-1)
        probs = torch.softmax(torch.tensor(pred_out.predictions), dim=-1)[:, 1].numpy()
        
        bert_metrics = eval_predictions(y_test, preds, probs)
        bert_results.append(bert_metrics)
        print(f"ModernBERT Fold {fold+1} Test AUC: {bert_metrics[-1]:.3f}")
        
        # Cleanup memory
        del trainer
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    cbm_means = np.mean(cbm_results, axis=0) * 100
    cbm_stds = np.std(cbm_results, axis=0) * 100
    
    bert_means = np.mean(bert_results, axis=0) * 100
    bert_stds = np.std(bert_results, axis=0) * 100
    
    # ROC is typically not percentage
    cbm_means[-1] /= 100
    cbm_stds[-1] /= 100
    bert_means[-1] /= 100
    bert_stds[-1] /= 100

    def format_row(name, means, stds):
        # order: acc, prec, rec, spec, f1, gmean, roc
        s = f"{name}"
        for i in range(6):
            s += f" & {means[i]:.2f}\\% $\\pm$ {stds[i]:.2f}\\%"
        s += f" & {means[6]:.3f} $\\pm$ {stds[6]:.3f} \\\\"
        return s
        
    # --- Generate Text Report ---
    txt_lines = ["Performance Results - 5-Fold CV on Held-Out Test Set", "="*60, ""]
    metric_names = ["Accuracy", "Precision", "Recall", "Specificity", "F1 Score", "G-Mean", "ROC-AUC"]
    
    txt_lines.append("--- ModernBERT Fold Results ---")
    for i, res in enumerate(bert_results):
        res_pct = [r * 100 if j < 6 else r for j, r in enumerate(res)]
        metrics_str = ", ".join([f"{name}: {val:.2f}" + ("%" if j < 6 else "") for j, (name, val) in enumerate(zip(metric_names, res_pct))])
        txt_lines.append(f"Fold {i+1}: {metrics_str}")
        
    txt_lines.append("\n--- CBM (LLM CAVs) Fold Results ---")
    for i, res in enumerate(cbm_results):
        res_pct = [r * 100 if j < 6 else r for j, r in enumerate(res)]
        metrics_str = ", ".join([f"{name}: {val:.2f}" + ("%" if j < 6 else "") for j, (name, val) in enumerate(zip(metric_names, res_pct))])
        txt_lines.append(f"Fold {i+1}: {metrics_str}")
        
    txt_lines.append("\n--- Summary (Mean ± Std) ---")
    bert_summary = ", ".join([f"{name}: {bert_means[j]:.2f} ± {bert_stds[j]:.2f}" + ("%" if j < 6 else "") for j, name in enumerate(metric_names)])
    txt_lines.append(f"ModernBERT: {bert_summary}")
    
    cbm_summary = ", ".join([f"{name}: {cbm_means[j]:.2f} ± {cbm_stds[j]:.2f}" + ("%" if j < 6 else "") for j, name in enumerate(metric_names)])
    txt_lines.append(f"CBM:        {cbm_summary}")
    
    with open(txt_out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(txt_lines) + '\n')
    print(f"Detailed performance results saved to {txt_out_path.absolute()}")

    tex_lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Performance comparison between baseline ModernBERT and the interpretable Concept Bottleneck Model (CBM) across 5-folds on the held-out test set.}",
        "\\label{tab:performance_comparison}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{lccccccc}",
        "\\toprule",
        "\\textbf{Model} & \\textbf{Accuracy} & \\textbf{Precision} & \\textbf{Recall} & \\textbf{Specificity} & \\textbf{F1 Score} & \\textbf{G-Mean} & \\textbf{ROC-AUC} \\\\",
        "\\midrule",
        format_row("ModernBERT Baseline", bert_means, bert_stds),
        format_row("CBM (LLM CAVs)", cbm_means, cbm_stds),
        "\\bottomrule",
        "\\end{tabular}%",
        "}",
        "\\end{table}"
    ]
    
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(tex_lines) + '\n')
        
    print(f"Performance LaTeX table generated successfully at {out_path.absolute()}")

if __name__ == '__main__':
    generate_performance_table()
