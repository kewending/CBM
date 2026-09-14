import numpy as np
import pandas as pd
import json
import argparse
from pathlib import Path
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
from cav import config

def load_data():
    if not config.BOTTLENECK_OUT.exists():
        raise FileNotFoundError(f"Bottleneck matrix not found at {config.BOTTLENECK_OUT}. Run CAV pipeline first.")
        
    bottleneck_matrix = np.load(config.BOTTLENECK_OUT)
    
    with open(config.BOTTLENECK_NAMES, 'r') as f:
        concept_names = json.load(f)
        
    # Reconstruct labels the same way the pipeline did
    train_df = pd.read_csv(config.TRAIN_CSV)
    test_df = pd.read_csv(config.TEST_CSV)
    train_df = train_df[train_df['par_transcript'].notna()].reset_index(drop=True)
    test_df = test_df[test_df['par_transcript'].notna()].reset_index(drop=True)
    train_len = len(train_df)
    
    df = pd.concat([train_df, test_df], ignore_index=True)
    labels = df['label'].values
    transcripts = df['par_transcript'].values
    
    assert len(labels) == len(bottleneck_matrix), "Mismatch between bottleneck matrix size and transcripts."
    return bottleneck_matrix, labels, transcripts, concept_names, train_len

def explain_prediction(transcript_idx, bottleneck_matrix, clf, scaler, concept_names, true_label):
    # Scale the single sample
    scores_unscaled = bottleneck_matrix[transcript_idx]
    scores = scaler.transform([scores_unscaled])[0]
    
    weights = clf.coef_[0]
    contributions = scores * weights
    
    sorted_contrib = sorted(
        zip(concept_names, contributions, scores_unscaled),
        key=lambda x: abs(x[1]), reverse=True
    )
    
    pred_prob = clf.predict_proba([scores])[0][1]
    pred_label = 1 if pred_prob > 0.5 else 0
    
    print("\n" + "="*50)
    print(f"Explanation for Transcript Index: {transcript_idx}")
    print(f"True Label: {'Dementia' if true_label == 1 else 'Control'}")
    print(f"Prediction: {'Dementia' if pred_label == 1 else 'Control'}")
    print(f"Probability: {pred_prob:.1%}\n")
    print("Contributing factors:")
    
    for name, contrib, raw_score in sorted_contrib:
        if abs(contrib) < 1e-4:
            continue # Skip zeroed out concepts
        direction = "+" if contrib > 0 else "-"
        print(f"  {direction} {name:30s} score={raw_score:>5.2f}, impact={contrib:>+.3f}")
    print("="*50 + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--explain_idx", type=int, default=0, help="Index of transcript to explain")
    args = parser.parse_args()
    
    print("Loading bottleneck data...")
    bottleneck_matrix, labels, transcripts, concept_names, train_len = load_data()
    
    print("Training Sparse L1 Concept Bottleneck Model...")
    # Scale the bottleneck matrix first for L1 regularization to be fair across concepts
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(bottleneck_matrix)
    
    clf = LogisticRegressionCV(
        Cs=np.logspace(-3, 1, 20),
        penalty='l1',
        solver='saga',
        cv=5,
        scoring='roc_auc',
        max_iter=10000,
        random_state=42
    )
    
    X_train = X_scaled[:train_len]
    y_train = labels[:train_len]
    X_test = X_scaled[train_len:]
    y_test = labels[train_len:]
    
    clf.fit(X_train, y_train)
    
    # Evaluate on Test Set
    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, preds)
    roc = roc_auc_score(y_test, probs)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)
    spec = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0.0
    
    print(f"\n--- CBM Performance ---")
    print(f"Accuracy:    {acc:.3f}")
    print(f"ROC-AUC:     {roc:.3f}")
    print(f"Precision:   {prec:.3f}")
    print(f"Recall:      {rec:.3f}")
    print(f"F1 Score:    {f1:.3f}")
    print(f"Specificity: {spec:.3f}")
    
    # Print and save the discovered weights
    print("\n--- Learned Concept Weights (L1 Sparse) ---")
    weights = clf.coef_[0]
    intercept = clf.intercept_[0]
    print(f"Intercept: {intercept:+.3f}\n")
    
    selected_concepts = 0
    report_data = []
    
    for name, w in zip(concept_names, weights):
        if abs(w) > 1e-4:
            selected_concepts += 1
            print(f"{name:30s}: {w:+.3f}")
            report_data.append({"Concept": name, "Weight": w, "Status": "Selected"})
        else:
            print(f"{name:30s}: 0.000 (pruned)")
            report_data.append({"Concept": name, "Weight": 0.0, "Status": "Pruned"})
            
    print(f"\nModel selected {selected_concepts} out of {len(concept_names)} concepts.")
    
    # Save report
    report_df = pd.DataFrame(report_data)
    out_csv = Path("outputs/cav_llm/cbm_feature_weights.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(out_csv, index=False)
    
    out_txt = Path("outputs/cav_llm/cbm_model_summary.txt")
    with open(out_txt, "w") as f:
        f.write("Concept Bottleneck Model (L1 Sparse) Summary\n")
        f.write("============================================\n\n")
        f.write(f"Intercept: {intercept:.4f}\n\n")
        f.write("Selected Features:\n")
        for idx, row in report_df[report_df['Status'] == 'Selected'].iterrows():
            f.write(f"{row['Concept']:30s}: {row['Weight']:+.4f}\n")
        f.write("\nPruned Features:\n")
        for idx, row in report_df[report_df['Status'] == 'Pruned'].iterrows():
            f.write(f"{row['Concept']:30s}: 0.0000\n")
            
    print(f"Feature weights report saved to {out_csv} and {out_txt}\n")
    
    # Explain a specific transcript
    explain_prediction(args.explain_idx, bottleneck_matrix, clf, scaler, concept_names, labels[args.explain_idx])

if __name__ == "__main__":
    main()
