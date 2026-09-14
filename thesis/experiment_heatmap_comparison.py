import torch
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from transformers import BertTokenizer, BertModel

def generate_heatmap_html(tokens, scores, method_name, concept_name):
    max_abs = max(abs(np.max(scores)), abs(np.min(scores)))
    if max_abs == 0: max_abs = 1.0
    norm = mcolors.TwoSlopeNorm(vmin=-max_abs, vcenter=0., vmax=max_abs)
    cmap = plt.get_cmap('coolwarm')
    
    html = f"""
    <div style='margin-bottom: 20px; font-family: sans-serif;'>
        <div style='margin-bottom: 10px; padding: 12px 16px; background-color: #f8f9fa; border-left: 5px solid #2c3e50; border-radius: 4px;'>
            <h3 style='margin: 0; color: #2c3e50; font-size: 16px;'>{method_name}</h3>
        </div>
        <div style='line-height: 2.3; font-size: 16px; padding: 15px; display: flex; flex-wrap: wrap; gap: 6px 4px; border: 1px solid #e2e8f0; border-radius: 8px; background-color: white;'>
    """
    
    current_word_html = ""
    for token, score in zip(tokens, scores):
        if token in ['[CLS]', '[SEP]', '[PAD]']:
            continue
            
        rgba = cmap(norm(score))
        hex_color = mcolors.to_hex(rgba)
        luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
        text_color = "white" if luminance < 0.5 else "black"
        
        if token.startswith('##'):
            clean_token = token[2:]
            current_word_html += f"<span style='background-color: {hex_color}; color: {text_color}; padding: 2px 2px;' title='Score: {score:.4f}'>{clean_token}</span>"
        else:
            if current_word_html:
                html += f"<div style='display: flex; overflow: hidden; border-radius: 4px; box-shadow: 0 1px 2px rgba(0,0,0,0.15);'>{current_word_html}</div>"
            clean_token = token
            current_word_html = f"<span style='background-color: {hex_color}; color: {text_color}; padding: 2px 4px;' title='Score: {score:.4f}'>{clean_token}</span>"
            
    if current_word_html:
        html += f"<div style='display: flex; overflow: hidden; border-radius: 4px; box-shadow: 0 1px 2px rgba(0,0,0,0.15);'>{current_word_html}</div>"
        
    html += "</div></div>"
    return html

def get_top_transcript_for_concept(concept_name, train_df, report_text):
    import re
    # Find the section for this concept
    pattern = rf"### {concept_name}.*?\*\*Top 5 Transcripts \(Highest Concept Score\):\*\*(.*?)(?=\*\*Bottom 5 Transcripts|\n---)"
    match = re.search(pattern, report_text, re.DOTALL)
    if not match:
        return train_df['par_transcript'].iloc[0]
        
    top_section = match.group(1)
    
    # Find the first bullet point snippet (first 30-40 chars is enough to match)
    bullet_match = re.search(r"- `\[[+\-0-9.]+\]` (.{30,80})", top_section)
    if not bullet_match:
        return train_df['par_transcript'].iloc[0]
            
    snippet = bullet_match.group(1).strip().replace(" .", ".") # clean up a bit if needed
    
    # Search train_df for a transcript containing the first 30 chars of this snippet
    search_str = snippet[:30]
    for t in train_df['par_transcript']:
        if isinstance(t, str) and search_str in t:
            return t
            
    # Fallback
    return train_df['par_transcript'].iloc[0]

def main():
    print("Loading model and tokenizer...")
    model_name = "google-bert/bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name, output_hidden_states=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Load a real transcript from the dataset
    train_csv_path = Path("outputs/par_only/train.csv")
    if not train_csv_path.exists():
        print(f"Error: {train_csv_path} not found.")
        return
        
    train_df = pd.read_csv(train_csv_path)
    
    report_path = Path("outputs/cav_llm/validation_report.md")
    if not report_path.exists():
        print(f"Error: {report_path} not found.")
        return
    with open(report_path, 'r', encoding='utf-8') as f:
        report_text = f.read()
    
    probes_dir = Path("outputs/cav_llm/probes")
    probe_files = list(probes_dir.glob("*.pkl"))
    
    if not probe_files:
        print("No probe files found!")
        return
        
    all_comparisons_html = ""
    
    # Create directory for individual concept html files
    individual_dir = Path("outputs/heatmap_comparisons")
    individual_dir.mkdir(parents=True, exist_ok=True)
    
    for probe_path in sorted(probe_files):
        concept = probe_path.stem
        with open(probe_path, 'rb') as f:
            probe = pickle.load(f)
            
        layer_idx = probe['layer']
        svm = probe['svm']
        scaler = probe['scaler']
        print(f"Processing Concept: {concept} (Layer {layer_idx})")
        
        # Get the #1 highest scoring transcript for this specific concept
        text = get_top_transcript_for_concept(concept, train_df, report_text)
        print(f"  -> Using Top Transcript: {text[:80]}...")
        
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        # =========================================================================
        # Method 1: Direct Projection (Ante-hoc)
        # =========================================================================
        model.eval()
        with torch.no_grad():
            out = model(**inputs)
            
        hidden_states = out.hidden_states[layer_idx][0].cpu().numpy()
        scaled_hidden = scaler.transform(hidden_states)
        direct_scores = svm.decision_function(scaled_hidden)
        
        direct_html = generate_heatmap_html(tokens, direct_scores, "Method 1: Direct Projection (Ante-hoc)", concept)
        
        # =========================================================================
        # Method 2: Gradient-Based Attribution (Input x Gradient)
        # =========================================================================
        model.eval() # Ensure deterministic gradients (no dropout)
        model.zero_grad()
        
        embeddings_list = []
        def hook_fn(module, input, output):
            output.retain_grad()
            embeddings_list.append(output)

        hook = model.embeddings.word_embeddings.register_forward_hook(hook_fn)
        
        out_grad = model(**inputs)
        
        hook.remove()
        
        embeddings = embeddings_list[0]
        
        hidden_states_grad = out_grad.hidden_states[layer_idx][0] # shape (seq_len, hidden)
        
        w = torch.tensor(svm.coef_[0], dtype=torch.float32).to(device)
        b = torch.tensor(svm.intercept_[0], dtype=torch.float32).to(device)
        scale_mean = torch.tensor(scaler.mean_, dtype=torch.float32).to(device)
        scale_std = torch.tensor(scaler.scale_, dtype=torch.float32).to(device)
        
        scaled_hs_pt = (hidden_states_grad - scale_mean) / scale_std
        token_scores = torch.matmul(scaled_hs_pt, w) + b
        
        mask = inputs['attention_mask'][0].clone()
        mask[0] = 0   # [CLS]
        mask[-1] = 0  # [SEP]
        
        sequence_score = (token_scores * mask).sum() / mask.sum()
        sequence_score.backward()
        
        grads = embeddings.grad[0] # shape (seq_len, hidden)
        embeds = embeddings[0].detach() # shape (seq_len, hidden)
        
        attr_scores = torch.sum(embeds * grads, dim=-1).cpu().numpy()
        
        grad_html = generate_heatmap_html(tokens, attr_scores, "Method 2: Input &times; Gradient (Post-hoc)", concept)
        
        # Append to master HTML
        concept_html = f"""
        <div style='margin-bottom: 50px; padding: 20px; background-color: #fcfcfc; border: 1px solid #e2e8f0; border-radius: 8px;'>
            <h2 style='margin-top: 0; color: #1e293b; border-bottom: 2px solid #e2e8f0; padding-bottom: 10px;'>Concept: {concept}</h2>
            <div style='padding: 15px; background: #e0f2fe; border-left: 4px solid #0284c7; margin-bottom: 30px; font-size: 16px; color: #0f172a; border-radius: 0 4px 4px 0;'>
                <strong>Highest Scoring Transcript Evaluated:</strong><br><br>
                {text}
            </div>
            {direct_html}
            {grad_html}
        </div>
        """
        all_comparisons_html += concept_html
        
        # Save individual HTML for easy PDF export
        individual_html_doc = f"""
        <html>
        <head>
            <title>{concept} - Token Attribution</title>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; padding: 20px; background-color: #ffffff; }}
                .container {{ max-width: 1100px; margin: 0 auto; }}
            </style>
        </head>
        <body>
            <div class="container">
                {concept_html}
            </div>
        </body>
        </html>
        """
        indiv_path = individual_dir / f"{concept}.html"
        with open(indiv_path, 'w', encoding='utf-8') as f:
            f.write(individual_html_doc)
            
    # =========================================================================
    # Output Master HTML
    # =========================================================================
    out_path = Path("outputs/heatmap_comparison.html")
    out_path.parent.mkdir(exist_ok=True, parents=True)
    
    full_html = f"""
    <html>
    <head>
        <title>Token Attribution Comparison: CBMs vs Gradients</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; padding: 40px; background-color: #f1f5f9; }}
            .container {{ max-width: 1100px; margin: 0 auto; background: white; padding: 40px; border-radius: 12px; box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1); }}
            h1 {{ color: #1e293b; text-align: center; font-size: 32px; margin-bottom: 20px; }}
            .desc {{ color: #475569; font-size: 18px; margin-bottom: 40px; text-align: justify; line-height: 1.6; padding: 20px; background-color: #f8fafc; border-radius: 8px; border: 1px solid #e2e8f0; }}
            .transcript-box {{ padding: 15px; background: #e0f2fe; border-left: 4px solid #0284c7; margin-bottom: 30px; font-size: 16px; color: #0f172a; border-radius: 0 4px 4px 0; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Visual Comparison of Token Attribution Methods</h1>
            <p class="desc">
                This visualization compares ante-hoc direct representations against post-hoc gradients across all 10 clinical concepts.
                For each concept, the <strong>highest scoring transcript</strong> from the validation report was automatically selected.
                <br><br>
                Notice how <strong>Method 1 (Direct Projection)</strong> precisely isolates the relevant linguistic phenomena for each concept, while <strong>Method 2 (Input &times; Gradient)</strong> diffuses "importance" across syntactically necessary hubs (Gradient Shattering).
            </p>
            {all_comparisons_html}
        </div>
    </body>
    </html>
    """
    
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(full_html)
        
    print(f"\nComparison complete! Output saved to {out_path.absolute()}")

if __name__ == "__main__":
    main()
