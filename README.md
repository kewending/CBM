# Objective 3: Interpretable Concept Bottleneck Models for Alzheimer's Detection

This repository contains the codebase for Research Objective 3, focusing on an explainable AI pipeline for detecting Alzheimer's disease from clinical speech. By leveraging **Concept Bottleneck Models (CBMs)**, the pipeline maps raw text inputs to human-interpretable clinical concepts (e.g., semantic specificity, syntactic simplification) before making a final diagnostic prediction.

This approach bridges the gap between high-accuracy deep learning (ModernBERT) and the transparent interpretability required in clinical settings.

## 🚀 The CBM Pipeline

The end-to-end pipeline consists of data extraction, contrastive dataset generation, probing for Concept Activation Vectors (CAVs), and training a sparse L1-regularized diagnostic model.

### 1. Data Extraction & Preprocessing
Extracts `*PAR` (participant-only) transcripts and audio from DementiaBank CHAT (`.cha`) files.
```powershell
uv run extract_par_data.py
```
- Filters out investigator prompts to isolate patient-specific cognitive markers.
- Merges audio files and generates train/test splits (`outputs/par_only/`).

### 2. LLM-Driven Concept Generation
Generates a contrastive mini-dataset for 10 clinically grounded linguistic markers using LLM prompting.
```powershell
uv run generate_llm_dataset.py
```
- Creates positive and negative exemplars for each clinical concept (e.g., *Noun-to-Pronoun Ratio*, *Semantic Paraphasias*).
- Applies programmatic weak supervision to filter valid pairs.

### 3. Concept Activation Vectors (CAV) Training
Bridges discrete concepts with continuous neural representations by training linear probes on ModernBERT layers.
```powershell
uv run run_cav_pipeline.py
```
- Extracts token-level embeddings using `BERT-base`.
- Trains Support Vector Machines (LinearSVC) to find the optimal layer and direction for each concept.
- Projects unseen patient transcripts onto these vectors to build the **Concept Bottleneck Matrix**.

### 4. Diagnostic Model Training
Trains the final L1-Regularized Logistic Regression (Lasso) model on the bottleneck matrix.
```powershell
uv run train_cbm.py
```
- Enforces sparsity to retain only the most clinically relevant features (e.g., pruning redundant or highly correlated markers).
- Outputs diagnostic probabilities and global concept weights.

### 5. Baselines & Evaluation
- **ModernBERT Baseline**: `uv run train_modernbert_cv.py` runs 5-fold stratified cross-validation to establish black-box baseline metrics.
- **Sparsity Analysis**: `uv run thesis/experiment_l1_sparsity.py` charts the L1 regularization path to evaluate the trade-off between conceptual sparsity and diagnostic accuracy.
- **Distributions**: `uv run thesis/plot_concept_distributions.py` plots the probability density functions of the retained vs. pruned concepts.

---

## 🔍 Interactive CBM & Audio Visualizer (Gradio)

To make the pipeline actionable for clinicians, we provide a unified interactive web application built with Gradio.

```powershell
uv run gradio_unified_cbm_explorer.py
```

### Features:
1. **Global Explanations**: A real-time prediction summary displaying the exact logistic regression formula, showing how the standardized concept scores combine to form the Dementia probability.
2. **Local Explanations (Heatmaps)**: Token-level attribution highlighting the specific words driving a concept's score. Select a row in the explanation table to instantly see the active linguistic phenomena.
3. **Multimodal Alignment**: 
   - Upload the source `.wav` audio.
   - Automatically synchronizes the text transcript with the audio waveform and spectrogram using **WhisperX**.
   - Features an Audacity-style click-drag zoom panel. The audio playback automatically updates to play only the visible region.

---

## 📂 Project Structure & Codebase Overview

```text
CBM/
├── cav/                        # Core library for Concept Activation Vectors
│   ├── config.py               # Central configuration (hyperparameters, paths)
│   ├── concepts.py             # Logic for weak supervision, LLM prompting & label extraction
│   ├── embed.py                # Extracts hidden layer embeddings from ModernBERT
│   ├── train.py                # Trains linear SVM probes on embeddings
│   ├── score.py                # Projects text into the CAV bottleneck space
│   └── validate.py             # Validates CAV probe accuracy
├── data/                       # Datasets, metadata, and notes (audio/transcripts ignored)
├── thesis/                     # Scripts generating plots, tables, and figures for the thesis
│   ├── experiment_*.py         # Runs specific sparsity or heatmap experiments
│   ├── generate_*.py           # Generates LaTeX tables or diagrams
│   └── plot_*.py               # Generates visualizations (distributions, boxplots)
├── README.md                   # Main project documentation (this file)
├── CBM_Pipeline_Overview.md    # Detailed breakdown of the theory and pipeline stages
├── concept_definitions.md      # Definitions of the 10 linguistic concepts used
├── extract_par_data.py         # Step 1: Preprocesses DementiaBank CHAT transcripts
├── generate_llm_dataset.py     # Step 2: Uses LLMs to generate contrastive concept examples
├── run_cav_pipeline.py         # Step 3: Orchestrates CAV training and embedding extraction
├── train_cbm.py                # Step 4: Trains the sparse L1 Logistic Regression model
├── train_modernbert_cv.py      # Baseline: 5-fold cross-validation of ModernBERT directly
├── alignment.py                # Audio-text forced alignment using WhisperX
├── gradio_unified_cbm_explorer.py # Interactive Web UI for visualizing explanations
├── pyproject.toml / uv.lock    # Python dependency management (uv)
└── .env.example                # Example environment variables file
```

---

## 📦 Installation & Setup

Install the required dependencies using `uv`:

```powershell
uv add gradio librosa matplotlib numpy soundfile plotly pandas scikit-learn transformers torch
```

Optional forced-alignment dependencies for the visualizer:
```powershell
uv add --optional forced-alignment whisperx torchaudio
```

*Note: On Windows, ensure `ffmpeg` is installed for WhisperX model/audio backend support.*

---
