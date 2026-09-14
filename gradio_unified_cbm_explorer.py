from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
import json

import gradio as gr
import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
import librosa
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from alignment import AlignmentError, WordTimestamp, align_words
from cav import config


# ==========================================
# 1. Audacity-Style Audio Visualizer Engine
# ==========================================

JS_GET_VISIBLE_RANGE = """
() => {
  const root = (typeof gradioApp === "function") ? gradioApp() : document;
  const panelRoot = root.querySelector("#main-panel");
  const plot = panelRoot ? panelRoot.querySelector(".js-plotly-plot") : root.querySelector(".js-plotly-plot");

  if (!plot || !plot.layout || !plot.layout.xaxis || !Array.isArray(plot.layout.xaxis.range)) {
    return [""];
  }

  const r = plot.layout.xaxis.range;
  if (!Array.isArray(r) || r.length !== 2) {
    return [""];
  }

  return [`${r[0]},${r[1]}`];
}
"""


@dataclass
class AudioData:
    source_path: str
    samples: np.ndarray
    sr: int
    duration: float
    waveform_t: np.ndarray
    waveform_y: np.ndarray
    spec_db: np.ndarray
    spec_t: np.ndarray
    spec_f: np.ndarray
    transcript_text: str = ""
    word_timestamps: list[WordTimestamp] | None = None


def load_audio(file_path: str | Path | None) -> AudioData:
    if not file_path:
        raise gr.Error("No source .wav file provided.")
    
    path_str = str(file_path)
    if not path_str.lower().endswith(".wav"):
        raise gr.Error("File is not a .wav file.")

    y, sr = librosa.load(path_str, sr=None, mono=True)
    if y.size == 0:
        raise gr.Error("Audio file is empty.")

    y = y.astype(np.float32)
    duration = float(y.size / sr)
    waveform_t, waveform_y = _downsample_waveform(y, sr, duration)
    spec_db, spec_t, spec_f = _build_mel_spectrogram(y, sr)

    return AudioData(
        source_path=path_str,
        samples=y,
        sr=sr,
        duration=duration,
        waveform_t=waveform_t,
        waveform_y=waveform_y,
        spec_db=spec_db,
        spec_t=spec_t,
        spec_f=spec_f,
        transcript_text="",
        word_timestamps=[],
    )


def _downsample_waveform(samples: np.ndarray, sr: int, duration: float) -> tuple[np.ndarray, np.ndarray]:
    max_points = 160_000
    if samples.size > max_points:
        step = max(1, samples.size // max_points)
        y = samples[::step]
    else:
        y = samples

    t = np.linspace(0.0, duration, num=y.size, endpoint=False)
    return t, y


def _build_mel_spectrogram(samples: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target_frames = 2200
    hop_length = max(128, samples.size // max(1, target_frames))
    n_fft = 2048 if samples.size >= 2048 else 512

    mel = librosa.feature.melspectrogram(
        y=samples,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=128,
        fmax=sr / 2,
    )
    spec_db = librosa.power_to_db(mel, ref=np.max)
    t = librosa.frames_to_time(np.arange(spec_db.shape[1]), sr=sr, hop_length=hop_length)
    f = librosa.mel_frequencies(n_mels=128, fmin=0.0, fmax=sr / 2)
    return spec_db, t, f


def clamp_range(start_sec: float, end_sec: float, duration: float) -> tuple[float, float]:
    start = max(0.0, min(float(start_sec), duration))
    end = max(0.0, min(float(end_sec), duration))
    if end <= start:
        end = min(duration, start + 0.01)
    return start, end


def build_combined_panel(audio: AudioData) -> go.Figure:
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.01,
        row_heights=[0.24, 0.56, 0.20],
    )

    fig.add_trace(
        go.Scattergl(
            x=audio.waveform_t,
            y=audio.waveform_y,
            mode="lines",
            line={"color": "#0B5FFF", "width": 1},
            name="Waveform",
            hovertemplate="t=%{x:.3f}s<br>amp=%{y:.4f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Heatmap(
            z=audio.spec_db,
            x=audio.spec_t,
            y=audio.spec_f,
            colorscale="Magma",
            colorbar={"title": "dB"},
            hovertemplate="t=%{x:.3f}s<br>f=%{y:.0f} Hz<br>dB=%{z:.1f}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    _add_transcript_subplot(fig, audio)

    fig.update_layout(
        height=650,
        margin={"l": 60, "r": 30, "t": 40, "b": 50},
        dragmode="zoom",
        showlegend=False,
        plot_bgcolor="#FFFFFF",
    )
    fig.update_yaxes(title_text="Amplitude", row=1, col=1)
    fig.update_yaxes(title_text="Frequency (Hz)", row=2, col=1)
    fig.update_yaxes(
        title_text="Transcript",
        row=3,
        col=1,
        range=[0.0, 1.0],
        showticklabels=False,
        showgrid=False,
        zeroline=False,
    )
    fig.update_xaxes(title_text="Time (s)", row=3, col=1)

    x_start, x_end = 0.0, audio.duration
    fig.update_xaxes(range=[x_start, x_end])

    return fig


def _add_transcript_subplot(fig: go.Figure, audio: AudioData) -> None:
    words = audio.word_timestamps or []
    if not words:
        fig.add_trace(
            go.Scatter(
                x=[audio.duration / 2 if audio.duration > 0 else 0.0],
                y=[0.5],
                mode="text",
                text=["No aligned word timestamps available"],
                textfont={"color": "#6B7280", "size": 12},
                hoverinfo="skip",
            ),
            row=3,
            col=1,
        )
        return

    text_x: list[float] = []
    text_y: list[float] = []
    text_words: list[str] = []
    hover_text: list[str] = []

    for word in words:
        fill = "#DBEAFE"
        fig.add_shape(
            type="rect",
            x0=word.start_sec,
            x1=word.end_sec,
            y0=0.0,
            y1=1.0,
            fillcolor=fill,
            line={"color": "#93C5FD", "width": 1},
            layer="below",
            row=3,
            col=1,
        )

        midpoint = (word.start_sec + word.end_sec) / 2
        text_x.append(midpoint)
        text_y.append(0.5)
        text_words.append(word.word)
        hover_text.append(
            f"{word.word}<br>{word.start_sec:.2f}s - {word.end_sec:.2f}s"
        )

    fig.add_trace(
        go.Scatter(
            x=text_x,
            y=text_y,
            mode="text",
            text=text_words,
            textfont={"size": 11, "color": "#0F172A"},
            hovertext=hover_text,
            hovertemplate="%{hovertext}<extra></extra>",
        ),
        row=3,
        col=1,
    )


def parse_visible_range_text(range_text: str | None, duration: float) -> tuple[float, float]:
    if not range_text:
        return 0.0, duration

    parts = str(range_text).split(",", maxsplit=1)
    if len(parts) != 2:
        return 0.0, duration

    try:
        start = float(parts[0].strip())
        end = float(parts[1].strip())
    except ValueError:
        return 0.0, duration

    return clamp_range(start, end, duration)


def clip_audio(audio: AudioData, start: float, end: float) -> tuple[int, np.ndarray]:
    s = int(start * audio.sr)
    e = int(end * audio.sr)
    segment = audio.samples[s:e]
    if segment.size == 0:
        segment = np.zeros(1, dtype=np.int16)
    return audio.sr, segment


def section_player_update(audio: AudioData, start: float, end: float):
    section = clip_audio(audio, start, end)
    label = f"Selected Section Playback ({start:.2f}s - {end:.2f}s)"
    return gr.update(value=section, label=label)


def auto_update_playback(range_text: str | None, audio: AudioData | None, last_range_key: str):
    if audio is None:
        return gr.skip(), gr.skip()

    start, end = parse_visible_range_text(range_text, audio.duration)
    range_key = f"{start:.2f},{end:.2f}"

    if range_key == (last_range_key or ""):
        return gr.skip(), gr.skip()

    return section_player_update(audio, start, end), range_key


# ==========================================
# 2. CBM & CAV Concept Explorer Engine
# ==========================================

# 2.1 Load Data
try:
    train_df = pd.read_csv(config.TRAIN_CSV)
    test_df = pd.read_csv(config.TEST_CSV)
    train_df = train_df.dropna(subset=['par_transcript']).reset_index(drop=True)
    test_df = test_df.dropna(subset=['par_transcript']).reset_index(drop=True)
    train_len = len(train_df)
    
    df = pd.concat([train_df, test_df], ignore_index=True)
    labels = df['label'].values
    transcript_dict = {
        f"[{'Train' if i < train_len else 'Test'} - {i}] {row['subject_id']} (Label: {row['label']})": (i, str(row['subject_id']), row['label'], row['par_transcript'])
        for i, row in df.iterrows()
    }
    transcript_options = list(transcript_dict.keys())
except Exception as e:
    print(f"Error loading transcripts: {e}")
    transcript_options = []
    transcript_dict = {}

# 2.2 Load Probes
probes = {}
try:
    probe_paths = list(config.PROBE_OUT_DIR.glob("*.pkl"))
    for p in probe_paths:
        with open(p, 'rb') as f:
            probe_data = pickle.load(f)
            probes[probe_data['concept']] = probe_data
except Exception as e:
    print(f"Error loading probes: {e}")

# 2.3 CBM Model & Scaler State
bottleneck_matrix = None
cbm_scaler = None
clf = None
saved_concept_names = []
weights = []

def init_cbm():
    global bottleneck_matrix, cbm_scaler, clf, saved_concept_names, weights
    if not config.BOTTLENECK_OUT.exists():
        print("Bottleneck matrix not found. Please run the CAV pipeline first.")
        return False
        
    print("Training CBM for Explorer...")
    bottleneck_matrix = np.load(config.BOTTLENECK_OUT)
    with open(config.BOTTLENECK_NAMES, 'r') as f:
        saved_concept_names = json.load(f)
        
    cbm_scaler = StandardScaler()
    X_scaled = cbm_scaler.fit_transform(bottleneck_matrix)
    
    # Train only on the train set to match evaluation weights
    X_train = X_scaled[:train_len]
    y_train = labels[:train_len]
    
    clf = LogisticRegressionCV(Cs=np.logspace(-3, 1, 20), penalty='l1', solver='saga', cv=5, scoring='roc_auc', max_iter=10000, random_state=42)
    clf.fit(X_train, y_train)
    weights = clf.coef_[0]
    print("CBM initialized with LLM CAVs.")
    return True

# 2.4 BERT Initialization (Lazy)
tokenizer = None
model = None
device = None

def init_model():
    global tokenizer, model, device
    if model is None:
        print("Initializing BERT model for Heatmaps...")
        tokenizer = BertTokenizer.from_pretrained(config.BERT_MODEL_NAME)
        model = BertModel.from_pretrained(config.BERT_MODEL_NAME, output_hidden_states=True)
        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        print("BERT Model initialized.")

def get_heatmap(text: str, concept_name: str) -> str:
    if concept_name not in probes:
        return f"<p>Error: Concept '{concept_name}' not found in loaded probes.</p>"
        
    init_model()
    probe = probes[concept_name]
    layer = probe['layer']
    
    enc = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    enc_gpu = {k: v.to(device) for k, v in enc.items()}
    
    with torch.no_grad():
        out = model(**enc_gpu)
        
    hidden_states = out.hidden_states[layer][0].cpu().numpy()
    scaled = probe['scaler'].transform(hidden_states)
    scores = probe['svm'].decision_function(scaled)
    tokens = tokenizer.convert_ids_to_tokens(enc['input_ids'][0])
    
    max_abs = max(abs(scores.max()), abs(scores.min()))
    if max_abs == 0: max_abs = 1.0
    norm = mcolors.TwoSlopeNorm(vmin=-max_abs, vcenter=0., vmax=max_abs)
    cmap = plt.get_cmap('coolwarm')
    
    html = f"""
    <div style='margin-bottom: 10px; padding: 8px 12px; background-color: #e8f4f8; border-left: 4px solid #2980b9; border-radius: 4px;'>
        <h4 style='margin: 0; color: #2c3e50;'>Word Heatmap for Concept: <strong>{concept_name}</strong></h4>
    </div>
    <div style='line-height: 2.3; font-family: sans-serif; font-size: 15px; padding: 10px; display: flex; flex-wrap: wrap; gap: 6px 4px; border: 1px solid #e2e8f0; border-radius: 8px; max-height: 220px; overflow-y: auto;'>
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
            current_word_html += f"<span style='background-color: {hex_color}; color: {text_color}; padding: 2px 2px;' title='Score: {score:.3f}'>{clean_token}</span>"
        else:
            if current_word_html:
                html += f"<div style='display: flex; overflow: hidden; border-radius: 4px; box-shadow: 0 1px 2px rgba(0,0,0,0.1);'>{current_word_html}</div>"
            clean_token = token
            current_word_html = f"<span style='background-color: {hex_color}; color: {text_color}; padding: 2px 4px;' title='Score: {score:.3f}'>{clean_token}</span>"
            
    if current_word_html:
        html += f"<div style='display: flex; overflow: hidden; border-radius: 4px; box-shadow: 0 1px 2px rgba(0,0,0,0.1);'>{current_word_html}</div>"
        
    html += "</div>"
    return html

def analyze_transcript(transcript_selection):
    if not transcript_selection or clf is None:
        return "Select a transcript...", pd.DataFrame(), "<p>Select a transcript.</p>"
        
    idx, subject_id, label, text = transcript_dict[transcript_selection]
    
    # Get CBM Explanation
    scores_unscaled = bottleneck_matrix[idx]
    scores = cbm_scaler.transform([scores_unscaled])[0]
    contributions = scores * weights
    logit = np.sum(contributions) + clf.intercept_[0]
    
    data = []
    for name, contrib, raw_score, w, std_score in zip(saved_concept_names, contributions, scores_unscaled, weights, scores):
        if abs(w) < 1e-5:
            continue
            
        direction = "↑" if contrib > 0 else "↓" if contrib < 0 else "-"
        data.append({
            "": direction,
            "Concept": name,
            "Weight": round(w, 4),
            "Std Score": round(std_score, 4),
            "Raw Score": round(raw_score, 4),
            "Impact": round(contrib, 4),
            "AbsImpact": abs(contrib)
        })
            
    df_exp = pd.DataFrame(data).sort_values("AbsImpact", ascending=False).drop(columns=["AbsImpact"])
    
    pred_prob = 1 / (1 + np.exp(-logit))
    pred_label = "Dementia" if pred_prob > 0.5 else "Control"
    true_label = "Dementia" if label == 1 else "Control"
    
    summary = f"**True Label:** {true_label} | **Model Prediction:** {pred_label} (Probability: {pred_prob:.1%})<br>**Logit Score:** {logit:.3f} = Intercept ({clf.intercept_[0]:.3f}) + sum(Std Score × Weight)"
    
    if not df_exp.empty:
        top_concept = df_exp.iloc[0]["Concept"]
        heatmap_html = get_heatmap(text, top_concept)
    else:
        heatmap_html = "<p>No active concepts found for this model.</p>"
        
    return summary, df_exp, heatmap_html

def recalculate_predictions(df_exp, transcript_selection):
    if df_exp is None or df_exp.empty or not transcript_selection or clf is None:
        return gr.update(), gr.update()
        
    try:
        concept_to_raw = dict(zip(df_exp["Concept"], df_exp["Raw Score"]))
        
        scores_unscaled = []
        for name in saved_concept_names:
            scores_unscaled.append(float(concept_to_raw.get(name, 0.0)))
            
        scores_unscaled = np.array(scores_unscaled)
        scores = cbm_scaler.transform([scores_unscaled])[0]
        
        contributions = scores * weights
        logit = np.sum(contributions) + clf.intercept_[0]
        pred_prob = 1 / (1 + np.exp(-logit))
        
        data = []
        for name, contrib, raw_score, w, std_score in zip(saved_concept_names, contributions, scores_unscaled, weights, scores):
            if abs(w) < 1e-5:
                continue
                
            direction = "↑" if contrib > 0 else "↓" if contrib < 0 else "-"
            data.append({
                "": direction,
                "Concept": name,
                "Weight": round(w, 4),
                "Std Score": round(std_score, 4),
                "Raw Score": round(raw_score, 4),
                "Impact": round(contrib, 4),
                "AbsImpact": abs(contrib)
            })
                
        new_df_exp = pd.DataFrame(data).sort_values("AbsImpact", ascending=False).drop(columns=["AbsImpact"])
        
        pred_label = "Dementia" if pred_prob > 0.5 else "Control"
        idx, subject_id, label, _ = transcript_dict[transcript_selection]
        true_label = "Dementia" if label == 1 else "Control"
        
        summary = f"**True Label:** {true_label} | **Model Prediction:** {pred_label} (Probability: {pred_prob:.1%})<br>**Logit Score:** {logit:.3f} = Intercept ({clf.intercept_[0]:.3f}) + sum(Std Score × Weight)"
        
        return summary, new_df_exp
    except Exception as e:
        print(f"Error recalculating: {e}")
        return gr.update(), gr.update()

def update_heatmap_from_df(evt: gr.SelectData, df_exp: pd.DataFrame, transcript_selection: str):
    if df_exp is None or df_exp.empty or not transcript_selection:
        return "<p>Selection Error</p>"
        
    row = evt.index[0]
    concept_name = df_exp.iloc[row]["Concept"]
    idx, subject_id, label, text = transcript_dict[transcript_selection]
    
    return get_heatmap(text, concept_name)


# ==========================================
# 3. Audio File Path Resolution
# ==========================================

def get_audio_path_for_subject(subject_id: str) -> Path | None:
    candidates = [
        Path("outputs/par_only/audio") / f"{subject_id}_PAR.wav",
        Path("outputs/par_only/audio") / f"{subject_id}.wav",
        Path("data/audio") / f"{subject_id}_PAR.wav",
        Path("data/audio") / f"{subject_id}.wav",
    ]
    for p in candidates:
        if p.exists():
            return p

    # Fallback glob check
    par_dir = Path("outputs/par_only/audio")
    if par_dir.exists():
        matches = list(par_dir.glob(f"{subject_id}*.wav"))
        if matches:
            return matches[0]

    data_dir = Path("data/audio")
    if data_dir.exists():
        matches = list(data_dir.glob(f"{subject_id}*.wav"))
        if matches:
            return matches[0]

    return None


# ==========================================
# 4. Integrated Selection Handler (Auto Align)
# ==========================================

def on_transcript_select(transcript_selection: str):
    if not transcript_selection or transcript_selection not in transcript_dict:
        return (
            "Select a transcript...",
            pd.DataFrame(),
            "<p>Select a transcript.</p>",
            "",
            None,
            None,
            gr.update(value=None, label="Selected Section Playback"),
            "",
            "Alignment status: Select a transcript to begin."
        )

    idx, subject_id, label, text = transcript_dict[transcript_selection]

    # 1. Get CBM & Heatmap predictions
    summary, df_exp, heatmap_html = analyze_transcript(transcript_selection)
    raw_text_val = text

    # 2. Find and load audio file
    audio_path = get_audio_path_for_subject(subject_id)
    if audio_path is None:
        status_msg = f"⚠️ No audio file found for subject '{subject_id}' in outputs/par_only/audio or data/audio."
        return (
            summary,
            df_exp,
            heatmap_html,
            raw_text_val,
            None,
            None,
            gr.update(value=None, label="Selected Section Playback"),
            "",
            status_msg
        )

    try:
        audio = load_audio(audio_path)
        audio.transcript_text = text

        # 3. Perform automatic forced alignment
        align_status = ""
        try:
            cleaned_text = " ".join((text or "").split())
            if cleaned_text:
                result = align_words(
                    audio_path=audio.source_path,
                    transcript_text=cleaned_text,
                    preferred_device="auto",
                    language_code="en",
                )
                audio.word_timestamps = result.words
                if result.words:
                    align_status = f"✅ Auto-aligned {len(result.words)} words ({result.backend}, device={result.device})."
                else:
                    align_status = "⚠️ Alignment completed but returned no word timestamps."
            else:
                align_status = "⚠️ Transcript text empty."
        except Exception as align_exc:
            print(f"Auto alignment error for {subject_id}: {align_exc}")
            align_status = f"ℹ️ Audio loaded ({audio_path.name}). Alignment unavailable: {align_exc}"

        panel = build_combined_panel(audio)

        start, end = 0.0, audio.duration
        range_key = f"{start:.2f},{end:.2f}"
        section_update = section_player_update(audio, start, end)

        return (
            summary,
            df_exp,
            heatmap_html,
            raw_text_val,
            audio,
            panel,
            section_update,
            range_key,
            align_status
        )
    except Exception as exc:
        print(f"Error loading audio for {subject_id}: {exc}")
        status_msg = f"Error loading audio file {audio_path}: {exc}"
        return (
            summary,
            df_exp,
            heatmap_html,
            raw_text_val,
            None,
            None,
            gr.update(value=None, label="Selected Section Playback"),
            "",
            status_msg
        )


# ==========================================
# 5. Gradio Interface Builder
# ==========================================

def build_interface():
    is_ready = init_cbm()
    
    with gr.Blocks(title="CBM & Audio Visualizer Explorer", theme=gr.themes.Soft()) as app:
        gr.Markdown("# Concept Bottleneck Model (CBM) & Heatmap Explorer")
        gr.Markdown(
            "Select a participant transcript to analyze CBM classification explanations, explore token-level concept heatmaps, "
            "and inspect the **Audacity-Style Audio Visualizer** with auto-aligned word timestamps and interactive section audio playback."
        )
        
        if not is_ready:
            gr.Markdown("### ⚠️ Bottleneck matrix missing. Please run `run_cav_pipeline.py` first.")
            return app
            
        # Audio & Alignment state elements
        audio_state = gr.State(value=None)
        last_range_state = gr.State(value="")
        zoom_range_text = gr.Textbox(value="", visible=False)
        zoom_poller = gr.Timer(value=0.35, active=True)

        transcript_dd = gr.Dropdown(choices=transcript_options, label="Select Participant Transcript", interactive=True)
        
        with gr.Row():
            # Left Column: CBM Explanations & Raw Text
            with gr.Column(scale=1):
                summary_md = gr.Markdown("### Prediction Summary")
                explanation_df = gr.Dataframe(
                    headers=["", "Concept", "Weight", "Std Score", "Raw Score", "Impact"],
                    column_widths=["30px", "30%", "15%", "15%", "15%", "15%"],
                    interactive=True,
                    label="Contributing Factors (Edit 'Raw Score' to simulate changes, Click row for heatmap)"
                )
                raw_text = gr.Textbox(label="Raw Transcript Text", lines=7, interactive=False)
                
            # Right Column: Word Heatmap + Audacity-Style Audio Visualizer
            with gr.Column(scale=2):
                output_html = gr.HTML(
                    label="Concept Heatmap Visualization",
                    value="<div style='padding: 15px; text-align: center; color: #666;'>Select a transcript to view concept heatmaps.</div>"
                )
                
                gr.Markdown("### 🎵 Audacity-Style Audio Visualizer")
                section_player = gr.Audio(label="Selected Section Playback", type="numpy")

                alignment_status = gr.Markdown("Alignment status: waiting for transcript selection.")

                panel = gr.Plot(label="Waveform + Spectrogram + Transcript", elem_id="main-panel")

        # Event Handlers
        transcript_dd.change(
            fn=on_transcript_select,
            inputs=[transcript_dd],
            outputs=[
                summary_md,
                explanation_df,
                output_html,
                raw_text,
                audio_state,
                panel,
                section_player,
                last_range_state,
                alignment_status
            ]
        )

        explanation_df.change(
            fn=recalculate_predictions,
            inputs=[explanation_df, transcript_dd],
            outputs=[summary_md, explanation_df]
        )
        
        explanation_df.select(
            fn=update_heatmap_from_df,
            inputs=[explanation_df, transcript_dd],
            outputs=[output_html]
        )

        zoom_poller.tick(
            fn=None,
            inputs=[],
            outputs=[zoom_range_text],
            js=JS_GET_VISIBLE_RANGE,
            queue=False,
            show_progress="hidden",
        )

        zoom_range_text.change(
            fn=auto_update_playback,
            inputs=[zoom_range_text, audio_state, last_range_state],
            outputs=[section_player, last_range_state],
            queue=False,
            show_progress="hidden",
        )

    return app


if __name__ == "__main__":
    app = build_interface()
    app.launch(share=False)
