import os
from pathlib import Path

def generate_flowchart():
    tex_code = r"""
\documentclass{standalone}
\usepackage{tikz}
\usetikzlibrary{shapes.geometric, arrows.meta, positioning, fit, backgrounds}

\begin{document}
\begin{tikzpicture}[
    node distance=1.5cm and 2cm,
    box/.style={rectangle, draw=black!60, fill=blue!5, thick, minimum width=3cm, minimum height=1cm, align=center, font=\sffamily},
    concept/.style={rectangle, draw=black!60, fill=orange!5, thick, minimum width=3cm, minimum height=1cm, align=center, font=\sffamily},
    llm/.style={ellipse, draw=purple!60, fill=purple!5, thick, minimum width=3cm, minimum height=1cm, align=center, font=\sffamily},
    dataset/.style={cylinder, draw=black!60, fill=green!5, thick, minimum width=3cm, minimum height=1cm, shape border rotate=90, aspect=0.25, align=center, font=\sffamily},
    validator/.style={diamond, draw=red!60, fill=red!5, thick, minimum width=3.5cm, minimum height=1cm, align=center, font=\sffamily},
    arrow/.style={-{Stealth[scale=1.2]}, thick, draw=black!80}
]

% Nodes
\node (base) [box] {Base Healthy Transcripts\\(Cookie Theft Description)};
\node (concept) [concept, below=of base] {Clinical Concept Definition\\(e.g., Semantic Paraphasias)};
\node (prompt) [llm, right=of base, yshift=-1cm] {GPT-4 Augmentation\\Engine};

\node (pos) [dataset, right=of prompt, yshift=1.5cm] {Synthetic Positive Set ($D_c^+$)\\(Exhibits Concept)};
\node (neg) [dataset, right=of prompt, yshift=-1.5cm] {Synthetic Negative Set ($D_c^-$)\\(Concept Absent)};

\node (validator) [validator, right=of pos, yshift=-1.5cm] {Programmatic\\NLP Validation};
\node (filter) [box, right=of validator] {Strict Contrastive\\Filtering};
\node (final) [dataset, below=of filter] {Final Weakly-Supervised\\Dataset (CAV Training)};

% Arrows
\draw [arrow] (base) -| (prompt);
\draw [arrow] (concept) -| (prompt);

\draw [arrow] (prompt) |- (pos) node[pos=0.75, above, font=\sffamily\footnotesize] {Inject Concept};
\draw [arrow] (prompt) |- (neg) node[pos=0.75, below, font=\sffamily\footnotesize] {Control Generation};

\draw [arrow] (pos) -| (validator);
\draw [arrow] (neg) -| (validator);

\draw [arrow] (validator) -- (filter) node[midway, above, font=\sffamily\footnotesize] {Score Deviation};
\draw [arrow] (filter) -- (final) node[midway, right, font=\sffamily\footnotesize] {Verified Pairs};

% Groupings / Backgrounds
\begin{scope}[on background layer]
    \node[draw=gray!50, dashed, rounded corners, fit=(pos) (neg), fill=gray!5, inner sep=0.3cm] (synthetic) {};
    \node[above, font=\sffamily\bfseries, text=gray!80] at (synthetic.north) {Synthetic Generation};
\end{scope}

\end{tikzpicture}
\end{document}
"""

    out_path = Path("LLM_Dataset_Flowchart.tex")
    out_path.write_text(tex_code, encoding="utf-8")
    print(f"Flowchart LaTeX generated successfully at {out_path.absolute()}")

if __name__ == '__main__':
    generate_flowchart()
