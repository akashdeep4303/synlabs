from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)


def box(ax, xy, w, h, text, fc="#e8eef7"):
    p = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.08", linewidth=1.2, edgecolor="#1f3b73", facecolor=fc)
    ax.add_patch(p)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=8, wrap=True)


def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", color="#1f3b73", lw=1.2))


def rag_diagram() -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("Problem 1 — NimbusLedger RAG")
    box(ax, (0.3, 4.4), 2.2, 1.0, "PDF / HTML / MD\ncorpus")
    box(ax, (3.0, 4.4), 2.2, 1.0, "Chunk\nsize=800 ov=150")
    box(ax, (5.7, 4.4), 2.2, 1.0, "MiniLM-L6-v2\n384-d ONNX")
    box(ax, (8.4, 4.4), 2.3, 1.0, "Chroma cosine\n+ source, doc_type")
    arrow(ax, 2.5, 4.9, 3.0, 4.9)
    arrow(ax, 5.2, 4.9, 5.7, 4.9)
    arrow(ax, 7.9, 4.9, 8.4, 4.9)
    box(ax, (0.3, 2.4), 2.4, 1.1, "HTTP POST /query\nor CLI")
    box(ax, (3.2, 2.4), 2.4, 1.1, "Top-k retrieve\noptional doc_type filter")
    box(ax, (6.1, 3.1), 2.2, 0.9, "sim >= 0.28?\nYES → generate")
    box(ax, (6.1, 1.6), 2.2, 0.9, "NO → refuse\nno hallucination", fc="#fde8e8")
    box(ax, (8.7, 2.4), 2.1, 1.1, "LLM + citations\nllama3.2:3b")
    arrow(ax, 2.7, 2.95, 3.2, 2.95)
    arrow(ax, 5.6, 2.95, 6.1, 3.4)
    arrow(ax, 7.2, 2.05, 7.2, 1.6)
    arrow(ax, 8.3, 3.5, 8.7, 3.0)
    box(ax, (0.3, 0.3), 10.4, 0.9, "Logs: latency · chunk_count · tokens  |  Idempotent upsert by sha256(source|index|text)")
    fig.tight_layout()
    fig.savefig(OUT / "rag_architecture.png", dpi=140)
    plt.close()


def judge_diagram() -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("Problem 2 — LLM-as-judge pipeline")
    box(ax, (0.3, 4.3), 2.3, 1.1, "Suite JSON/YAML\ninput, prompt, output")
    box(ax, (3.0, 4.3), 2.4, 1.1, "Rubric + few-shot\nanchors 1/3/5")
    box(ax, (5.8, 4.3), 2.2, 1.1, "Judge call\nJUDGE_MODEL")
    box(ax, (8.4, 4.3), 2.3, 1.1, "Parse JSON\nelse regex fallback")
    arrow(ax, 2.6, 4.85, 3.0, 4.85)
    arrow(ax, 5.4, 4.85, 5.8, 4.85)
    arrow(ax, 8.0, 4.85, 8.4, 4.85)
    box(ax, (0.3, 2.3), 3.2, 1.2, "Pairwise path\nrun AB and BA\n(position bias)", fc="#fff4d6")
    box(ax, (4.0, 2.3), 3.2, 1.2, "Pointwise path\nper-criterion scores")
    box(ax, (7.6, 2.3), 3.1, 1.2, "Aggregate\npass rate, mean,\nwin rate, flip rate")
    arrow(ax, 1.9, 4.3, 1.9, 3.5)
    arrow(ax, 7.0, 4.3, 5.6, 3.5)
    arrow(ax, 9.5, 4.3, 9.1, 3.5)
    box(ax, (0.3, 0.35), 10.4, 1.3, "Probes: verbose-but-wrong · terse-but-correct · padded · confidently-wrong\nLogs: every judge prompt + raw response in logs/judge.jsonl")
    fig.tight_layout()
    fig.savefig(OUT / "judge_architecture.png", dpi=140)
    plt.close()


if __name__ == "__main__":
    rag_diagram()
    judge_diagram()
    print(f"wrote {OUT}")
