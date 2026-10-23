from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    index: int
    start: int
    end: int


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[Chunk]:
    text = _normalize(text)
    if not text:
        return []
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    chunks: list[Chunk] = []
    start = 0
    idx = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_size)
        if end < n:
            end = _snap_to_break(text, start, end)
        piece = text[start:end].strip()
        if piece:
            chunks.append(Chunk(text=piece, index=idx, start=start, end=start + len(piece)))
            idx += 1
        if end >= n:
            break
        start = max(0, end - overlap)
        if start >= end:
            start = end
    return chunks


def _normalize(text: str) -> str:
    lines = [ln.strip() for ln in text.replace("\r\n", "\n").split("\n")]
    out: list[str] = []
    blank = False
    for ln in lines:
        if not ln:
            if not blank:
                out.append("")
            blank = True
        else:
            out.append(ln)
            blank = False
    return "\n".join(out).strip()


def _snap_to_break(text: str, start: int, end: int) -> int:
    window = text[start:end]
    for sep in ["\n\n", "\n", ". ", "; ", ", ", " "]:
        pos = window.rfind(sep)
        if pos >= chunk_min_keep(end - start):
            return start + pos + len(sep)
    return end


def chunk_min_keep(size: int) -> int:
    return max(40, int(size * 0.4))
