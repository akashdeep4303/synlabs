from __future__ import annotations

from pathlib import Path


def write_pdf(path: Path, title: str, body: str) -> None:
    """Minimal single-page PDF (Helvetica) so the corpus includes real PDFs without extra deps."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [title, ""]
    for para in body.split("\n"):
        para = para.strip()
        if not para:
            lines.append("")
            continue
        while para:
            lines.append(para[:90])
            para = para[90:]
    lines = lines[:48]

    def esc(s: str) -> str:
        return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    y = 760
    ops = ["BT", "/F1 14 Tf", f"50 {y} Td", f"({esc(lines[0])}) Tj"]
    for ln in lines[1:]:
        ops.append("0 -16 Td")
        ops.append("/F1 10 Tf" if ln != title else "/F1 14 Tf")
        ops.append(f"({esc(ln)}) Tj")
    ops.append("ET")
    stream = "\n".join(ops).encode("latin-1", "replace")
    objs = []
    objs.append(b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n")
    objs.append(b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n")
    objs.append(
        b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj\n"
    )
    objs.append(
        b"4 0 obj << /Length "
        + str(len(stream)).encode()
        + b" >> stream\n"
        + stream
        + b"\nendstream endobj\n"
    )
    objs.append(b"5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n")
    header = b"%PDF-1.4\n"
    offsets = [0]
    body_bytes = header
    pos = len(header)
    for obj in objs:
        offsets.append(pos)
        body_bytes += obj
        pos += len(obj)
    xref_pos = pos
    xref = f"xref\n0 {len(objs)+1}\n0000000000 65535 f \n".encode()
    for off in offsets[1:]:
        xref += f"{off:010d} 00000 n \n".encode()
    trailer = (
        f"trailer << /Size {len(objs)+1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n".encode()
    )
    path.write_bytes(body_bytes + xref + trailer)
