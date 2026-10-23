from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from bs4 import BeautifulSoup
from pypdf import PdfReader

SUPPORTED = {".md", ".html", ".htm", ".pdf", ".txt"}


@dataclass
class Document:
    path: Path
    text: str
    doc_type: str
    source: str


def load_file(path: Path, corpus_root: Path) -> Document | None:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED:
        return None
    if suffix in {".md", ".txt"}:
        text = path.read_text(encoding="utf-8")
        doc_type = "markdown" if suffix == ".md" else "text"
    elif suffix in {".html", ".htm"}:
        html = path.read_text(encoding="utf-8")
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = soup.get_text("\n")
        doc_type = "html"
    else:
        reader = PdfReader(str(path))
        text = "\n".join((page.extract_text() or "") for page in reader.pages)
        doc_type = "pdf"
    rel = str(path.relative_to(corpus_root))
    return Document(path=path, text=text, doc_type=doc_type, source=rel)


def iter_corpus(corpus_dir: Path) -> list[Document]:
    docs: list[Document] = []
    for path in sorted(corpus_dir.rglob("*")):
        if path.is_file():
            doc = load_file(path, corpus_dir)
            if doc and doc.text.strip():
                docs.append(doc)
    return docs
