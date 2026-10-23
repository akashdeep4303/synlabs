from __future__ import annotations

import argparse
import json
import time

from src.rag.store import ingest_corpus


def main() -> None:
    parser = argparse.ArgumentParser(description="Idempotent corpus ingest into Chroma.")
    parser.add_argument("--corpus", default=None, help="Corpus directory (default: data/corpus)")
    args = parser.parse_args()
    t0 = time.perf_counter()
    stats = ingest_corpus(args.corpus)
    stats["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
