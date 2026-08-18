"""
Ingestion entrypoint.

Usage:
    python scripts/ingest.py
    python scripts/ingest.py --no-website
    python scripts/ingest.py --data-dir data/raw --max-website-pages 100

Triggers the ingestion pipeline (graph/ingestion_graph.py). All actual
loading/cleaning/chunking/embedding/upsert logic lives in ingestion/ and
graph/ingestion_graph.py -- this script only wires up CLI args and calls it.
"""

from __future__ import annotations

import argparse
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the ingestion pipeline.")
    parser.add_argument("--data-dir", default="data/raw", help="Root directory containing pdfs/ and documents/")
    parser.add_argument(
        "--no-website", action="store_true", help="Skip crawling the configured website; files only."
    )
    parser.add_argument(
        "--max-website-pages", type=int, default=50, help="Max pages to crawl if website ingestion is enabled."
    )
    args = parser.parse_args()

    from graph.ingestion_graph import run_ingestion

    count = run_ingestion(
        data_dir=args.data_dir,
        include_website=not args.no_website,
        max_website_pages=args.max_website_pages,
    )
    print(f"Ingestion complete. Upserted {count} chunks into Qdrant.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
