#!/usr/bin/env python3
"""Eval harness for the RAG pipeline.

Hits a running instance of app.py over HTTP (not an in-process import),
so it measures the system exactly as deployed. Two metrics, both free
and deterministic (no LLM judge, no extra dependencies):

  - retrieval hit rate: for questions with an expected_source, did that
    source appear among the sources the answer cited?
  - refusal accuracy: for questions with expected_source == null
    (intentionally out-of-domain), did the system correctly decline
    rather than answer from general knowledge?

This is a real, if narrow, quality signal: it directly measures whether
retrieval/reranking find the right document and whether the groundedness
guardrail (app.py's low_confidence / no-context paths) actually fires on
questions the indexed documents don't cover. It does NOT measure answer
correctness or fluency -- that needs either human review or an LLM-judge
metric (e.g. RAGAS), deliberately left out here since it adds a heavy,
fragile dependency chain for a signal this harness has no way to validate
without live model access.

Usage:
    python eval/run_eval.py \
        --api-url http://localhost:8000 \
        --api-key $RAG_API_KEY \
        --dataset eval/dataset.jsonl \
        --index-dir eval/fixtures

Requires a running app.py instance with real credentials (ANTHROPIC_API_KEY,
RAG_API_KEY set on the server) -- this is a live-system check, not a unit
test, and is intentionally not run in CI.
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

REFUSAL_PHRASES = (
    "don't have any indexed documents",
    "don't cover this",
    "documents don't cover",
)


def load_dataset(path: Path) -> list[dict]:
    items = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def index_documents(api_url: str, headers: dict, index_dir: Path) -> None:
    pdfs = sorted(index_dir.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {index_dir}, skipping indexing.")
        return
    for pdf_path in pdfs:
        with pdf_path.open("rb") as f:
            response = requests.post(
                f"{api_url}/ingest",
                files={"file": (pdf_path.name, f, "application/pdf")},
                headers=headers,
            )
        if response.status_code != 200:
            print(f"WARNING: failed to index {pdf_path.name}: {response.status_code} {response.text}")
        else:
            print(f"Indexed {pdf_path.name}: {response.json()}")


def parse_response(text: str) -> tuple[list[str], str]:
    """Split the streamed /query/advanced body into (sources, answer_text)."""
    first_line, _, rest = text.partition("\n")
    if first_line.startswith("METADATA:"):
        metadata = json.loads(first_line[len("METADATA:"):])
        return metadata.get("sources", []), rest.strip()
    return [], text.strip()


def run_question(api_url: str, headers: dict, item: dict) -> dict:
    start = time.monotonic()
    response = requests.post(
        f"{api_url}/query/advanced",
        json={"question": item["question"], "history": []},
        headers=headers,
    )
    latency = time.monotonic() - start

    if response.status_code != 200:
        return {
            **item,
            "status_code": response.status_code,
            "error": response.text,
            "latency_seconds": round(latency, 3),
        }

    sources, answer = parse_response(response.text)
    expected_source = item.get("expected_source")

    result = {
        **item,
        "status_code": 200,
        "sources": sources,
        "answer": answer,
        "latency_seconds": round(latency, 3),
    }

    if expected_source is None:
        result["correctly_refused"] = any(p in answer.lower() for p in REFUSAL_PHRASES)
    else:
        result["retrieval_hit"] = expected_source in sources

    return result


def summarize(results: list[dict]) -> dict:
    retrieval_results = [r for r in results if "retrieval_hit" in r]
    refusal_results = [r for r in results if "correctly_refused" in r]

    summary = {
        "total_questions": len(results),
        "errors": sum(1 for r in results if r.get("status_code") != 200),
        "avg_latency_seconds": (
            round(sum(r.get("latency_seconds", 0) for r in results) / len(results), 3)
            if results
            else 0
        ),
    }
    if retrieval_results:
        hits = sum(1 for r in retrieval_results if r["retrieval_hit"])
        summary["retrieval_hit_rate"] = round(hits / len(retrieval_results), 3)
        summary["retrieval_questions"] = len(retrieval_results)
    if refusal_results:
        correct = sum(1 for r in refusal_results if r["correctly_refused"])
        summary["refusal_accuracy"] = round(correct / len(refusal_results), 3)
        summary["refusal_questions"] = len(refusal_results)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--dataset", type=Path, default=Path(__file__).parent / "dataset.jsonl")
    parser.add_argument(
        "--index-dir", type=Path, default=None, help="Directory of PDFs to ingest before evaluating"
    )
    parser.add_argument(
        "--min-hit-rate",
        type=float,
        default=None,
        help="Exit non-zero if retrieval_hit_rate falls below this",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    headers = {"X-API-Key": args.api_key}

    if args.index_dir:
        index_documents(args.api_url, headers, args.index_dir)

    dataset = load_dataset(args.dataset)
    if not dataset:
        print(f"No questions found in {args.dataset}")
        return 1

    results = [run_question(args.api_url, headers, item) for item in dataset]
    summary = summarize(results)

    print("\n=== Eval Results ===")
    for r in results:
        status = "OK" if r.get("status_code") == 200 else f"ERROR({r.get('status_code')})"
        if "retrieval_hit" in r:
            outcome = "HIT " if r["retrieval_hit"] else "MISS"
            print(f"[{outcome}] [{status}] {r['question']!r} -> sources={r.get('sources')}")
        elif "correctly_refused" in r:
            outcome = "OK  " if r["correctly_refused"] else "FAIL"
            print(f"[{outcome}] [{status}] {r['question']!r} (expected refusal) -> {r.get('answer', '')[:80]!r}")
        else:
            print(f"[????] [{status}] {r['question']!r}")

    print("\n=== Summary ===")
    for key, value in summary.items():
        print(f"{key}: {value}")

    output_path = (
        args.output
        or Path(__file__).parent
        / "results"
        / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"summary": summary, "results": results}, indent=2))
    print(f"\nFull results written to {output_path}")

    if args.min_hit_rate is not None:
        hit_rate = summary.get("retrieval_hit_rate")
        if hit_rate is not None and hit_rate < args.min_hit_rate:
            print(f"\nFAILED: retrieval_hit_rate {hit_rate} is below --min-hit-rate {args.min_hit_rate}")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
