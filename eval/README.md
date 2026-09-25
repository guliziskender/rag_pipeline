# Eval harness

Measures retrieval and groundedness quality against a running instance of the
API — a live-system check, not a unit test. It is **not** run in CI: it needs
a real `ANTHROPIC_API_KEY` on the server and makes real (billed) LLM calls.

## What it measures

- **Retrieval hit rate** — for each question with a known `expected_source`,
  did that document actually come back in the answer's cited sources? This
  directly measures whether hybrid retrieval + reranking are working, and is
  the metric to watch when tuning chunk size, retrieval `k`, or rerank
  weights.
- **Refusal accuracy** — for questions with `expected_source: null`
  (deliberately out-of-domain, not covered by any indexed document), did the
  system correctly decline rather than answer from the LLM's own general
  knowledge? This exercises the groundedness guardrail in `app.py` directly.

## What it does *not* measure

Answer correctness, fluency, or faithfulness to the retrieved context. Those
need either human review or an LLM-judge-based metric (e.g.
[RAGAS](https://github.com/explodinggraphs/ragas)). That's deliberately left
out of this harness for now — it's a heavier, more fragile dependency chain,
and worth adding once the two structural metrics above are being tracked and
you have a concrete need for a finer-grained quality signal.

## Running it

```bash
uvicorn app:app &   # the real app, with real ANTHROPIC_API_KEY / RAG_API_KEY set

python eval/run_eval.py \
    --api-url http://localhost:8000 \
    --api-key "$RAG_API_KEY" \
    --index-dir eval/fixtures    # only needed once per fresh Chroma directory
```

Results print to stdout and are also written as JSON to `eval/results/`
(gitignored — it's run output, not source).

Use `--min-hit-rate 0.8` (or similar) to make the script exit non-zero when
retrieval quality regresses below a threshold — useful for wiring this into a
manual or scheduled check once you're tracking it over time.

## The bundled dataset

`eval/dataset.jsonl` ships with 5 synthetic questions against two throwaway
fixture PDFs (the same content used in `tests/fixtures/`), just to prove the
harness runs end-to-end. **This is a demo, not a real eval set** — it says
nothing about retrieval quality on your actual documents. Replace it with
real questions against your real corpus, with `expected_source` set to the
filename you actually expect to be cited, to get a signal worth acting on.
