"""
One-command evaluation.

    python -m eval.run --variants fake
    python -m eval.run --variants cloud,cloud-norerank,local

Each variant is a configuration of the same pipeline. For every case the
harness records retrieval quality, answer quality, latency per stage and cost,
then writes:

    results/raw/<variant>.json   every case, every span — the evidence
    results/EVAL.md              the summary table

Nothing is hand-written into the table. README numbers are checked against
results/raw by scripts/check_readme_numbers.py, so prose and measurement
cannot drift apart.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eval.metrics import (  # noqa: E402
    dedupe,
    groundedness,
    hit_at_k,
    keyword_coverage,
    mean,
    ndcg_at_k,
    percentile,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    refusal,
)
from rag.config import load_settings  # noqa: E402
from rag.pipeline import RagPipeline  # noqa: E402
from rag.tracing import cost_usd  # noqa: E402

# A variant is a named set of overrides on top of config/default.yaml.
VARIANTS: dict[str, dict] = {
    "fake": {"provider": "fake", "qdrant_url": ":memory:"},
    "fake-norerank": {"provider": "fake", "qdrant_url": ":memory:", "rerank": False},
    "cloud": {"provider": "cloud"},
    "cloud-norerank": {"provider": "cloud", "rerank": False},
    "local": {"provider": "local"},
    "local-norerank": {"provider": "local", "rerank": False},
}


def load_cases(path: Path) -> list[dict]:
    cases = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    for case in cases:
        case.setdefault("answerable", True)
        case.setdefault("must_include", [])
        case.setdefault("relevant_docs", [])
    return cases


def evaluate_case(pipeline: RagPipeline, case: dict, k: int, retrieval_only: bool) -> dict:
    relevant = set(case["relevant_docs"])
    started = time.perf_counter()

    if retrieval_only:
        outcome = pipeline.retrieve(case["question"])
        chunks, trace = outcome.chunks, outcome.trace
        answer_text = ""
        cost = cost_usd(
            pipeline.settings,
            embed_tokens=outcome.embed_tokens,
            tokens_in=outcome.rerank_tokens_in,
            tokens_out=outcome.rerank_tokens_out,
        )
    else:
        answer = pipeline.answer(case["question"])
        answer_text = answer.text
        cost = answer.cost_usd
        trace = None
        chunks = []

    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)

    if retrieval_only:
        chunk_docs = [c.chunk.doc_id for c in chunks]
        spans = trace.as_dict()["spans"]
    else:
        chunk_docs = [s["doc_id"] for s in answer.sources]
        spans = answer.trace["spans"]

    # Metrics are document-level; the raw ranking is chunk-level and repeats
    # documents. See eval.metrics.dedupe.
    docs = dedupe(chunk_docs)

    record = {
        "id": case["id"],
        "question": case["question"],
        "answerable": case["answerable"],
        "retrieved_chunk_docs": chunk_docs,
        "retrieved_docs": docs,
        "latency_ms": elapsed_ms,
        "cost_usd": cost,
        "spans": spans,
    }

    if case["answerable"] and relevant:
        record["retrieval"] = {
            f"hit@{k}": hit_at_k(docs, relevant, k),
            f"recall@{k}": recall_at_k(docs, relevant, k),
            f"precision@{k}": precision_at_k(docs, relevant, k),
            f"ndcg@{k}": ndcg_at_k(docs, relevant, k),
            "mrr": reciprocal_rank(docs, relevant),
        }

    if not retrieval_only:
        record["answer"] = answer_text
        if case["answerable"]:
            record["quality"] = {
                "keyword_coverage": keyword_coverage(answer_text, case["must_include"]),
                "groundedness": groundedness(
                    answer_text, "\n".join(s.get("text", "") for s in answer.sources)
                ),
            }
        else:
            record["quality"] = {"refused": 1.0 if refusal(answer_text) else 0.0}

    return record


def summarise(records: list[dict], k: int, retrieval_only: bool) -> dict:
    answerable = [r for r in records if r["answerable"] and "retrieval" in r]
    unanswerable = [r for r in records if not r["answerable"]]
    latencies = [r["latency_ms"] for r in records]

    summary = {
        "cases": len(records),
        f"hit@{k}": round(mean([r["retrieval"][f"hit@{k}"] for r in answerable]), 4),
        f"recall@{k}": round(mean([r["retrieval"][f"recall@{k}"] for r in answerable]), 4),
        f"precision@{k}": round(mean([r["retrieval"][f"precision@{k}"] for r in answerable]), 4),
        f"ndcg@{k}": round(mean([r["retrieval"][f"ndcg@{k}"] for r in answerable]), 4),
        "mrr": round(mean([r["retrieval"]["mrr"] for r in answerable]), 4),
        "latency_p50_ms": round(percentile(latencies, 50), 1),
        "latency_p95_ms": round(percentile(latencies, 95), 1),
        "cost_per_query_usd": round(mean([r["cost_usd"] for r in records]), 6),
    }

    if not retrieval_only:
        scored = [r for r in records if r["answerable"] and "quality" in r]
        summary["keyword_coverage"] = round(
            mean([r["quality"]["keyword_coverage"] for r in scored]), 4
        )
        summary["groundedness"] = round(mean([r["quality"]["groundedness"] for r in scored]), 4)
        if unanswerable:
            summary["refusal_rate"] = round(
                mean([r["quality"]["refused"] for r in unanswerable]), 4
            )

    # Per-stage median latency: which step actually costs the time.
    stage_times: dict[str, list[float]] = {}
    for record in records:
        for span in record["spans"]:
            stage_times.setdefault(span["name"], []).append(span["duration_ms"])
    summary["stage_p50_ms"] = {
        name: round(percentile(values, 50), 1) for name, values in stage_times.items()
    }
    return summary


def render(results: dict[str, dict], k: int) -> str:
    lines = [
        "# Evaluation results",
        "",
        f"Generated by `python -m eval.run`. {len(results)} variant(s), "
        f"{next(iter(results.values()))['summary']['cases']} cases.",
        "",
        "## Retrieval",
        "",
        f"| Variant | hit@{k} ↑ | recall@{k} ↑ | precision@{k} ↑ | nDCG@{k} ↑ | MRR ↑ |",
        "|---|---|---|---|---|---|",
    ]
    for name, payload in results.items():
        s = payload["summary"]
        lines.append(
            f"| {name} | {s[f'hit@{k}']:.3f} | {s[f'recall@{k}']:.3f} | "
            f"{s[f'precision@{k}']:.3f} | {s[f'ndcg@{k}']:.3f} | {s['mrr']:.3f} |"
        )

    lines += ["", "## Answers, latency and cost", "",
              "| Variant | keyword coverage ↑ | groundedness ↑ | refusal rate ↑ | "
              "p50 ms ↓ | p95 ms ↓ | USD / query ↓ |", "|---|---|---|---|---|---|---|"]
    for name, payload in results.items():
        s = payload["summary"]
        coverage = f"{s['keyword_coverage']:.3f}" if "keyword_coverage" in s else "—"
        grounded = f"{s['groundedness']:.3f}" if "groundedness" in s else "—"
        refused = f"{s['refusal_rate']:.3f}" if "refusal_rate" in s else "—"
        lines.append(
            f"| {name} | {coverage} | {grounded} | {refused} | "
            f"{s['latency_p50_ms']:.1f} | {s['latency_p95_ms']:.1f} | "
            f"{s['cost_per_query_usd']:.6f} |"
        )

    lines += ["", "## Where the time goes (median per stage, ms)", ""]
    stages = sorted({stage for p in results.values() for stage in p["summary"]["stage_p50_ms"]})
    lines.append("| Variant | " + " | ".join(stages) + " |")
    lines.append("|---" * (len(stages) + 1) + "|")
    for name, payload in results.items():
        row = payload["summary"]["stage_p50_ms"]
        lines.append(
            f"| {name} | " + " | ".join(f"{row.get(stage, 0.0):.1f}" for stage in stages) + " |"
        )

    lines += [
        "",
        "Refusal rate is measured on the cases marked `answerable: false` in",
        "`eval/cases.yaml`: higher is better, because the corpus genuinely does",
        "not contain those answers. Cost for local models is API spend, which is",
        "zero; it says nothing about the hardware the model runs on.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline")
    parser.add_argument("--variants", default="fake", help=f"comma-separated: {', '.join(VARIANTS)}")
    parser.add_argument("--cases", default=str(ROOT / "eval" / "cases.yaml"))
    parser.add_argument("--corpus", default=str(ROOT / "data" / "sample"))
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--retrieval-only", action="store_true", help="skip generation")
    parser.add_argument("--out-dir", default=str(ROOT / "results"))
    args = parser.parse_args()

    cases = load_cases(Path(args.cases))
    out_dir = Path(args.out_dir)
    (out_dir / "raw").mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    for name in [v.strip() for v in args.variants.split(",") if v.strip()]:
        if name not in VARIANTS:
            raise SystemExit(f"unknown variant {name!r}; known: {', '.join(VARIANTS)}")
        print(f"=== {name} ===", flush=True)

        settings = load_settings(**VARIANTS[name])
        pipeline = RagPipeline(settings)
        indexed = pipeline.ingest_path(args.corpus)
        print(f"indexed {indexed} chunks from {args.corpus}", flush=True)

        records = [
            evaluate_case(pipeline, case, args.k, args.retrieval_only) for case in cases
        ]
        summary = summarise(records, args.k, args.retrieval_only)
        payload = {
            "variant": name,
            "config": {
                "provider": settings.provider,
                "embed_model": settings.embed_model,
                "chat_model": settings.chat_model,
                "rerank": settings.rerank,
                "reranker": pipeline.reranker.name,
                "chunk_size": settings.chunk_size,
                "dense_top_k": settings.dense_top_k,
                "sparse_top_k": settings.sparse_top_k,
                "rerank_top_n": settings.rerank_top_n,
                "retrieval_only": args.retrieval_only,
            },
            "summary": summary,
            "cases": records,
        }
        (out_dir / "raw" / f"{name}.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        results[name] = payload
        print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)

    report = render(results, args.k)
    (out_dir / "EVAL.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
