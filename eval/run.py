"""Run the 30-question eval and print a scorecard.

    uv run python -m eval.run --tag baseline
    uv run python -m eval.run --tag after-fix --only q21,q22

Every metric is computed in code from the pipeline's structured output. Each run
is saved to eval/results/<timestamp>-<tag>.json so runs can be compared.
"""
import argparse
import json
import shutil
import statistics
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from app.config import GENERATOR_MODEL, ROOT, ROUTER_MODEL, USE_RERANKER
from app.pipeline import AskResult, ask
from app.trace import trace_path

QUESTIONS = ROOT / "eval" / "questions.jsonl"
RESULTS = ROOT / "eval" / "results"
SAVED_TRACES = ROOT / "eval" / "traces"  # traces for saved runs are kept in the repo
TARGETS = {"routing": 0.90, "entities": 0.95, "recall@5": 0.85, "citation_validity": 1.0, "correctness": 0.80,
           "abstain_recall": 1.0}


def load_questions() -> list[dict]:
    return [json.loads(line) for line in QUESTIONS.read_text().splitlines() if line.strip()]


def _contains(text: str, groups: list[list[str]]) -> bool:
    low = text.lower()
    return all(any(alt.lower() in low for alt in group) for group in groups)


def recall_at_5(q: dict, r: AskResult) -> bool | None:
    """Each expected source type (trial / count / label) must have a hit in the top 5 of that type."""
    if not q["expected_sources"]:
        return None
    for kind in ("trial:", "count:", "label:"):
        expected = [s for s in q["expected_sources"] if s.startswith(kind)]
        if not expected:
            continue
        top5 = [e for e in r.evidence_ids if e.startswith(kind)][:5]
        if not any(e.startswith(x) for e in top5 for x in expected):
            return False
    return True


def score(q: dict, r: AskResult) -> dict:
    entities = sorted(e.generic_name for e in r.entities)
    groups = q["must_contain"] + q.get("must_contain_any", [])
    correct = (r.abstained if q["should_abstain"]
               else (not r.abstained) and _contains(r.answer, groups))
    evidence = set(r.evidence_ids)
    return {
        "id": q["id"], "type": q["type"],
        "route_ok": r.route == q["expected_route"],
        "entities_ok": entities == sorted(q["expected_entities"]),
        "recall_ok": recall_at_5(q, r),
        # Did the model's own draft cite only real evidence? (The post-check enforces it either way.)
        "raw_citations_valid": all(c in evidence for c in r.raw_citations) if r.raw_citations else None,
        "final_citations_valid": all(c["id"] in evidence for c in r.citations),
        "correct": correct,
        "abstained": r.abstained, "should_abstain": q["should_abstain"],
        "latency_ms": r.latency_ms, "cost_usd": r.cost_usd,
        "route": r.route, "entities": entities, "answer": r.answer, "abstain_reason": r.abstain_reason,
        "trace": _keep_trace(r.trace_id),
    }


def _keep_trace(trace_id: str) -> str:
    SAVED_TRACES.mkdir(parents=True, exist_ok=True)
    dest = SAVED_TRACES / f"{trace_id}.json"
    shutil.copy(trace_path(trace_id), dest)
    return str(dest.relative_to(ROOT))


def _rate(rows: list[dict], key: str) -> float | None:
    vals = [row[key] for row in rows if row[key] is not None]
    return sum(vals) / len(vals) if vals else None


def metrics(rows: list[dict]) -> dict:
    abstained = [r for r in rows if r["abstained"]]
    should = [r for r in rows if r["should_abstain"]]
    tp = sum(1 for r in abstained if r["should_abstain"])
    return {
        "routing": _rate(rows, "route_ok"),
        "entities": _rate(rows, "entities_ok"),
        "recall@5": _rate(rows, "recall_ok"),
        "citation_validity_raw": _rate(rows, "raw_citations_valid"),
        "citation_validity": _rate(rows, "final_citations_valid"),
        "correctness": _rate([r for r in rows if not r["should_abstain"]], "correct"),
        "abstain_precision": tp / len(abstained) if abstained else None,
        "abstain_recall": tp / len(should) if should else None,
        "p50_latency_ms": statistics.median(r["latency_ms"] for r in rows),
        "mean_cost_usd": sum(r["cost_usd"] for r in rows) / len(rows),
        "n": len(rows),
    }


def by_type(rows: list[dict]) -> dict:
    out = {}
    for t in dict.fromkeys(r["type"] for r in rows):
        sub = [r for r in rows if r["type"] == t]
        out[t] = {"n": len(sub), "correct": _rate(sub, "correct"), "routing": _rate(sub, "route_ok"),
                  "entities": _rate(sub, "entities_ok"), "recall@5": _rate(sub, "recall_ok")}
    return out


def _fmt(v, pct=True) -> str:
    if v is None:
        return "  n/a"
    return f"{v * 100:5.1f}%" if pct else f"{v:.4f}"


def print_scorecard(m: dict, bt: dict, rows: list[dict]) -> None:
    print("\nMetric                     Score   Target")
    print("-" * 42)
    labels = [("routing", "Routing accuracy"), ("entities", "Entity accuracy"), ("recall@5", "Retrieval recall@5"),
              ("citation_validity_raw", "Citation validity (draft)"), ("citation_validity", "Citation validity (final)"),
              ("correctness", "Answer correctness"), ("abstain_precision", "Abstain precision"),
              ("abstain_recall", "Abstain recall")]
    for key, label in labels:
        target = TARGETS.get(key)
        print(f"{label:26s} {_fmt(m[key])}   {'' if target is None else f'≥ {target * 100:.0f}%'}")
    print(f"{'p50 latency':26s} {m['p50_latency_ms'] / 1000:5.1f}s")
    print(f"{'Mean cost per question':26s} ${m['mean_cost_usd']:.4f}")

    print("\nBy type            n   correct  routing  entities  recall@5")
    for t, s in bt.items():
        print(f"{t:16s} {s['n']:3d}   {_fmt(s['correct'])}   {_fmt(s['routing'])}   {_fmt(s['entities'])}    "
              f"{_fmt(s['recall@5'])}")

    failed = [r for r in rows if not (r["correct"] and r["route_ok"] and r["entities_ok"] and r["recall_ok"] is not False)]
    print(f"\nFailed ({len(failed)}):")
    for r in failed:
        why = [k for k, ok in (("answer", r["correct"]), ("route", r["route_ok"]), ("entities", r["entities_ok"]),
                               ("recall", r["recall_ok"] is not False)) if not ok]
        print(f"  {r['id']}  [{', '.join(why)}]  {r['trace']}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="run")
    p.add_argument("--only", help="comma-separated question ids")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    qs = load_questions()
    if args.only:
        keep = set(args.only.split(","))
        qs = [q for q in qs if q["id"] in keep]
    started = datetime.now(timezone.utc)
    print(f"Running {len(qs)} questions (router={ROUTER_MODEL}, generator={GENERATOR_MODEL}, rerank={USE_RERANKER})")
    with ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(lambda q: ask(q["question"]), qs))
    rows = [score(q, r) for q, r in zip(qs, results)]
    m, bt = metrics(rows), by_type(rows)
    print_scorecard(m, bt, rows)

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"{started.strftime('%Y%m%dT%H%M%S')}-{args.tag}.json"
    out.write_text(json.dumps({
        "tag": args.tag, "started_at": started.isoformat(),
        "config": {"router": ROUTER_MODEL, "generator": GENERATOR_MODEL, "rerank": USE_RERANKER,
                   "questions": len(qs)},
        "metrics": m, "by_type": bt, "rows": rows,
    }, indent=1))
    print(f"\nSaved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
