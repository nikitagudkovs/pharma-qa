"""One JSON trace per request: enough to debug any eval failure."""
import json
from datetime import datetime, timezone
from pathlib import Path

from app.config import TRACE_DIR


def trace_path(trace_id: str) -> Path:
    return TRACE_DIR / f"{trace_id}.json"


def write_trace(trace_id: str, result, log, draft) -> Path:
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    path = trace_path(trace_id)
    path.write_text(json.dumps({
        "trace_id": trace_id,
        "at": datetime.now(timezone.utc).isoformat(),
        "question": result.question,
        "entities": [e.model_dump() for e in result.entities],
        "route": {"kind": result.route, "reason": result.route_reason, "label_query": result.label_query},
        "tool_calls": [r.model_dump() for r in result.tool_runs],
        "retrieved": result.evidence_ids,
        "draft": draft.model_dump() if draft else None,
        "answer": {"text": result.answer, "abstained": result.abstained, "abstain_reason": result.abstain_reason,
                   "citations": [c["id"] for c in result.citations], "invented": result.invented_citations},
        "llm_calls": log.calls,
        "latency_ms": result.latency_ms,
        "cost_usd": result.cost_usd,
    }, indent=1))
    return path
