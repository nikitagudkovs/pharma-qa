"""resolve -> route -> retrieve / call tools -> generate -> check."""
import time
import uuid
from datetime import date

from pydantic import BaseModel, ValidationError

from app.config import GENERATOR_MODEL
from app.entities import Entity, resolve
from app.generate import Answer, DraftAnswer, Evidence, generate
from app.llm import AnthropicClient, CallLog, LLMClient, LLMRefusal
from app.rag.search import get_index
from app.router import Route, route
from app.tools.trials import TOOLS, CountTrialsArgs, TrialDetail, TrialSummary, TrialsAPIError, tool_schemas
from app.trace import write_trace

MAX_TOOL_CALLS = 3
LABEL_K = 5

PLANNER_SYSTEM = f"""You turn a question into ClinicalTrials.gov tool calls. Today is {date.today().isoformat()}.
- Call count_trials for any "how many" question. Call search_trials to list trials. Call get_trial for a specific NCT ID.
- Use the resolved generic drug name, not the brand name.
- Keep the condition as the user phrased it (e.g. "lung cancer"); don't narrow it to a subtype.
- Only add phase or status filters the question asks for. "Currently recruiting" means status RECRUITING.
- Make at most {MAX_TOOL_CALLS} calls, all in this one turn. Do not answer the question yourself."""


class ToolRun(BaseModel):
    name: str
    input: dict
    ok: bool
    error: str | None = None
    result_ids: list[str] = []


class AskResult(BaseModel):
    question: str
    answer: str
    citations: list[dict]
    route: str
    route_reason: str
    label_query: str | None
    entities: list[Entity]
    abstained: bool
    abstain_reason: str | None
    evidence_ids: list[str]
    tool_runs: list[ToolRun]
    raw_citations: list[str]
    invented_citations: list[str]
    trace_id: str
    latency_ms: int
    cost_usd: float


def _trial_evidence(t: TrialSummary) -> Evidence:
    text = (f"Phase: {', '.join(t.phase) or 'n/a'}. Status: {t.status}. Sponsor: {t.sponsor}. "
            f"Conditions: {', '.join(t.conditions)}. Interventions: {', '.join(t.interventions[:6])}. "
            f"Start: {t.start_date}. Enrollment: {t.enrollment}.")
    if isinstance(t, TrialDetail):
        text += (f" Official title: {t.official_title}. Primary outcomes: {'; '.join(t.primary_outcomes)}. "
                 f"Primary completion: {t.primary_completion_date}. Summary: {t.summary}")
    return Evidence(id=f"trial:{t.nct_id}", source="trials", title=t.title, url=t.url, text=text)


def run_trial_tools(question: str, entities: list[Entity], llm: LLMClient,
                    log: CallLog) -> tuple[list[Evidence], list[ToolRun]]:
    drugs = ", ".join(f"{e.mention} -> {e.generic_name}" for e in entities) or "none"
    t0 = time.perf_counter()
    calls, usage = llm.tool_calls(GENERATOR_MODEL, PLANNER_SYSTEM,
                                  f"Question: {question}\nResolved drugs: {drugs}", tool_schemas())
    log.add("trial_planner", usage, int((time.perf_counter() - t0) * 1000))

    evidence: list[Evidence] = []
    runs: list[ToolRun] = []
    for i, call in enumerate(calls[:MAX_TOOL_CALLS], start=1):
        if call.name not in TOOLS:
            runs.append(ToolRun(name=call.name, input=call.input, ok=False, error="unknown tool"))
            continue
        args_model, fn, _ = TOOLS[call.name]
        try:
            args = args_model.model_validate(call.input)  # validate before any request goes out
            result = fn(args)
        except (ValidationError, ValueError, TrialsAPIError) as e:
            runs.append(ToolRun(name=call.name, input=call.input, ok=False, error=str(e)[:300]))
            continue
        if call.name == "count_trials":
            a: CountTrialsArgs = args
            desc = ", ".join(f"{k}={v}" for k, v in {
                "drug": a.drug, "condition": a.condition,
                "phase": [p.value for p in a.phase] if a.phase else None,
                "status": [s.value for s in a.status] if a.status else None}.items() if v)
            ev = [Evidence(id=f"count:{i}", source="trials", title=f"count_trials({desc})",
                           url="https://clinicaltrials.gov/search",
                           text=f"Total number of ClinicalTrials.gov studies matching {desc}: {result} "
                                f"(computed in code from the registry's totalCount).")]
        elif call.name == "get_trial":
            ev = [_trial_evidence(result)]
        else:
            ev = [_trial_evidence(t) for t in result]
        evidence.extend(ev)
        runs.append(ToolRun(name=call.name, input=args.model_dump(mode="json", exclude_none=True), ok=True,
                            result_ids=[e.id for e in ev]))
    return evidence, runs


def label_evidence(query: str, entities: list[Entity]) -> list[Evidence]:
    # Filter by every drug the question names. An unresolved drug has no label,
    # so it contributes no chunks rather than letting other drugs' chunks leak in.
    drugs = [e.generic_name for e in entities]
    hits = get_index().search(query, drugs=drugs or None, k=LABEL_K)
    return [Evidence(id=h.chunk.id, source="label",
                     title=f"{h.chunk.drug.title()} label — {h.chunk.section_name} (effective {h.chunk.effective_time})",
                     url=h.chunk.url, text=h.chunk.text) for h in hits]


def ask(question: str, llm: LLMClient | None = None) -> AskResult:
    llm = llm or AnthropicClient()
    t0 = time.perf_counter()
    log = CallLog()
    trace_id = uuid.uuid4().hex[:12]
    entities: list[Entity] = []
    rt = Route(kind="out_of_scope", reason="not routed", label_query=None)
    evidence: list[Evidence] = []
    runs: list[ToolRun] = []
    draft: DraftAnswer | None = None

    try:
        entities = resolve(question, llm, log)
        rt = route(question, entities, llm, log)
        if rt.kind == "out_of_scope":
            answer = Answer(text="I can't answer that from trial or label data.", citations=[], abstained=True,
                            abstain_reason=f"out of scope: {rt.reason}")
        else:
            if rt.kind in ("trials", "both"):
                ev, runs = run_trial_tools(question, entities, llm, log)
                evidence += ev
            if rt.kind in ("label", "both"):
                evidence += label_evidence(rt.label_query or question, entities)
            if not evidence:
                missing = [e.mention for e in entities if not e.resolved]
                answer = Answer(text="I can't answer that from trial or label data.", citations=[], abstained=True,
                                abstain_reason=("no label in the corpus for " + ", ".join(missing)) if missing
                                else "no evidence retrieved")
            else:
                draft, answer = generate(question, entities, evidence, llm, log)
    except LLMRefusal as e:
        answer = Answer(text="I can't answer that.", citations=[], abstained=True,
                        abstain_reason=f"model refused ({e})")

    result = AskResult(
        question=question, answer=answer.text, citations=[c.model_dump() for c in answer.citations],
        route=rt.kind, route_reason=rt.reason, label_query=rt.label_query, entities=entities, abstained=answer.abstained,
        abstain_reason=answer.abstain_reason, evidence_ids=[e.id for e in evidence], tool_runs=runs,
        raw_citations=draft.citations if draft else [], invented_citations=answer.invented_citations,
        trace_id=trace_id, latency_ms=int((time.perf_counter() - t0) * 1000), cost_usd=round(log.cost_usd, 6),
    )
    write_trace(trace_id, result, log, draft)
    return result
