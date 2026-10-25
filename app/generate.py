"""Answer generation over evidence the pipeline retrieved, plus the citation
post-check. The model can only cite evidence it was handed, and code, not the
model, decides whether a citation is real.
"""
import re
import time

from pydantic import BaseModel

from app.config import GENERATOR_MODEL
from app.entities import Entity
from app.llm import CallLog, LLMClient


class Evidence(BaseModel):
    id: str          # citation ID, e.g. "trial:NCT01905657" or "label:warfarin:boxed_warning:0"
    source: str      # "trials" | "label"
    title: str
    url: str
    text: str


class DraftAnswer(BaseModel):
    """What the model returns."""
    text: str
    citations: list[str]
    abstained: bool
    abstain_reason: str | None


class Citation(BaseModel):
    id: str
    source: str
    title: str
    url: str


class Answer(BaseModel):
    text: str
    citations: list[Citation]
    abstained: bool
    abstain_reason: str | None
    # How the post-check changed the draft, for traces and the eval
    invented_citations: list[str] = []


SYSTEM = """You answer questions about drugs using ONLY the evidence provided below the question.
Each evidence item has an ID in square brackets.

Rules:
1. Use only facts stated in the evidence. Do not add outside knowledge, even if you are sure of it.
2. Cite every factual sentence by putting the evidence ID in square brackets right after it,
   e.g. "Warfarin can cause major or fatal bleeding [label:warfarin:boxed_warning:0]."
   Only cite IDs that appear in the evidence. List every ID you cited in `citations`.
3. Counts of trials come from count_trials evidence. Report that number exactly; never count list items yourself
   when a count is given.
4. If the evidence does not answer the question, set abstained=true, explain why in abstain_reason
   (e.g. "The label corpus has no label for zuranolone."), and leave text as a one-sentence refusal and citations empty.
   A partial answer is fine if you say what is missing.
5. Never give personal medical advice.
6. Be concise: a short answer, then supporting details. Lists of trials: NCT ID, title, phase, status."""


def _render_evidence(evidence: list[Evidence]) -> str:
    return "\n\n".join(f"[{e.id}] ({e.source}) {e.title}\n{e.text}" for e in evidence)


_MARKER = re.compile(r"\[((?:trial|label|count):[^\]\s]+)\]")


def check_citations(draft: DraftAnswer, evidence: list[Evidence]) -> Answer:
    """Every cited ID, in the list or inline in the text, must exist in the evidence.
    Any invented ID drops the whole answer to abstain."""
    by_id = {e.id: e for e in evidence}
    cited = list(dict.fromkeys([*draft.citations, *_MARKER.findall(draft.text)]))
    invented = [c for c in cited if c not in by_id]
    if draft.abstained:
        return Answer(text=draft.text, citations=[], abstained=True, abstain_reason=draft.abstain_reason)
    if invented:
        return Answer(text="I can't give a verified answer to that from the trial and label data.",
                      citations=[], abstained=True,
                      abstain_reason=f"answer cited evidence that was never retrieved: {', '.join(invented)}",
                      invented_citations=invented)
    if not cited:
        return Answer(text="I can't give a verified answer to that from the trial and label data.",
                      citations=[], abstained=True, abstain_reason="answer had no citations")
    return Answer(text=draft.text, abstained=False, abstain_reason=None,
                  citations=[Citation(id=c, source=by_id[c].source, title=by_id[c].title, url=by_id[c].url)
                             for c in cited])


def generate(question: str, entities: list[Entity], evidence: list[Evidence], llm: LLMClient,
             log: CallLog) -> tuple[DraftAnswer, Answer]:
    unresolved = [e.mention for e in entities if not e.resolved]
    notes = f"\nNote: these drug names could not be resolved and have no label in the corpus: {', '.join(unresolved)}" \
        if unresolved else ""
    resolved = ", ".join(f"{e.mention} = {e.generic_name}" for e in entities if e.resolved) or "none"
    user = f"Question: {question}\nResolved drugs: {resolved}{notes}\n\nEvidence:\n{_render_evidence(evidence)}"
    t0 = time.perf_counter()
    draft, usage = llm.parse(GENERATOR_MODEL, SYSTEM, user, DraftAnswer)
    log.add("generator", usage, int((time.perf_counter() - t0) * 1000))
    return draft, check_citations(draft, evidence)
