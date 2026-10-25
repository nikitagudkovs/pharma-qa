"""Route a question to the trials tools, label RAG, both, or abstain.

A small, cheap model with structured output and two examples per route.
"""
import time
from typing import Literal

from pydantic import BaseModel

from app.config import ROUTER_MODEL
from app.entities import Entity
from app.llm import CallLog, LLMClient


class Route(BaseModel):
    kind: Literal["trials", "label", "both", "out_of_scope"]
    reason: str
    # The part of the question the drug label must answer, as a standalone search query.
    # Searching labels with the whole question lets trial terms ("Phase 3 lung cancer
    # trials") drag in the wrong label chunks on "both" questions.
    label_query: str | None


SYSTEM = """You route questions for a pharma Q&A service with exactly two evidence sources:
- trials: ClinicalTrials.gov registry records (which trials exist, their phase, status, sponsor, counts of trials).
- label: FDA drug labels (indications, dosing, boxed warnings, contraindications, adverse reactions, interactions, mechanism, pregnancy).

Choose:
- "trials" when the answer comes from trial registry records only.
- "label" when the answer comes from a drug label only.
- "both" when the question asks for both kinds of information.
- "out_of_scope" for anything neither source can answer: revenue, sales, pricing, cost, market share, forecasts,
  future events, or medical advice about a specific person's own treatment.

For "label" and "both", set label_query to the label part of the question rewritten as a standalone search query,
with the drug's generic name and without any trial wording. For "trials" and "out_of_scope", set label_query to null.

Examples:
Q: Which Phase 3 trials of osimertinib are recruiting? -> trials
Q: How many trials have tested rituximab in lymphoma? -> trials
Q: What are the adverse reactions of lenalidomide? -> label (label_query: "lenalidomide adverse reactions")
Q: What is the recommended dose of apixaban for atrial fibrillation? -> label
Q: Is olaparib in any breast cancer trials, and what is its boxed warning? -> both (label_query: "olaparib boxed warning")
Q: What does the label say about nivolumab's mechanism, and which melanoma trials use it? -> both
Q: What were Humira's 2025 sales? -> out_of_scope
Q: My mother takes sertraline; should she increase her dose? -> out_of_scope"""


def route(question: str, entities: list[Entity], llm: LLMClient, log: CallLog) -> Route:
    names = ", ".join(f"{e.mention} -> {e.generic_name}" + ("" if e.resolved else " (not in label corpus)")
                      for e in entities) or "none"
    user = f"Question: {question}\nResolved drugs: {names}"
    t0 = time.perf_counter()
    result, usage = llm.parse(ROUTER_MODEL, SYSTEM, user, Route)
    log.add("router", usage, int((time.perf_counter() - t0) * 1000))
    return result
