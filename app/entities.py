"""Entity resolution: map drug mentions to a canonical generic name and RxCUI.

Step 1 is a deterministic lookup in the synonym map built from label metadata.
Step 2, only when nothing matched, asks the LLM to extract mentions and looks
those up. Unknown drugs are kept and flagged, never dropped.
"""
import json
import re
import time
from functools import lru_cache

from pydantic import BaseModel

from app.config import ROUTER_MODEL, SYNONYMS_PATH
from app.llm import CallLog, LLMClient


class Entity(BaseModel):
    mention: str
    generic_name: str
    rxcui: list[str] = []
    resolved: bool = True
    source: str = "synonym_map"  # or "llm_fallback"


class Mentions(BaseModel):
    drugs: list[str]


EXTRACT_SYSTEM = (
    "Extract every drug or medicine name mentioned in the user's question, exactly as written. "
    "Include brand names, generic names and misspellings. Do not include diseases, genes or trial IDs. "
    "Return an empty list if there are none."
)


@lru_cache(maxsize=1)
def synonym_lookup() -> dict[str, tuple[str, list[str]]]:
    """lowercase name -> (generic, rxcui) for generics, label name variants and brands."""
    data = json.loads(SYNONYMS_PATH.read_text())
    out: dict[str, tuple[str, list[str]]] = {}
    for generic, info in data.items():
        for name in [generic, *info["variants"], *info["brands"]]:
            out.setdefault(name.lower(), (generic, info["rxcui"]))
    return out


@lru_cache(maxsize=1)
def _pattern() -> re.Pattern:
    # Longest names first so "insulin glargine" wins over a shorter overlap
    names = sorted(synonym_lookup(), key=len, reverse=True)
    return re.compile(r"(?<![a-z0-9])(" + "|".join(re.escape(n) for n in names) + r")(?![a-z0-9])")


def match_synonyms(text: str) -> list[Entity]:
    lookup = synonym_lookup()
    found: dict[str, Entity] = {}
    for m in _pattern().finditer(text.lower()):
        generic, rxcui = lookup[m.group(1)]
        found.setdefault(generic, Entity(mention=text[m.start():m.end()], generic_name=generic, rxcui=rxcui))
    return list(found.values())


def resolve(question: str, llm: LLMClient, log: CallLog) -> list[Entity]:
    entities = match_synonyms(question)
    if entities:
        return entities
    t0 = time.perf_counter()
    mentions, usage = llm.parse(ROUTER_MODEL, EXTRACT_SYSTEM, question, Mentions)
    log.add("entity_fallback", usage, int((time.perf_counter() - t0) * 1000))
    out: list[Entity] = []
    for mention in mentions.drugs:
        hits = match_synonyms(mention)
        if hits:
            out.extend(h.model_copy(update={"mention": mention, "source": "llm_fallback"}) for h in hits)
        else:
            out.append(Entity(mention=mention, generic_name=mention.lower().strip(), resolved=False,
                              source="llm_fallback"))
    return out
