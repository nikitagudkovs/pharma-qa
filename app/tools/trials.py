"""ClinicalTrials.gov API v2 client and the typed tools the LLM can call.

Trials are structured data, so they are reached through tools, not RAG.
Every argument is validated by Pydantic before any request goes out, and
counting happens here in code, never in the model.
"""
import time
from enum import Enum

import httpx
from pydantic import BaseModel, Field, field_validator

from app.cache import DiskCache
from app.config import CACHE_DIR

BASE_URL = "https://clinicaltrials.gov/api/v2/studies"
SUMMARY_FIELDS = ",".join([
    "NCTId", "BriefTitle", "Phase", "OverallStatus", "LeadSponsorName",
    "Condition", "InterventionName", "StartDate", "EnrollmentCount",
])
DETAIL_FIELDS = SUMMARY_FIELDS + ",OfficialTitle,BriefSummary,PrimaryOutcomeMeasure,PrimaryCompletionDate"

_cache = DiskCache(CACHE_DIR / "trials")


class Phase(str, Enum):
    PHASE1 = "PHASE1"
    PHASE2 = "PHASE2"
    PHASE3 = "PHASE3"
    PHASE4 = "PHASE4"


class Status(str, Enum):
    NOT_YET_RECRUITING = "NOT_YET_RECRUITING"
    RECRUITING = "RECRUITING"
    ENROLLING_BY_INVITATION = "ENROLLING_BY_INVITATION"
    ACTIVE_NOT_RECRUITING = "ACTIVE_NOT_RECRUITING"
    COMPLETED = "COMPLETED"
    SUSPENDED = "SUSPENDED"
    TERMINATED = "TERMINATED"
    WITHDRAWN = "WITHDRAWN"


class TrialSummary(BaseModel):
    nct_id: str
    title: str
    phase: list[str]
    status: str
    sponsor: str | None
    conditions: list[str]
    interventions: list[str] = []
    start_date: str | None = None
    enrollment: int | None = None
    url: str


class TrialDetail(TrialSummary):
    official_title: str | None = None
    summary: str | None = None
    primary_outcomes: list[str] = []
    primary_completion_date: str | None = None


class SearchTrialsArgs(BaseModel):
    drug: str | None = Field(None, description="Generic drug name, e.g. pembrolizumab")
    condition: str | None = Field(None, description="Condition or disease, e.g. non-small cell lung cancer")
    phase: list[Phase] | None = None
    status: list[Status] | None = None
    limit: int = Field(10, ge=1, le=50)

    @field_validator("drug", "condition")
    @classmethod
    def _strip(cls, v: str | None) -> str | None:
        v = (v or "").strip()
        return v or None


class CountTrialsArgs(BaseModel):
    drug: str | None = None
    condition: str | None = None
    phase: list[Phase] | None = None
    status: list[Status] | None = None


class GetTrialArgs(BaseModel):
    nct_id: str = Field(pattern=r"^NCT\d{8}$")


class TrialsAPIError(RuntimeError):
    pass


def _get(url: str, params: dict) -> dict:
    cached = _cache.get(url, params)
    if cached is not None:
        return cached
    for attempt in range(4):
        resp = httpx.get(url, params=params, timeout=20)
        if resp.status_code == 429:
            time.sleep(2 ** attempt * 2)
            continue
        if resp.status_code >= 400:
            raise TrialsAPIError(f"ClinicalTrials.gov {resp.status_code}: {resp.text[:200]}")
        data = resp.json()
        _cache.put(url, params, data)
        return data
    raise TrialsAPIError("ClinicalTrials.gov rate limit: retries exhausted")


def _query_params(drug, condition, phase, status) -> dict:
    if not (drug or condition):
        raise ValueError("search needs a drug or a condition")
    params: dict = {}
    if drug:
        params["query.intr"] = drug
    if condition:
        params["query.cond"] = condition
    if phase:
        # v2 has no filter.phase; phases go through aggFilters, space-separated = OR
        params["aggFilters"] = "phase:" + " ".join(p.value[-1] for p in phase)
    if status:
        params["filter.overallStatus"] = ",".join(s.value for s in status)
    return params


def _summary(study: dict, cls=TrialSummary, **extra) -> TrialSummary:
    ps = study["protocolSection"]
    ident = ps.get("identificationModule", {})
    status = ps.get("statusModule", {})
    nct = ident["nctId"]
    return cls(
        nct_id=nct,
        title=ident.get("briefTitle", ""),
        phase=ps.get("designModule", {}).get("phases", []),
        status=status.get("overallStatus", "UNKNOWN"),
        sponsor=ps.get("sponsorCollaboratorsModule", {}).get("leadSponsor", {}).get("name"),
        conditions=ps.get("conditionsModule", {}).get("conditions", []),
        interventions=[i.get("name", "") for i in ps.get("armsInterventionsModule", {}).get("interventions", [])],
        start_date=status.get("startDateStruct", {}).get("date"),
        enrollment=ps.get("designModule", {}).get("enrollmentInfo", {}).get("count"),
        url=f"https://clinicaltrials.gov/study/{nct}",
        **extra,
    )


def search_trials(args: SearchTrialsArgs) -> list[TrialSummary]:
    params = _query_params(args.drug, args.condition, args.phase, args.status)
    params.update({"pageSize": args.limit, "fields": SUMMARY_FIELDS})
    data = _get(BASE_URL, params)
    return [_summary(s) for s in data.get("studies", [])]


def count_trials(args: CountTrialsArgs) -> int:
    params = _query_params(args.drug, args.condition, args.phase, args.status)
    params.update({"pageSize": 1, "countTotal": "true", "fields": "NCTId"})
    return int(_get(BASE_URL, params).get("totalCount", 0))


def get_trial(args: GetTrialArgs) -> TrialDetail:
    data = _get(f"{BASE_URL}/{args.nct_id}", {"fields": DETAIL_FIELDS})
    ps = data["protocolSection"]
    return _summary(
        data,
        cls=TrialDetail,
        official_title=ps.get("identificationModule", {}).get("officialTitle"),
        summary=ps.get("descriptionModule", {}).get("briefSummary"),
        primary_outcomes=[o.get("measure", "") for o in ps.get("outcomesModule", {}).get("primaryOutcomes", [])],
        primary_completion_date=ps.get("statusModule", {}).get("primaryCompletionDateStruct", {}).get("date"),
    )


TOOLS = {
    "search_trials": (SearchTrialsArgs, search_trials,
                      "Search ClinicalTrials.gov for trials of a drug and/or condition, optionally filtered by phase "
                      "and overall status. Returns up to `limit` trial summaries with NCT IDs."),
    "count_trials": (CountTrialsArgs, count_trials,
                     "Count ALL ClinicalTrials.gov trials matching a drug and/or condition, phase and status. "
                     "Use this for any 'how many' question; never count search results yourself."),
    "get_trial": (GetTrialArgs, get_trial,
                  "Fetch one trial's details (summary, primary outcomes, dates) by NCT ID."),
}


def tool_schemas() -> list[dict]:
    """Anthropic tool definitions generated from the Pydantic argument models."""
    out = []
    for name, (model, _, desc) in TOOLS.items():
        schema = model.model_json_schema()
        out.append({"name": name, "description": desc, "input_schema": schema})
    return out
