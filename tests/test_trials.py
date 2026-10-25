import pytest
from pydantic import ValidationError

from app.tools.trials import CountTrialsArgs, GetTrialArgs, Phase, SearchTrialsArgs, Status, _query_params, count_trials


def test_phase_and_status_map_to_v2_params():
    p = _query_params("nivolumab", "melanoma", [Phase.PHASE2, Phase.PHASE3], [Status.RECRUITING])
    assert p == {"query.intr": "nivolumab", "query.cond": "melanoma", "aggFilters": "phase:2 3",
                 "filter.overallStatus": "RECRUITING"}


def test_search_needs_drug_or_condition():
    with pytest.raises(ValueError):
        _query_params(None, None, None, None)


def test_arguments_are_validated_before_any_request():
    with pytest.raises(ValidationError):
        GetTrialArgs(nct_id="NCT123")
    with pytest.raises(ValidationError):
        SearchTrialsArgs(drug="x", phase=["PHASE9"])
    with pytest.raises(ValidationError):
        SearchTrialsArgs(drug="x", limit=500)


def test_count_comes_from_registry_total():
    # Served from the committed disk cache, so no network
    assert count_trials(CountTrialsArgs(drug="nivolumab", condition="melanoma", phase=[Phase.PHASE2])) == 207
