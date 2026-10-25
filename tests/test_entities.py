from app.entities import match_synonyms, resolve
from app.llm import CallLog
from tests.fakes import FakeLLM


def names(text):
    return [e.generic_name for e in match_synonyms(text)]


def test_brand_resolves_to_generic():
    assert names("Contraindications for Ozempic?") == ["semaglutide"]
    assert names("Which Phase 3 trials is Keytruda being tested in?") == ["pembrolizumab"]
    assert names("How does Lipitor work?") == ["atorvastatin"]


def test_generic_and_multiword_names():
    assert names("warfarin and apixaban") == ["warfarin", "apixaban"]
    assert names("Is insulin glargine safe in pregnancy?") == ["insulin glargine"]


def test_no_partial_word_matches():
    assert names("NCT02339571 melanoma trial") == []


def test_unknown_drug_is_flagged_not_dropped():
    llm = FakeLLM(parsed={"Mentions": {"drugs": ["zuranolone"]}})
    [e] = resolve("What are the contraindications for zuranolone?", llm, CallLog())
    assert (e.generic_name, e.resolved, e.source) == ("zuranolone", False, "llm_fallback")


def test_llm_fallback_only_when_map_misses():
    llm = FakeLLM()
    resolve("Contraindications for Ozempic?", llm, CallLog())
    assert llm.calls == []
