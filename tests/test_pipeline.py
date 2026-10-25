from app.llm import ToolCall
from app.pipeline import ask
from tests.fakes import FakeLLM


def test_out_of_scope_abstains_without_generating():
    llm = FakeLLM(parsed={"Route": {"kind": "out_of_scope", "reason": "revenue forecast", "label_query": None}})
    r = ask("What will Keytruda earn in 2030?", llm)
    assert r.abstained and r.route == "out_of_scope"
    assert "DraftAnswer" not in llm.calls


def test_drug_not_in_corpus_abstains_in_code():
    llm = FakeLLM(parsed={"Mentions": {"drugs": ["zuranolone"]},
                          "Route": {"kind": "label", "reason": "label question", "label_query": "zuranolone contraindications"}})
    r = ask("What are the contraindications for zuranolone?", llm)
    assert r.abstained and r.evidence_ids == [] and "zuranolone" in r.abstain_reason
    assert "DraftAnswer" not in llm.calls


def test_count_goes_through_tool_and_is_cited():
    llm = FakeLLM(
        parsed={"Route": {"kind": "trials", "reason": "count", "label_query": None},
                "DraftAnswer": {"text": "There are 207 such trials [count:1].", "citations": ["count:1"],
                                "abstained": False, "abstain_reason": None}},
        tool_calls=[ToolCall("count_trials", {"drug": "nivolumab", "condition": "melanoma", "phase": ["PHASE2"]})],
    )
    r = ask("How many Phase 2 trials test nivolumab in melanoma?", llm)
    assert not r.abstained and r.evidence_ids == ["count:1"] and r.citations[0]["id"] == "count:1"


def test_invalid_tool_arguments_are_rejected_not_sent():
    llm = FakeLLM(parsed={"Route": {"kind": "trials", "reason": "lookup", "label_query": None}},
                  tool_calls=[ToolCall("get_trial", {"nct_id": "12345"})])
    r = ask("Tell me about trial 12345 for nivolumab", llm)
    assert r.tool_runs[0].ok is False and r.abstained
