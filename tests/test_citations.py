from app.generate import DraftAnswer, Evidence, check_citations

EV = [Evidence(id="label:warfarin:boxed_warning:0", source="label", title="Warfarin label", url="u", text="...")]


def draft(text, citations, abstained=False):
    return DraftAnswer(text=text, citations=citations, abstained=abstained, abstain_reason=None)


def test_valid_citations_pass():
    a = check_citations(draft("Bleeding risk [label:warfarin:boxed_warning:0].", ["label:warfarin:boxed_warning:0"]), EV)
    assert not a.abstained and [c.id for c in a.citations] == ["label:warfarin:boxed_warning:0"]


def test_invented_citation_in_list_forces_abstain():
    a = check_citations(draft("Bleeding risk.", ["label:warfarin:boxed_warning:7"]), EV)
    assert a.abstained and a.invented_citations == ["label:warfarin:boxed_warning:7"]


def test_invented_inline_marker_forces_abstain():
    a = check_citations(draft("Bleeding [label:warfarin:boxed_warning:0] and INR [trial:NCT00000000].",
                              ["label:warfarin:boxed_warning:0"]), EV)
    assert a.abstained and a.invented_citations == ["trial:NCT00000000"]


def test_uncited_answer_forces_abstain():
    a = check_citations(draft("Warfarin causes bleeding.", []), EV)
    assert a.abstained and a.abstain_reason == "answer had no citations"
