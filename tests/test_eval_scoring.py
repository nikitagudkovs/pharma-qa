from eval.run import _contains


def test_must_contain_groups_are_and_of_ors():
    text = "Contraindicated with a personal or family history of MTC or MEN 2."
    assert _contains(text, [["MTC", "medullary thyroid"], ["MEN 2"]])
    assert not _contains(text, [["MTC"], ["pancreatitis"]])
