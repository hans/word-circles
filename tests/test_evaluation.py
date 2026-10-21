import pytest

from conftest import write_words_textgrid
from word_circles.evaluation import score_boundaries

WORDS = [(0.0, 0.2, ""), (0.2, 0.6, "hello"), (0.6, 1.0, "world"), (1.0, 1.2, "")]


@pytest.fixture
def ref_dir(tmp_path):
    d = tmp_path / "refs"
    d.mkdir()
    for stem in ("a-1-0000", "a-1-0001"):
        write_words_textgrid(d / f"{stem}.TextGrid", {"words": WORDS})
    return d


def test_stem_mismatch_fails_listing_the_difference(ref_dir, evaluator_dir, tmp_path):
    predictions = {"a-1-0000": [0.6], "b-2-0000": [0.6]}
    with pytest.raises(ValueError, match=r"(?s)a-1-0001.*b-2-0000"):
        score_boundaries(predictions, ref_dir, evaluator_dir, tmp_path / "work")


def test_predicting_every_reference_end_scores_perfectly(ref_dir, evaluator_dir, tmp_path):
    # Reference boundaries are the end of every words interval, silences included.
    predictions = {stem: [0.2, 0.6, 1.0, 1.2] for stem in ("a-1-0000", "a-1-0001")}
    result = score_boundaries(predictions, ref_dir, evaluator_dir, tmp_path / "work")
    assert result.scores == pytest.approx({"precision": 1.0, "recall": 1.0, "F1": 1.0, "OS": 0.0, "R": 1.0,
                                           "token_precision": 1.0, "token_recall": 1.0, "token_F1": 1.0})
    assert result.empty == []


def test_empty_prediction_scores_as_placeholder_at_one_frame_and_is_counted(ref_dir, evaluator_dir, tmp_path):
    empty = score_boundaries({"a-1-0000": [0.6], "a-1-0001": []}, ref_dir, evaluator_dir, tmp_path / "w1")
    explicit = score_boundaries({"a-1-0000": [0.6], "a-1-0001": [0.02]}, ref_dir, evaluator_dir, tmp_path / "w2")
    assert empty.scores == explicit.scores
    assert empty.empty == ["a-1-0001"]
    assert explicit.empty == []
