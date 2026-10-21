import csv

import pytest

from conftest import ROOT, write_words_textgrid
from word_circles.fig6 import score_by_syllable_count
from word_circles.gold import gold_syllable_boundaries

DEV_TEXTGRIDS = ROOT / "data" / "textgrids" / "dev"
TARGETS = ROOT / "docs" / "research" / "fig6-gold-targets.csv"
METRIC_KEYS = {"F1": "F1", "OS": "OS", "TokenF1": "token_F1", "R": "R"}


@pytest.mark.data
def test_gold_syllables_per_group_reproduce_digitized_fig6_targets(evaluator_dir):
    if not DEV_TEXTGRIDS.exists():
        pytest.fail("dev TextGrids missing: run `uv run snakemake -j1 data/textgrids/dev`")
    result = score_by_syllable_count(gold_syllable_boundaries(DEV_TEXTGRIDS), DEV_TEXTGRIDS, evaluator_dir)

    misses = []
    for row in csv.DictReader(TARGETS.open()):
        got = 100 * result.scores[row["group"]][METRIC_KEYS[row["metric"]]]
        band = 5.0 if row["metric"] == "OS" else 3.0
        if abs(got - float(row["value"])) > band:
            misses.append((row["group"], row["metric"], round(got, 2), row["value"]))
    assert misses == []
    # Fig 6's `all` sits above Table II's gold row (85.5 F1 / 71.1 R).
    assert 100 * result.scores["all"]["F1"] > 85.5
    assert 100 * result.scores["all"]["R"] > 71.1
    assert result.word_counts == {"1": 38923, "2": 10986, "3": 3229, "4": 1014, "5": 228, "6": 22}


def syllables(*spans):
    return [(lo, hi, "s") for lo, hi in spans]


@pytest.fixture
def two_word_utterance(tmp_path):
    """One 1-syllable word (0.1-0.3 s, frames 5-15) then one 2-syllable word (0.3-0.7 s, frames 15-35)."""
    d = tmp_path / "tg"
    d.mkdir()
    write_words_textgrid(d / "u.TextGrid", {
        "words": [(0.0, 0.1, ""), (0.1, 0.3, "a"), (0.3, 0.7, "bc"), (0.7, 0.8, "")],
        "syllables": [(0.0, 0.1, "")] + syllables((0.1, 0.3), (0.3, 0.5), (0.5, 0.7)) + [(0.7, 0.8, "")],
    })
    return d


def test_boundary_shared_by_two_groups_counts_in_both_and_twice_in_all(two_word_utterance, evaluator_dir):
    result = score_by_syllable_count({"u": [0.3]}, two_word_utterance, evaluator_dir)
    # Group 1 refs {5, 15}, group 2 refs {15, 35}; the one prediction at frame 15 sits in both spans.
    for group in ("1", "2"):
        assert (result.scores[group]["precision"], result.scores[group]["recall"]) == (1.0, 0.5)
    assert (result.scores["all"]["precision"], result.scores["all"]["recall"]) == (1.0, 0.5)
    assert result.word_counts == {"1": 1, "2": 1}


def test_prediction_belongs_to_a_group_within_one_frame_of_its_word_span(two_word_utterance, evaluator_dir):
    inside = score_by_syllable_count({"u": [0.32]}, two_word_utterance, evaluator_dir)  # frame 16
    outside = score_by_syllable_count({"u": [0.34]}, two_word_utterance, evaluator_dir)  # frame 17
    assert inside.scores["1"]["recall"] == 0.5
    assert outside.scores["1"]["recall"] == 0.0


def test_utterance_without_a_group_is_left_out_of_that_group(two_word_utterance, evaluator_dir):
    write_words_textgrid(two_word_utterance / "v.TextGrid", {
        "words": [(0.0, 0.2, "a"), (0.2, 0.4, "")],
        "syllables": syllables((0.0, 0.2)) + [(0.2, 0.4, "")],
    })
    both = score_by_syllable_count({"u": [0.3], "v": [0.2]}, two_word_utterance, evaluator_dir)
    # v has no 2-syllable word, so group 2 still scores only u.
    assert (both.scores["2"]["precision"], both.scores["2"]["recall"]) == (1.0, 0.5)
    assert both.word_counts == {"1": 2, "2": 1}
    assert "3" not in both.scores


def test_words_with_seven_or_more_syllables_are_dropped_and_counted(tmp_path, evaluator_dir):
    write_words_textgrid(tmp_path / "u.TextGrid", {
        "words": [(0.0, 0.7, "long"), (0.7, 0.9, "a")],
        "syllables": syllables(*[(k / 10, (k + 1) / 10) for k in range(7)], (0.7, 0.9)),
    })
    result = score_by_syllable_count({"u": [0.7, 0.9]}, tmp_path, evaluator_dir)
    assert result.n_dropped_7plus == 1
    assert result.word_counts == {"1": 1}
