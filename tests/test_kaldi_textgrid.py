import shutil

import pytest
import textgrids

from conftest import ROOT
from word_circles.evaluation import score_boundaries
from word_circles.gold import gold_syllable_boundaries
from word_circles.kaldi_textgrid import kaldi_to_textgrids, write_words_textgrid

DEV_TEXTGRIDS = ROOT / "data" / "textgrids" / "dev"


def words(path):
    return [(round(i.xmin, 6), round(i.xmax, 6), i.text) for i in textgrids.TextGrid(path)["words"]]


def test_kaldi_rows_become_a_contiguous_words_tier_with_silences_as_empty_text(tmp_path):
    ali = tmp_path / "x.ali"
    ali.write_text("utt_id start_time word_dur word\n"
                   "a-1-0000 0.00 0.16 <SIL>\na-1-0000 0.16 0.21 FOR\na-1-0000 0.40 0.30 THE\n"
                   "b-2-0000 0.00 0.50 FOO\n")
    kaldi_to_textgrids(ali, ["a-1-0000"], tmp_path / "out")
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["a-1-0000.TextGrid"]
    # The 0.37-0.40 gap becomes a silence interval.
    assert words(tmp_path / "out" / "a-1-0000.TextGrid") == [
        (0.0, 0.16, ""), (0.16, 0.37, "FOR"), (0.37, 0.4, ""), (0.4, 0.7, "THE")]


@pytest.mark.data
def test_rewritten_words_tiers_score_identically_to_the_originals(evaluator_dir, tmp_path):
    stems = sorted(p.stem for p in DEV_TEXTGRIDS.glob("*.TextGrid"))[:200]
    original, rewritten = tmp_path / "orig", tmp_path / "rewritten"
    original.mkdir(), rewritten.mkdir()
    for stem in stems:
        shutil.copy(DEV_TEXTGRIDS / f"{stem}.TextGrid", original)
        write_words_textgrid(rewritten / f"{stem}.TextGrid", words(original / f"{stem}.TextGrid"))
    predictions = gold_syllable_boundaries(original)
    a = score_boundaries(predictions, original, evaluator_dir, tmp_path / "w1")
    b = score_boundaries(predictions, rewritten, evaluator_dir, tmp_path / "w2")
    assert a.scores == b.scores
