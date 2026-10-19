import pytest

from word_circles.labels import check_clipping, subset_alignment

ALI = """utt_id start_time word_dur word
a-1-0000 0.00 0.10 <SIL>
a-1-0000 0.10 0.30 HELLO
a-1-0000 0.40 0.30 WORLD
b-2-0000 0.00 0.50 FOO
"""


@pytest.fixture
def ali(tmp_path):
    path = tmp_path / "x-word.ali"
    path.write_text(ALI)
    return path


def test_subset_keeps_header_and_only_selected_utterances(ali, tmp_path):
    out = tmp_path / "subset.ali"
    subset_alignment(ali, ["a-1-0000"], out)
    assert out.read_text() == "".join(ALI.splitlines(keepends=True)[:4])


def test_words_overrunning_the_feature_frames_fail_above_the_threshold(ali):
    # WORLD spans frames 20-35; with 34 frames it overruns. FOO (0-25) fits in 30.
    frames = {"a-1-0000": 34, "b-2-0000": 30}
    with pytest.raises(ValueError, match="1 of 3 words"):
        check_clipping(ali, frames, max_fraction=0.001)
    assert check_clipping(ali, frames, max_fraction=0.5) == {"n_words": 3, "n_clipped": 1}
    assert check_clipping(ali, {"a-1-0000": 35, "b-2-0000": 30}, max_fraction=0.001)["n_clipped"] == 0
