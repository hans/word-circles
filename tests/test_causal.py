import math

import numpy as np
import pytest
import torch

from word_circles.causal import (CausalWav2Vec2Features, compare, frame_count, int16_scale, peak_normalise,
                                 reference_features, sample_stems, timit_stem)


@pytest.fixture(scope="module")
def tiny_wav2vec2():
    """A small random wav2vec2 with base-960h's architecture choices (group-norm CNN, post-LN encoder)."""
    from transformers import Wav2Vec2Config, Wav2Vec2FeatureExtractor, Wav2Vec2Model

    config = Wav2Vec2Config(hidden_size=32, num_hidden_layers=3, num_attention_heads=2, intermediate_size=64,
                            conv_dim=(16,) * 7, feat_extract_norm="group", do_stable_layer_norm=False,
                            num_conv_pos_embeddings=16, num_conv_pos_embedding_groups=4)
    torch.manual_seed(0)
    return Wav2Vec2Model(config).eval(), Wav2Vec2FeatureExtractor(do_normalize=True, return_attention_mask=False)


@pytest.mark.parametrize("n_samples", [4000, 4050, 4300])  # 4050 % 320 < 80: the last prefix is truncated
def test_batched_prefixes_match_one_pass_per_frame(tiny_wav2vec2, n_samples):
    model, processor = tiny_wav2vec2
    wav = 0.1 * np.random.default_rng(n_samples).standard_normal(n_samples).astype(np.float32)
    layers = CausalWav2Vec2Features(model, processor, "cpu", batch_size=4)(wav)
    reference = reference_features(wav, processor, model)
    assert len(layers) == 4 and layers[0].shape == (frame_count(n_samples), 32)
    np.testing.assert_allclose(np.stack(layers), reference, rtol=1e-4, atol=1e-5)


def test_compare_reports_each_layers_worst_frame():
    rng = np.random.default_rng(0)
    ref = rng.standard_normal((2, 5, 3)).astype(np.float32)
    ours = {0: {"s": ref[0].copy()}, 1: {"s": ref[1].copy()}}
    ours[1]["s"][3] = -ours[1]["s"][3]
    report = compare(ours, {"s": ref}, [0, 1])
    assert report["frame_mismatches"] == [] and report["by_layer"][0]["min_cos"] == pytest.approx(1)
    assert report["by_layer"][1]["min_cos"] == pytest.approx(-1) and report["rows"][1]["min_cos_frame"] == 3
    assert compare({0: {"s": ref[0][:4]}}, {"s": ref}, [0])["frame_mismatches"][0]["ours"] == 4


def test_int16_scale_normalises_every_prefix_like_the_raw_samples():
    from transformers import Wav2Vec2FeatureExtractor

    processor = Wav2Vec2FeatureExtractor(do_normalize=True)
    raw = np.random.default_rng(0).integers(-3000, 3000, 8000).astype(np.float32) / 32768
    stored = processor(raw, sampling_rate=16000).input_values[0].astype(np.float32)
    rescaled = int16_scale(stored)
    for n in (400, 2000, 8000):
        np.testing.assert_allclose(processor(rescaled[:n], sampling_rate=16000).input_values[0],
                                   processor(raw[:n], sampling_rate=16000).input_values[0], atol=1e-4)


def test_peak_normalise_puts_the_peak_at_99_percent_of_full_scale_on_the_16_bit_grid():
    raw = np.random.default_rng(0).integers(-3000, 3000, 8000)
    out = peak_normalise((raw + 0.0095).astype(np.float32) / 32768)
    levels = out.astype(np.float64) * 32768
    assert np.abs(levels).max() == int(0.99 * 32768) and np.allclose(levels, np.round(levels))
    gain = 0.99 * 32768 / np.abs(raw).max()
    np.testing.assert_array_equal(levels, np.round(raw * gain))


def test_int16_scale_rejects_audio_off_a_16_bit_grid():
    with pytest.raises(ValueError, match="16-bit grid"):
        int16_scale(np.random.default_rng(0).standard_normal(4000).astype(np.float32))


def test_timit_stem_is_speaker_folder_and_id_lower_cased():
    assert timit_stem("/data/TIMIT/TRAIN/DR1/FCJF0/SI1027.WAV", "SI1027") == "fcjf0_si1027"


def test_sample_is_seeded_sorted_and_whole_split_when_unlimited():
    stems = [f"u{i:03d}" for i in range(50)]
    sample = sample_stems(stems, 10, seed=0)
    assert sample == sorted(sample) and len(sample) == 10 and sample == sample_stems(stems[::-1], 10, seed=0)
    assert sample_stems(stems, None, seed=0) == stems


@pytest.mark.parametrize("n_samples, frames", [(400, 1), (639, 1), (640, 2), (16000, 50)])
def test_frame_count_is_samples_over_hop(n_samples, frames):
    assert frame_count(n_samples) == frames
