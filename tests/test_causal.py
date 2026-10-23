import math

import numpy as np
import pytest
import torch

from word_circles.causal import CausalWav2Vec2Features, frame_count, int16_scale, sample_stems, timit_stem


@pytest.fixture(scope="module")
def tiny_wav2vec2():
    """A small random wav2vec2 with base-960h's architecture choices (group-norm CNN, post-LN encoder)."""
    from transformers import Wav2Vec2Config, Wav2Vec2FeatureExtractor, Wav2Vec2Model

    config = Wav2Vec2Config(hidden_size=32, num_hidden_layers=3, num_attention_heads=2, intermediate_size=64,
                            conv_dim=(16,) * 7, feat_extract_norm="group", do_stable_layer_norm=False,
                            num_conv_pos_embeddings=16, num_conv_pos_embedding_groups=4)
    torch.manual_seed(0)
    return Wav2Vec2Model(config).eval(), Wav2Vec2FeatureExtractor(do_normalize=True, return_attention_mask=False)


def abstractography_features(audio, processor, model):
    """abstractography's `extract_features_of_audio`, one unbatched forward pass per frame."""
    out = []
    for frame_idx in range(len(audio) // 320):
        clipped = audio[:frame_idx * 320 + 400]
        inputs = processor(clipped, sampling_rate=16000, return_tensors="pt", padding=False)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        out.append(np.stack([hs[0, -1, :].cpu().numpy() for hs in outputs.hidden_states], axis=0))
    return np.stack(out, axis=0).transpose(1, 0, 2)


@pytest.mark.parametrize("n_samples", [4000, 4050, 4300])  # 4050 % 320 < 80: the last prefix is truncated
def test_batched_prefixes_match_one_pass_per_frame(tiny_wav2vec2, n_samples):
    model, processor = tiny_wav2vec2
    wav = 0.1 * np.random.default_rng(n_samples).standard_normal(n_samples).astype(np.float32)
    layers = CausalWav2Vec2Features(model, processor, "cpu", batch_size=4)(wav)
    reference = abstractography_features(wav, processor, model)
    assert len(layers) == 4 and layers[0].shape == (frame_count(n_samples), 32)
    np.testing.assert_allclose(np.stack(layers), reference, rtol=1e-4, atol=1e-5)


def test_int16_scale_normalises_every_prefix_like_the_raw_samples():
    from transformers import Wav2Vec2FeatureExtractor

    processor = Wav2Vec2FeatureExtractor(do_normalize=True)
    raw = np.random.default_rng(0).integers(-3000, 3000, 8000).astype(np.float32) / 32768
    stored = processor(raw, sampling_rate=16000).input_values[0].astype(np.float32)
    rescaled = int16_scale(stored)
    for n in (400, 2000, 8000):
        np.testing.assert_allclose(processor(rescaled[:n], sampling_rate=16000).input_values[0],
                                   processor(raw[:n], sampling_rate=16000).input_values[0], atol=1e-4)


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
