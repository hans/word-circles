import numpy as np
import pytest

from word_circles.features import frame_count


@pytest.mark.parametrize("n_samples, frames", [(400, 1), (719, 1), (720, 2), (16000, 49), (16320, 50)])
def test_frame_count_is_hubert_conv_stack_output_length(n_samples, frames):
    assert frame_count(n_samples) == frames


@pytest.fixture(scope="module")
def random_hubert(tmp_path_factory):
    from word_circles.features import HubertFeatures, build_random_hubert

    weights = tmp_path_factory.mktemp("weights") / "hubert-base-random-s0.safetensors"
    build_random_hubert(seed=0, path=weights)
    return HubertFeatures.from_random_weights(weights, device="cpu")


@pytest.mark.parametrize("n_samples", [16000, 16719, 23456])
def test_logmel_and_hubert_share_one_frame_grid(random_hubert, n_samples):
    from word_circles.features import logmel_features

    wav = np.random.default_rng(0).standard_normal(n_samples).astype(np.float32)
    layers = random_hubert(wav)
    assert len(layers) == 13
    assert {layer.shape for layer in layers} == {(frame_count(n_samples), 768)}
    assert logmel_features(wav).shape == (frame_count(n_samples), 80)


def test_layer_12_is_the_models_final_output(random_hubert):
    import torch

    wav = np.random.default_rng(1).standard_normal(16000).astype(np.float32)
    inputs = random_hubert.processor(wav, sampling_rate=16000, return_tensors="pt").input_values
    with torch.inference_mode():
        final = random_hubert.model(inputs).last_hidden_state[0].numpy()
    np.testing.assert_array_equal(random_hubert(wav)[12], final)


def write_flac(path, n_samples, seed):
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    wav = 0.1 * np.random.default_rng(seed).standard_normal(n_samples)
    sf.write(path, wav, 16000, subtype="PCM_16")


def test_cache_round_trips_through_the_authors_feature_reader(random_hubert, tmp_path):
    from concept_analysis.features.feature_reader import NpzFeatureReader
    from word_circles.features import extract_split

    audio = tmp_path / "audio"
    stems = {"84-121123-0000": 20000, "84-121123-0001": 33000}
    for k, (stem, n) in enumerate(stems.items()):
        speaker, chapter, _ = stem.split("-")
        write_flac(audio / speaker / chapter / f"{stem}.flac", n, seed=k)

    cache = tmp_path / "cache"
    extract_split(random_hubert, audio, list(stems), cache, layers=[0, 9, 12], provenance={"model_name": "random"})

    reader = NpzFeatureReader(cache)
    assert reader.feature_frame_rate_for_layer(9) == 50.0
    assert reader.feature_frame_rate_for_layer(9, legacy=True) == {"sr": 16000, "subsampling_factor": 320}
    for layer in (0, 9, 12):
        assert sorted(reader.list_ids(layer)) == sorted(stems)
        arrays = reader.read(list(stems), layer)
        assert [a.shape for a in arrays] == [(frame_count(n), 768) for n in stems.values()]
        assert all(a.dtype == np.float32 for a in arrays)
