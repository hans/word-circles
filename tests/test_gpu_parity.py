"""One-off prod check, not a pipeline gate: CPU and GPU extraction agree.

Run once on prod after the dev audio is fetched:
    uv run pytest -m gpu tests/test_gpu_parity.py
"""

import numpy as np
import pytest
import torch

from conftest import ROOT
from word_circles.features import HubertFeatures, audio_path, configure_precision, load_audio

REVISION = "dba3bb02fda4248b6e082697eee756de8fe8aa8a"
DEV_AUDIO = ROOT / "data" / "audio" / "dev"


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_cpu_and_gpu_hubert_features_agree():
    configure_precision()
    stems = sorted(p.stem for p in DEV_AUDIO.rglob("*.flac"))[:5]
    assert len(stems) == 5, "dev audio missing: run `uv run snakemake -j1 data/audio/dev`"
    cpu = HubertFeatures.from_pretrained(REVISION, "cpu")
    gpu = HubertFeatures.from_pretrained(REVISION, "cuda")
    for stem in stems:
        wav = load_audio(audio_path(DEV_AUDIO, stem))
        on_cpu, on_gpu = cpu(wav), gpu(wav)
        for layer in (0, 9, 12):
            np.testing.assert_allclose(on_gpu[layer], on_cpu[layer], rtol=1e-4, atol=1e-4,
                                       err_msg=f"{stem} L{layer}")
