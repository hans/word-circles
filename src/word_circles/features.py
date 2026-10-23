"""Frame-level features cached in the authors' (WTT) layout.

Models: trained HuBERT-base, randomly initialised HuBERT-base (one frozen
weights file per seed) and an 80-bin log-Mel spectrogram on the same frame
grid. Layer i is `hidden_states[i]`; log Mel has a single layer 0.
"""

import argparse
import hashlib
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from safetensors.torch import load_file, save_file
from tqdm import tqdm
from transformers import HubertConfig, HubertModel, Wav2Vec2FeatureExtractor

WIN = 400  # samples: 25 ms receptive field of HuBERT's conv stack
HOP = 320  # samples: 20 ms frame step


def frame_count(n_samples):
    """Frames HuBERT produces for `n_samples` of 16 kHz audio."""
    return (n_samples - WIN) // HOP + 1

SAMPLE_RATE = 16000
N_MELS = 80
N_FFT = 512
LOG_EPS = 1e-6
HUBERT_ID = "facebook/hubert-base-ls960"


def configure_precision():
    """Full fp32 on GPU: TF32 would change the CNN encoder's numerics vs CPU.

    A no-op on prod's Titan V (Volta has no TF32), kept so other GPUs behave the same.
    """
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def build_random_hubert(seed, path, revision=None):
    """Save HF default init of the full HuBERT-base (CNN included) under `seed`.

    Downstream code loads this file rather than re-initialising, so a
    `transformers`/`torch` upgrade cannot silently change the random model.
    """
    config = HubertConfig.from_pretrained(HUBERT_ID, revision=revision)
    torch.manual_seed(seed)
    model = HubertModel(config)
    save_file(model.state_dict(), str(path))


class HubertFeatures:
    """All hidden states of a HuBERT model for one whole utterance (batch 1, no padding)."""

    def __init__(self, model, device, do_normalize=True):
        self.model = model.to(device).eval()
        self.device = device
        self.processor = Wav2Vec2FeatureExtractor(do_normalize=do_normalize, sampling_rate=SAMPLE_RATE)

    @classmethod
    def from_pretrained(cls, revision, device, do_normalize=True):
        return cls(HubertModel.from_pretrained(HUBERT_ID, revision=revision), device, do_normalize)

    @classmethod
    def from_random_weights(cls, path, device, revision=None, do_normalize=True):
        model = HubertModel(HubertConfig.from_pretrained(HUBERT_ID, revision=revision))
        model.load_state_dict(load_file(str(path)))
        return cls(model, device, do_normalize)

    def __call__(self, wav):
        """Return [L0, ..., L12] as (T, 768) float32 arrays; L0 is the Transformer input."""
        inputs = self.processor(wav, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_values
        with torch.inference_mode():
            hidden = self.model(inputs.to(self.device), output_hidden_states=True).hidden_states
        layers = [h[0].float().cpu().numpy() for h in hidden]
        expected = frame_count(len(wav))
        if layers[0].shape[0] != expected:
            raise ValueError(f"HuBERT gave {layers[0].shape[0]} frames for {len(wav)} samples, expected {expected}")
        return layers


_MEL = None


def logmel_features(wav):
    """80-bin natural-log power Mel spectrogram on HuBERT's frame grid, unnormalised.

    With `center=False` the STFT centres each 400-sample window inside its
    512-sample FFT frame, so padding (512 - 400) / 2 zeros on each side puts
    frame k's window exactly on HuBERT's frame k, samples [320k, 320k + 400).
    """
    global _MEL
    if _MEL is None:
        _MEL = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE, n_fft=N_FFT, win_length=WIN, hop_length=HOP, n_mels=N_MELS,
            power=2.0, f_min=0.0, f_max=SAMPLE_RATE / 2, center=False,
        )
    with torch.inference_mode():
        pad = (N_FFT - WIN) // 2
        mel = _MEL(torch.nn.functional.pad(torch.as_tensor(wav, dtype=torch.float32), (pad, pad)))
    return torch.log(mel + LOG_EPS).T.contiguous().numpy()


def load_audio(path):
    """16 kHz mono float32 samples. LibriSpeech is never resampled."""
    wav, sr = sf.read(path, dtype="float32")
    if sr != SAMPLE_RATE or wav.ndim != 1:
        raise ValueError(f"{path}: expected 16 kHz mono, got sr={sr}, shape={wav.shape}")
    return wav


def audio_path(audio_dir, stem):
    speaker, chapter, _ = stem.split("-")
    return Path(audio_dir) / speaker / chapter / f"{stem}.flac"


def _toml_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'


def write_manifest(path, provenance, frames):
    """WTT's manifest keys, our provenance, and every utterance's frame count."""
    fields = {
        "audio_sampling_rate": SAMPLE_RATE,
        "feature_subsampling_factor": HOP,
        "feature_frame_rate": SAMPLE_RATE / HOP,
        **provenance,
    }
    lines = [f"{key} = {_toml_value(value)}" for key, value in fields.items()]
    lines += ["", "[frames]"] + [f'"{stem}" = {n}' for stem, n in frames.items()]
    Path(path).write_text("\n".join(lines) + "\n")


def extract_split(extractor, audio_dir, stems, cache_dir, layers, provenance):
    """Cache the LibriSpeech utterances `stems` from `audio_dir` (see `write_cache`)."""
    items = ((stem, load_audio(audio_path(audio_dir, stem))) for stem in stems)
    write_cache(extractor, items, len(stems), cache_dir, layers, provenance)


def write_cache(extractor, items, n_items, cache_dir, layers, provenance):
    """Write `<cache_dir>/<layer>.npz` ({stem: (T, D) float32}, uncompressed) and `manifest.toml`.

    `items` yields (stem, wav). Arrays stream into the archives one utterance
    at a time, so memory stays at one utterance regardless of split size.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    archives = {layer: zipfile.ZipFile(cache_dir / f"{layer}.npz", "w", zipfile.ZIP_STORED, allowZip64=True)
                for layer in layers}
    frames = {}
    try:
        for stem, wav in tqdm(items, total=n_items, desc=str(cache_dir), mininterval=30):
            outputs = extractor(wav)
            frames[stem] = len(outputs[0])
            for layer in layers:
                array = np.ascontiguousarray(outputs[layer], dtype=np.float32)
                with archives[layer].open(f"{stem}.npy", "w", force_zip64=True) as member:
                    np.lib.format.write_array(member, array, allow_pickle=False)
    finally:
        for archive in archives.values():
            archive.close()
    write_manifest(cache_dir / "manifest.toml", provenance, frames)


def provenance(model, device, revision, do_normalize, weights=None):
    fields = {"model_name": model, "device": str(device), "torch_version": torch.__version__,
              "cuda_version": torch.version.cuda or "none",
              "cudnn_version": torch.backends.cudnn.version() or 0,
              "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
              "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
              "float32_matmul_precision": torch.get_float32_matmul_precision()}
    if model == "logmel":
        fields.update(n_mels=N_MELS, n_fft=N_FFT, log_eps=LOG_EPS)
    else:
        fields.update(hf_model_id=HUBERT_ID, hf_revision=revision, do_normalize=do_normalize)
    if weights is not None:
        fields["weights_sha256"] = hashlib.sha256(Path(weights).read_bytes()).hexdigest()
    return fields


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    rand = sub.add_parser("random-weights")
    rand.add_argument("--seed", type=int, required=True)
    rand.add_argument("--revision", required=True)
    rand.add_argument("out", type=Path)
    ext = sub.add_parser("extract")
    ext.add_argument("--model", required=True, help="hubert-base-ls960, hubert-base-random-s<seed> or logmel")
    ext.add_argument("--weights", type=Path, help="safetensors state_dict for a random-init model")
    ext.add_argument("--revision", required=True)
    ext.add_argument("--do-normalize", type=lambda s: s.lower() == "true", default=True)
    ext.add_argument("--device", default="cpu")
    ext.add_argument("--layers", type=lambda s: [int(x) for x in s.split(",")], required=True)
    ext.add_argument("audio_dir", type=Path)
    ext.add_argument("utts", type=Path)
    ext.add_argument("cache_dir", type=Path)
    args = parser.parse_args()

    if args.command == "random-weights":
        args.out.parent.mkdir(parents=True, exist_ok=True)
        build_random_hubert(args.seed, args.out, revision=args.revision)
        return

    configure_precision()
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise SystemExit("device is CUDA but torch.cuda.is_available() is False (CPU wheel installed?)")
        print(f"extracting on {torch.cuda.get_device_name(args.device)}", flush=True)
    if args.model == "logmel":
        extractor = lambda wav: [logmel_features(wav)]
    elif args.weights is not None:
        extractor = HubertFeatures.from_random_weights(args.weights, args.device, args.revision, args.do_normalize)
    else:
        extractor = HubertFeatures.from_pretrained(args.revision, args.device, args.do_normalize)
    stems = args.utts.read_text().split()
    extract_split(extractor, args.audio_dir, stems, args.cache_dir, args.layers,
                  provenance(args.model, args.device, args.revision, args.do_normalize, args.weights))


if __name__ == "__main__":
    main()
