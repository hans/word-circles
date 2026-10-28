"""Causal wav2vec2-base-960h features on abstractography's frame grid, and its TIMIT sentences.

abstractography (`extract_model_features.py`) gives frame f the last position
of a forward pass on the first f * 320 + 400 samples, normalised per prefix.
An utterance therefore has len // 320 frames and no frame sees future audio.
Layer i is `hidden_states[i]`.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

from word_circles.data import link_references, take
from word_circles.features import (SAMPLE_RATE, configure_precision, extract_split, feature_reader, provenance,
                                   write_h5_cache)

MODEL_ID = "facebook/wav2vec2-base-960h"
MODEL_NAME = "wav2vec2-base-960h-causal"
HOP = 320
WIN = 400


def frame_count(n_samples):
    """Frames abstractography's causal extractor produces for `n_samples` of 16 kHz audio."""
    return n_samples // HOP


class CausalWav2Vec2Features:
    """All hidden states of every causal prefix of one utterance.

    The CNN and feature projection run on each prefix alone, so group norm and
    per-prefix input normalisation see only that prefix. Only the Transformer
    is batched: prefixes are right-padded and masked, which the encoder zeroes
    before its positional convolution, matching an unpadded pass.
    """

    def __init__(self, model, processor, device, batch_size=32):
        self.model = model.to(device).eval()
        self.processor = processor
        self.device = device
        self.batch_size = batch_size

    @classmethod
    def from_pretrained(cls, revision, device, batch_size=32):
        return cls(Wav2Vec2Model.from_pretrained(MODEL_ID, revision=revision),
                   Wav2Vec2FeatureExtractor.from_pretrained(MODEL_ID, revision=revision), device, batch_size)

    def _embed(self, prefix):
        inputs = self.processor(prefix, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_values
        features = self.model.feature_extractor(inputs.to(self.device)).transpose(1, 2)
        if features.shape[1] == 0:
            raise ValueError(f"a {len(prefix)}-sample prefix is shorter than the {WIN}-sample receptive field")
        return self.model.feature_projection(features)[0][0]

    def _encode(self, embeddings):
        """[L0, ..., L12] as (B, T_max, H) for right-padded `embeddings`."""
        lengths = [len(e) for e in embeddings]
        hidden = torch.zeros(len(embeddings), max(lengths), embeddings[0].shape[-1], device=self.device)
        mask = torch.zeros(len(embeddings), max(lengths), dtype=torch.bool, device=self.device)
        for j, e in enumerate(embeddings):
            hidden[j, :len(e)] = e
            mask[j, :len(e)] = True
        # Wav2Vec2Encoder does not return hidden states itself; L0 is its input layer norm's output.
        states = []
        keep = lambda module, args, output: states.append(output[0] if isinstance(output, tuple) else output)
        hooks = [self.model.encoder.layer_norm.register_forward_hook(keep)]
        hooks += [layer.register_forward_hook(keep) for layer in self.model.encoder.layers]
        try:
            self.model.encoder(hidden, attention_mask=mask)
        finally:
            for hook in hooks:
                hook.remove()
        return states

    def __call__(self, wav):
        """Return [L0, ..., L12] as (len(wav) // 320, 768) float32 arrays."""
        n_frames = frame_count(len(wav))
        layers = [[] for _ in range(self.model.config.num_hidden_layers + 1)]
        with torch.inference_mode():
            for start in range(0, n_frames, self.batch_size):
                frames = range(start, min(start + self.batch_size, n_frames))
                embeddings = [self._embed(wav[:f * HOP + WIN]) for f in frames]
                states = self._encode(embeddings)
                last = torch.tensor([len(e) - 1 for e in embeddings], device=self.device)
                rows = torch.arange(len(embeddings), device=self.device)
                for layer, state in zip(layers, states):
                    layer.append(state[rows, last].float().cpu().numpy())
        return [np.concatenate(layer) for layer in layers]


def timit_stem(file, utt_id):
    """abstractography's `stem_for`: `<speaker folder>_<utterance id>`, lower-cased."""
    return f"{file.split('/')[-2].lower()}_{utt_id.lower()}"


def int16_scale(input_values):
    """Rescale whole-utterance-normalised 16-bit audio back to int16 / 32768 steps, up to an offset.

    Per-prefix normalisation removes the offset, so the result normalises like
    the WAV samples abstractography read, up to the processor's 1e-7 epsilon.
    """
    x = input_values.astype(np.float64)
    x = x - x[np.argmin(np.abs(x - np.median(x)))]  # a sample at the dense centre of the distribution
    steps = np.diff(np.unique(x))
    step = float(np.median(steps[steps < 1.5 * steps.min()]))
    # Refit the step on ever wider level ranges, so rounding error never accumulates past half a level.
    for limit in (64, 1024, 16384, np.inf):
        near = x[np.abs(x) < limit * step]
        k = np.round(near / step)
        if k.any():
            step = float((k * near).sum() / (k * k).sum())
    levels = x / step
    if np.ptp(levels) >= 2 ** 16 or np.abs(levels - np.round(levels)).max() > 0.05:
        raise ValueError("input_values are not on a 16-bit grid")
    return (input_values / (step * 32768)).astype(np.float32)


def timit_index(dataset):
    return {timit_stem(f, i): k for k, (f, i) in enumerate(zip(dataset["file"], dataset["id"]))}


def timit_items(dataset_dir, stems):
    """(stem, wav) for `stems` from the HF `timit-no_repeats` dataset."""
    import datasets

    dataset = datasets.load_from_disk(str(dataset_dir))
    index = timit_index(dataset.select_columns(["file", "id"]))
    for stem in stems:
        yield stem, int16_scale(np.asarray(dataset[index[stem]]["input_values"], dtype=np.float32))


def select_timit(dataset_dir, stems_file, n_utts=None, expected=None):
    import datasets

    stems = sorted(Path(stems_file).read_text().split())
    index = timit_index(datasets.load_from_disk(str(dataset_dir)).select_columns(["file", "id"]))
    missing = sorted(set(stems) - set(index))
    if missing:
        raise ValueError(f"{dataset_dir} lacks {len(missing)} stems, e.g. {missing[:5]}")
    return take(stems, n_utts, expected, source=str(stems_file))


def sample_stems(stems, n_utts, seed):
    """A sorted, seeded sample of `n_utts` stems (all of them when `n_utts` is None)."""
    stems = sorted(stems)
    if n_utts is None or n_utts >= len(stems):
        return stems
    return sorted(random.Random(seed).sample(stems, n_utts))


def load_reference(path):
    """abstractography's .npz: a pickled dict saved positionally, or a `features=` archive. (L, T, H)."""
    archive = np.load(path, allow_pickle=True)
    return archive["arr_0"].item()["features"] if "arr_0" in archive else archive["features"]


def compare(ours, references, layers):
    """Per-layer agreement of `ours` ({layer: {stem: (T, H)}}) with `references` ({stem: (L, T, H)})."""
    rows, frame_mismatches = [], []
    for stem, ref in references.items():
        if ref.shape[1] != len(ours[layers[0]][stem]):
            frame_mismatches.append({"stem": stem, "ours": len(ours[layers[0]][stem]), "reference": ref.shape[1]})
            continue
        for layer in layers:
            a, b = ours[layer][stem], ref[layer]
            cos = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
            rows.append({"stem": stem, "layer": layer, "n_frames": len(a), "max_abs_diff": float(np.abs(a - b).max()),
                         "max_rel_diff": float(np.abs(a - b).max() / np.abs(b).max()), "min_cos": float(cos.min()),
                         "min_cos_frame": int(cos.argmin())})
    by_layer = {layer: {"max_rel_diff": max(r["max_rel_diff"] for r in rows if r["layer"] == layer),
                        "min_cos": min(r["min_cos"] for r in rows if r["layer"] == layer)}
                for layer in layers if any(r["layer"] == layer for r in rows)}
    return {"n_stems": len(references), "frame_mismatches": frame_mismatches, "by_layer": by_layer, "rows": rows}


def read_cache(cache_dir, stems, layers):
    reader = feature_reader(cache_dir)
    return {layer: dict(zip(stems, reader.read_layer(stems, layer))) for layer in layers}


def parity(cache_dir, reference_dir, stems, layers):
    """Per-layer agreement of our cache with abstractography's features on the stems both have."""
    references = {p.stem: p for p in Path(reference_dir).glob("*.npz")}
    shared = [s for s in stems if s in references]
    if not shared:
        raise ValueError(f"no reference .npz in {reference_dir} matches the cached stems")
    return compare(read_cache(cache_dir, shared, layers), {s: load_reference(references[s]) for s in shared}, layers)


def reference_features(audio, processor, model):
    """abstractography's `extract_features_of_audio`: one unbatched pass per prefix. (L, T, H)."""
    frames = []
    for f in range(frame_count(len(audio))):
        inputs = processor(audio[:f * HOP + WIN], sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=False)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        frames.append(np.stack([hs[0, -1, :].cpu().numpy() for hs in outputs.hidden_states]))
    return np.stack(frames).transpose(1, 0, 2)


def parity_unbatched(cache_dir, dataset_dir, stems, layers, revision):
    """Our cache against abstractography's extraction loop on CPU, run on the same rebuilt TIMIT audio.

    Agreement to float precision clears our batched extractor, leaving the
    input audio to explain any gap to abstractography's .npz.
    """
    processor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_ID, revision=revision)
    model = Wav2Vec2Model.from_pretrained(MODEL_ID, revision=revision).eval()
    references = {stem: reference_features(wav, processor, model) for stem, wav in timit_items(dataset_dir, stems)}
    return compare(read_cache(cache_dir, stems, layers), references, layers)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("extract-ls", "extract-timit"):
        ext = sub.add_parser(name)
        ext.add_argument("--revision", required=True)
        ext.add_argument("--device", default="cpu")
        ext.add_argument("--batch-size", type=int, default=32)
        ext.add_argument("--layers", type=lambda s: [int(x) for x in s.split(",")], required=True)
        ext.add_argument("source", type=Path, help="LibriSpeech audio dir or timit-no_repeats dataset dir")
        ext.add_argument("utts", type=Path)
        ext.add_argument("cache_dir", type=Path)
    n_utts = lambda s: None if s == "None" else int(s)
    sel = sub.add_parser("select-sample", help="seeded utterance sample, with its reference TextGrids")
    sel.add_argument("utts", type=Path)
    sel.add_argument("refs", type=Path)
    sel.add_argument("out_utts", type=Path)
    sel.add_argument("out_refs", type=Path)
    sel.add_argument("--n-utts", type=n_utts, default=None)
    sel.add_argument("--seed", type=int, default=0)
    first = sub.add_parser("select-first", help="the first n utterances by stem")
    first.add_argument("utts", type=Path)
    first.add_argument("out_utts", type=Path)
    first.add_argument("--n-utts", type=n_utts, default=None)
    tim = sub.add_parser("select-timit")
    tim.add_argument("dataset_dir", type=Path)
    tim.add_argument("stems", type=Path)
    tim.add_argument("out_utts", type=Path)
    tim.add_argument("--n-utts", type=n_utts, default=None)
    tim.add_argument("--expected", type=int, default=None)
    par = sub.add_parser("parity", help="compare a TIMIT cache with abstractography's .npz features")
    par.add_argument("cache_dir", type=Path)
    par.add_argument("reference_dir", type=Path)
    par.add_argument("utts", type=Path)
    par.add_argument("out", type=Path)
    par.add_argument("--layers", type=lambda s: [int(x) for x in s.split(",")], required=True)
    unb = sub.add_parser("parity-unbatched", help="compare a TIMIT cache with abstractography's loop on the same audio")
    unb.add_argument("cache_dir", type=Path)
    unb.add_argument("dataset_dir", type=Path)
    unb.add_argument("out", type=Path)
    unb.add_argument("--stems", type=lambda s: s.split(","), required=True)
    unb.add_argument("--revision", required=True)
    unb.add_argument("--layers", type=lambda s: [int(x) for x in s.split(",")], required=True)
    args = parser.parse_args()

    if args.command.startswith("select"):
        if args.command == "select-sample":
            stems = sample_stems(args.utts.read_text().split(), args.n_utts, args.seed)
            link_references(args.refs, stems, args.out_refs)
        elif args.command == "select-first":
            stems = take(args.utts.read_text().split(), args.n_utts)
        else:
            stems = select_timit(args.dataset_dir, args.stems, args.n_utts, args.expected)
        args.out_utts.parent.mkdir(parents=True, exist_ok=True)
        args.out_utts.write_text("".join(f"{s}\n" for s in stems))
        return
    if args.command.startswith("parity"):
        if args.command == "parity":
            report = parity(args.cache_dir, args.reference_dir, args.utts.read_text().split(), args.layers)
        else:
            report = parity_unbatched(args.cache_dir, args.dataset_dir, args.stems, args.layers, args.revision)
        args.out.write_text(json.dumps(report, indent=1) + "\n")
        print(json.dumps({k: report[k] for k in ("n_stems", "by_layer")}, indent=1))
        return

    configure_precision()
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise SystemExit("device is CUDA but torch.cuda.is_available() is False (CPU wheel installed?)")
        print(f"extracting on {torch.cuda.get_device_name(args.device)}", flush=True)
    extractor = CausalWav2Vec2Features.from_pretrained(args.revision, args.device, args.batch_size)
    fields = {**provenance(MODEL_NAME, args.device, args.revision, True), "hf_model_id": MODEL_ID,
              "causal_prefixes": True, "batch_size": args.batch_size}
    stems = args.utts.read_text().split()
    if args.command == "extract-ls":
        extract_split(extractor, args.source, stems, args.cache_dir, args.layers, fields, writer=write_h5_cache)
    else:
        write_h5_cache(extractor, timit_items(args.source, stems), len(stems), args.cache_dir, args.layers, fields)


if __name__ == "__main__":
    main()
