"""HuBERT-base CPU throughput / peak-RAM benchmark (synthetic audio, all hidden states)."""
import platform
import resource
import time

import torch
import torchaudio
from transformers import HubertConfig, HubertModel

SR = 16000
DURATIONS = [5.0, 10.0, 15.0, 30.0]
REPEATS = 3


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def run(fn, wav, repeats=REPEATS):
    fn(wav[:, : SR * 2])  # warmup
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        fn(wav)
        times.append(time.perf_counter() - t)
    return min(times)


def hf_forward(model):
    def f(w):
        with torch.inference_mode():
            return model(w, output_hidden_states=True).hidden_states
    return f


def ta_forward(model):
    def f(w):
        with torch.inference_mode():
            return model.extract_features(w)[0]
    return f


def bench(name, fn):
    print(f"\n== {name} ==")
    out = {}
    for d in DURATIONS:
        wav = torch.randn(1, int(d * SR))
        t = run(fn, wav)
        out[d] = d / t
        print(f"  {d:5.1f}s audio: {t:6.2f}s wall  -> {d / t:6.2f}x realtime")
    return out


print("CPU:", platform.processor() or platform.machine())
try:
    print([l for l in open("/proc/cpuinfo") if "model name" in l][:1])
except OSError:
    pass
print("torch", torch.__version__, "default threads", torch.get_num_threads())

hf = HubertModel.from_pretrained("facebook/hubert-base-ls960").eval()
untrained = HubertModel(HubertConfig.from_pretrained("facebook/hubert-base-ls960")).eval()
print(f"RSS after loading models: {rss_mb():.0f} MB")

for threads in (torch.get_num_threads(), 4, 8):
    torch.set_num_threads(threads)
    print(f"\n######## threads={threads}")
    bench(f"HF HubertModel trained", hf_forward(hf))
    if threads == torch.get_num_threads() and threads != 4:
        bench("HF HubertModel UNTRAINED (random init)", hf_forward(untrained))

torch.set_num_threads(torch.get_num_threads())
ta = torchaudio.pipelines.HUBERT_BASE.get_model().eval()
bench("torchaudio HUBERT_BASE", ta_forward(ta))

# peak RAM, long utterance, fresh measure
before = rss_mb()
w = torch.randn(1, 35 * SR)
hf_forward(hf)(w)
print(f"\nPeak RSS (process high-water mark) after 35 s utterance: {rss_mb():.0f} MB (before: {before:.0f} MB)")
