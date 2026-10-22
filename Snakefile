import json
import os

configfile: "config.yaml"

OUT = config["out_dir"]
DATA = config["data_dir"]
TP = config["third_party_dir"]
SPLITS = ["dev", "test", "train"]
FIG6_METHODS = ["gold_syllables", "dissim"] + [f"roseg_{e}_{v}" for e in ("cpca", "leace")
                                               for v in ("selected", "no_threshold")]
CACHE = config["cache_dir"] or f"{OUT}/cache"
FEAT = config["features"]
LS_SPLIT = config["librispeech"]["splits"]
HUBERT_MODELS = ["hubert-base-ls960"] + [f"hubert-base-random-s{s}" for s in FEAT["random_seeds"]]
EVAL_MODELS = HUBERT_MODELS + ["logmel"]


def cache_manifest(model, split):
    return f"{CACHE}/{model}/Librispeech/{LS_SPLIT[split]}/manifest.toml"


def model_layers(model):
    return [0] if model == "logmel" else FEAT["layers_eval"]


def layers_csv(layers):
    return ",".join(str(layer) for layer in layers)


wildcard_constraints:
    split="dev|test|train",
    method="[a-z0-9_]+?",
    seed=r"\d+",
    est="cpca|leace",


rule all:
    input:
        f"{OUT}/scores/gold_syllables_dev.csv",
        f"{OUT}/scores/fig6/gold_syllables_dev.csv",
        expand(f"{OUT}/utts/{{split}}.txt", split=SPLITS),
        [cache_manifest(m, s) for m in EVAL_MODELS for s in ("dev", "test")],
        [cache_manifest(m, "train") for m in HUBERT_MODELS],
        f"{OUT}/tables/sec2_metrics.csv",
        f"{OUT}/figures/fig3.pdf",
        f"{OUT}/figures/fig2",
        expand(f"{OUT}/tables/roseg/{{est}}_{{model}}.csv", est=["cpca", "leace"], model=HUBERT_MODELS),
        expand(f"{OUT}/scores/dissim_{{split}}.csv", split=["dev", "test"]),
        f"{OUT}/scores/kaldi/dissim_test.csv",
        f"{OUT}/figures/fig6.png",
        f"{OUT}/reports/replication_report.ipynb",
        [f"{OUT}/tables/dissim_retune.csv"] if config["dissim"]["retune"] else [],


rule fetch_third_party:
    output:
        directory(f"{TP}/{{name}}"),
    params:
        url=lambda w: config["third_party"][w.name]["url"],
        rev=lambda w: config["third_party"][w.name]["rev"],
    shell:
        "git clone --quiet {params.url} {output} && git -C {output} checkout --quiet {params.rev}"


rule fetch_textgrids:
    input:
        wtt=f"{TP}/wtt",
    output:
        directory(f"{DATA}/textgrids/{{split}}"),
    wildcard_constraints:
        split="dev|test",
    params:
        tarball=lambda w, input: f"{input.wtt}/alignments/Librispeech-Simon-flatten/"
        f"{config['librispeech']['splits'][w.split]}-flatten.tar.gz",
    shell:
        "mkdir -p {output} && tar --warning=no-unknown-keyword -xzf {params.tarball} -C {output} --strip-components=1 --wildcards '*.TextGrid'"


rule fetch_audio:
    output:
        directory(f"{DATA}/audio/{{split}}"),
    wildcard_constraints:
        split="dev|test",
    params:
        url=lambda w: f"{config['librispeech']['url']}/{config['librispeech']['splits'][w.split]}.tar.gz",
    shell:
        "mkdir -p {output} && curl -sSfL {params.url} | tar -xz -C {output} --strip-components=2"


# Stream the 6.4 GB train-clean-100 tarball once, keeping only the 10 h sample
# the authors' Kaldi alignments cover (2834 utterances, ~1 GB of FLAC).
rule fetch_train_audio:
    input:
        wtt=f"{TP}/wtt",
    output:
        members=f"{DATA}/audio/train.members",
        audio=directory(f"{DATA}/audio/train"),
    params:
        url=f"{config['librispeech']['url']}/{config['librispeech']['splits']['train']}.tar.gz",
        ali=lambda w, input: f"{input.wtt}/{config['alignments']['train']}",
    shell:
        "python -m word_circles.data tar-members {params.ali} LibriSpeech/train-clean-100 > {output.members}"
        " && mkdir -p {output.audio}"
        " && curl -sSfL {params.url} | tar -xz -C {output.audio} --strip-components=2 -T {output.members}"


rule select_train_utterances:
    input:
        wtt=f"{TP}/wtt",
    output:
        f"{OUT}/utts/train.txt",
    params:
        ali=lambda w, input: f"{input.wtt}/{config['alignments']['train']}",
        n_utts=config["n_utts"],
        expected=config["expected_utts"]["train"],
    shell:
        "python -m word_circles.data select-ali-utterances {params.ali} {output}"
        " --n-utts {params.n_utts} --expected {params.expected}"


rule select_utterances:
    input:
        textgrids=f"{DATA}/textgrids/{{split}}",
    output:
        f"{OUT}/utts/{{split}}.txt",
        directory(f"{OUT}/refs/{{split}}"),
    wildcard_constraints:
        split="dev|test",
    params:
        n_utts=config["n_utts"],
        expected=lambda w: config["expected_utts"][w.split],
    shell:
        "python -m word_circles.data select-utterances {input.textgrids} {output[0]} {output[1]}"
        " --n-utts {params.n_utts} --expected {params.expected}"


rule gold_syllables:
    input:
        refs=f"{OUT}/refs/{{split}}",
    output:
        f"{OUT}/predictions/gold_syllables_{{split}}.json",
    shell:
        "python -m word_circles.gold {input.refs} {output}"


rule score:
    input:
        predictions=f"{OUT}/predictions/{{method}}_{{split}}.json",
        refs=f"{OUT}/refs/{{split}}",
        evaluator=f"{TP}/evaluation",
    output:
        scores=f"{OUT}/scores/{{method}}_{{split}}.csv",
        empty=f"{OUT}/scores/{{method}}_{{split}}.empty.json",
    shell:
        "python -m word_circles.evaluation {input.predictions} {input.refs} {input.evaluator}"
        " {output.scores} {output.empty} --method {wildcards.method}"


rule kaldi_refs:
    input:
        wtt=f"{TP}/wtt",
        utts=f"{OUT}/utts/{{split}}.txt",
    output:
        directory(f"{OUT}/refs_kaldi/{{split}}"),
    wildcard_constraints:
        split="dev|test",
    params:
        ali=lambda w, input: f"{input.wtt}/{config['alignments'][w.split]}",
    shell:
        "python -m word_circles.kaldi_textgrid {params.ali} {input.utts} {output}"


rule score_kaldi:
    input:
        predictions=f"{OUT}/predictions/{{method}}_test.json",
        refs=f"{OUT}/refs_kaldi/test",
        evaluator=f"{TP}/evaluation",
    output:
        scores=f"{OUT}/scores/kaldi/{{method}}_test.csv",
        empty=f"{OUT}/scores/kaldi/{{method}}_test.empty.json",
    shell:
        "python -m word_circles.evaluation {input.predictions} {input.refs} {input.evaluator}"
        " {output.scores} {output.empty} --method {wildcards.method}"


rule score_fig6:
    input:
        predictions=f"{OUT}/predictions/{{method}}_dev.json",
        refs=f"{OUT}/refs/dev",
        evaluator=f"{TP}/evaluation",
    output:
        f"{OUT}/scores/fig6/{{method}}_dev.csv",
    shell:
        "python -m word_circles.fig6 {input.predictions} {input.refs} {input.evaluator} {output}"
        " --method {wildcards.method}"


rule random_weights:
    output:
        f"{DATA}/weights/hubert-base-random-s{{seed}}.safetensors",
    params:
        revision=FEAT["hf_revision"],
    shell:
        "python -m word_circles.features random-weights --seed {wildcards.seed} --revision {params.revision} {output}"


def extract_inputs(w):
    split = {v: k for k, v in LS_SPLIT.items()}[w.lsplit]
    inputs = {"audio": f"{DATA}/audio/{split}", "utts": f"{OUT}/utts/{split}.txt"}
    if w.model.startswith("hubert-base-random-s"):
        inputs["weights"] = f"{DATA}/weights/{w.model}.safetensors"
    return inputs


rule extract_features:
    input:
        unpack(extract_inputs),
    output:
        f"{CACHE}/{{model}}/Librispeech/{{lsplit}}/manifest.toml",
    params:
        weights=lambda w, input: f"--weights {input.weights}" if "weights" in input.keys() else "",
        layers=lambda w: "0" if w.model == "logmel"
        else layers_csv(FEAT["layers_train"] if w.lsplit == LS_SPLIT["train"] else FEAT["layers_eval"]),
        cache_dir=lambda w, output: os.path.dirname(output[0]),
        device=lambda w: "cpu" if w.model == "logmel" else FEAT["device"],
    resources:
        gpu=lambda w: 0 if w.model == "logmel" else 1,
    shell:
        "python -m word_circles.features extract --model {wildcards.model} {params.weights}"
        " --revision {FEAT[hf_revision]} --do-normalize {FEAT[do_normalize]} --device {params.device}"
        " --layers {params.layers} {input.audio} {input.utts} {params.cache_dir}"


rule labels:
    input:
        wtt=f"{TP}/wtt",
        utts=f"{OUT}/utts/{{split}}.txt",
        manifest=lambda w: cache_manifest("hubert-base-ls960", w.split),
    output:
        ali=f"{OUT}/alignments/{{split}}-word.ali",
        stats=f"{OUT}/alignments/{{split}}-word.clip.json",
    params:
        ali=lambda w, input: f"{input.wtt}/{config['alignments'][w.split]}",
    shell:
        "python -m word_circles.labels {params.ali} {input.utts} {input.manifest} {output.ali} {output.stats}"


rule sec2_fit:
    input:
        wtt=f"{TP}/wtt",
        manifest=lambda w: cache_manifest(w.model, "dev"),
        ali=f"{OUT}/alignments/dev-word.ali",
    output:
        directory(f"{OUT}/sec2/subspaces/{{model}}"),
    params:
        layers=lambda w: layers_csv(model_layers(w.model)),
        cache=lambda w, input: os.path.dirname(input.manifest),
        cfg=config["sec2"],
    shell:
        "python {input.wtt}/scripts/fit_temporal_subspace.py --input-feature-dir {params.cache}"
        " --alignment-path {input.ali} --layer-id {params.layers} --estimator Z_CPCA --annotation-unit word"
        " --min-dur {params.cfg[min_dur]} --max-dur {params.cfg[max_dur]}"
        " --low-freq-threshold {params.cfg[low_freq_threshold]} --output-dir {output}"


rule sec2_metrics:
    input:
        wtt=f"{TP}/wtt",
        fits=f"{OUT}/sec2/subspaces/{{model}}",
        dev=lambda w: cache_manifest(w.model, "dev"),
        test=lambda w: cache_manifest(w.model, "test"),
        dev_ali=f"{OUT}/alignments/dev-word.ali",
        test_ali=f"{OUT}/alignments/test-word.ali",
    output:
        f"{OUT}/sec2/metrics/{{model}}/score.csv",
    params:
        dev=lambda w, input: os.path.dirname(input.dev),
        test=lambda w, input: os.path.dirname(input.test),
    threads: 4
    shell:
        "python {input.wtt}/scripts/evaluate_temporal_metrics.py --subspace-dir {input.fits}"
        " --train-feature-dir {params.dev} --test-feature-dir {params.test}"
        " --train-alignment-path {input.dev_ali} --test-alignment-path {input.test_ali}"
        " --output-dir $(dirname {output}) --output-csv {output} --model-name {wildcards.model}"
        " --evaluation-level mean --n-jobs {threads}"


rule sec2_table:
    input:
        wtt=f"{TP}/wtt",
        scores=expand(f"{OUT}/sec2/metrics/{{model}}/score.csv", model=EVAL_MODELS),
    output:
        f"{OUT}/tables/sec2_metrics.csv",
    shell:
        "python {input.wtt}/scripts/gather_temporal_metrics.py --output-path {output} {input.scores}"


rule sec2_trajectories:
    input:
        wtt=f"{TP}/wtt",
        fits=f"{OUT}/sec2/subspaces/{{model}}",
    output:
        directory(f"{OUT}/figures/trajectories/{{model}}"),
    shell:
        "MPLBACKEND=Agg python {input.wtt}/scripts/plot_temporal_trajectories.py --input-dir {input.fits}"
        " --output-dir {output} --model-name {wildcards.model}"


rule fig2:
    input:
        table=f"{OUT}/tables/sec2_metrics.csv",
        plots=expand(f"{OUT}/figures/trajectories/{{model}}", model=EVAL_MODELS),
    output:
        directory(f"{OUT}/figures/fig2"),
    shell:
        "python -m word_circles.sec2 fig2 {input.table} {OUT}/figures/trajectories {output}"


rule fig3:
    input:
        f"{OUT}/tables/sec2_metrics.csv",
    output:
        pdf=f"{OUT}/figures/fig3.pdf",
        png=f"{OUT}/figures/fig3.png",
    shell:
        "MPLBACKEND=Agg python -m word_circles.sec2 fig3 {input} {output.pdf}"
        " && MPLBACKEND=Agg python -m word_circles.sec2 fig3 {input} {output.png}"


def roseg_subspace(est, model):
    return f"{OUT}/roseg/subspaces/{est}/{model}/{config['roseg']['layer']}"


rule roseg_fit_cpca:
    input:
        wtt=f"{TP}/wtt",
        manifest=lambda w: cache_manifest(w.model, "train"),
        ali=f"{OUT}/alignments/train-word.ali",
    output:
        directory(f"{OUT}/roseg/subspaces/cpca/{{model}}/{config['roseg']['layer']}"),
    params:
        cache=lambda w, input: os.path.dirname(input.manifest),
        cfg=config["sec2"],
        layer=config["roseg"]["layer"],
    shell:
        "python {input.wtt}/scripts/fit_temporal_subspace.py --input-feature-dir {params.cache}"
        " --alignment-path {input.ali} --layer-id {params.layer} --estimator Z_CPCA --annotation-unit word"
        " --min-dur {params.cfg[min_dur]} --max-dur {params.cfg[max_dur]}"
        " --low-freq-threshold {params.cfg[low_freq_threshold]} --output-dir {output}"


rule roseg_fit_leace:
    input:
        manifest=lambda w: cache_manifest(w.model, "train"),
        ali=f"{OUT}/alignments/train-word.ali",
    output:
        directory(f"{OUT}/roseg/subspaces/leace/{{model}}/{config['roseg']['layer']}"),
    params:
        cache=lambda w, input: os.path.dirname(input.manifest),
        cfg=config["sec2"],
        layer=config["roseg"]["layer"],
    shell:
        "python -m word_circles.roseg fit-leace {params.cache} {input.ali} {output} --layer {params.layer}"
        " --min-dur {params.cfg[min_dur]} --max-dur {params.cfg[max_dur]}"
        " --low-freq-threshold {params.cfg[low_freq_threshold]}"


rule roseg_tune:
    input:
        subspace=lambda w: roseg_subspace(w.est, w.model),
        dev=lambda w: cache_manifest(w.model, "dev"),
        utts=f"{OUT}/utts/dev.txt",
        refs=f"{OUT}/refs/dev",
        evaluator=f"{TP}/evaluation",
    output:
        grid=f"{OUT}/roseg/tune/{{est}}/{{model}}/grid.csv",
        selection=f"{OUT}/roseg/tune/{{est}}/{{model}}/selection.json",
    params:
        cache=lambda w, input: os.path.dirname(input.dev),
        grid=lambda w: "" if config["roseg"]["grid"] is None
        else "--grid '" + json.dumps(config["roseg"]["grid"]) + "'",
    shell:
        "python -m word_circles.roseg tune {input.subspace} {params.cache} {input.utts} {input.refs}"
        " {input.evaluator} {output.grid} {output.selection} {params.grid}"


rule roseg_report:
    input:
        subspace=lambda w: roseg_subspace(w.est, w.model),
        selection=f"{OUT}/roseg/tune/{{est}}/{{model}}/selection.json",
        dev=lambda w: cache_manifest(w.model, "dev"),
        test=lambda w: cache_manifest(w.model, "test"),
        dev_utts=f"{OUT}/utts/dev.txt",
        test_utts=f"{OUT}/utts/test.txt",
        dev_refs=f"{OUT}/refs/dev",
        test_refs=f"{OUT}/refs/test",
        kaldi_refs=f"{OUT}/refs_kaldi/test",
        evaluator=f"{TP}/evaluation",
    output:
        f"{OUT}/tables/roseg/{{est}}_{{model}}.csv",
    params:
        dev=lambda w, input: os.path.dirname(input.dev),
        test=lambda w, input: os.path.dirname(input.test),
        kind=lambda w: "trained" if w.model == "hubert-base-ls960" else "untrained",
    shell:
        "python -m word_circles.roseg report {input.subspace} {input.selection} {params.dev} {params.test}"
        " {input.dev_utts} {input.test_utts} {input.dev_refs} {input.test_refs} {input.evaluator} {output}"
        " --estimator {wildcards.est} --authors-check {params.kind} --kaldi-refs {input.kaldi_refs}"


rule roseg_predictions:
    """Dev predictions of the trained model's selected config and its theta = pi partner (Fig 6)."""
    input:
        subspace=roseg_subspace("{est}", "hubert-base-ls960"),
        selection=f"{OUT}/roseg/tune/{{est}}/hubert-base-ls960/selection.json",
        dev=cache_manifest("hubert-base-ls960", "dev"),
        utts=f"{OUT}/utts/dev.txt",
    output:
        f"{OUT}/predictions/roseg_{{est}}_{{variant}}_dev.json",
    wildcard_constraints:
        variant="selected|no_threshold",
    params:
        cache=lambda w, input: os.path.dirname(input.dev),
    shell:
        "python -m word_circles.roseg predict {input.subspace} {input.selection} {wildcards.variant}"
        " {params.cache} {input.utts} {output}"


rule dissim:
    input:
        manifest=lambda w: cache_manifest("hubert-base-ls960", w.split),
        utts=f"{OUT}/utts/{{split}}.txt",
    output:
        f"{OUT}/predictions/dissim_{{split}}.json",
    params:
        cache=lambda w, input: os.path.dirname(input.manifest),
        cfg=config["dissim"],
    shell:
        "python -m word_circles.dissim predict {params.cache} {input.utts} {output} --layer {params.cfg[layer]}"
        " --window {params.cfg[window]} --prominence {params.cfg[prominence]}"


rule dissim_retune:
    input:
        dev=cache_manifest("hubert-base-ls960", "dev"),
        test=cache_manifest("hubert-base-ls960", "test"),
        dev_utts=f"{OUT}/utts/dev.txt",
        test_utts=f"{OUT}/utts/test.txt",
        dev_refs=f"{OUT}/refs/dev",
        test_refs=f"{OUT}/refs/test",
        evaluator=f"{TP}/evaluation",
    output:
        f"{OUT}/tables/dissim_retune.csv",
    params:
        dev=lambda w, input: os.path.dirname(input.dev),
        test=lambda w, input: os.path.dirname(input.test),
    shell:
        "python -m word_circles.dissim retune {params.dev} {params.test} {input.dev_utts} {input.test_utts}"
        " {input.dev_refs} {input.test_refs} {input.evaluator} {output} --layer {config[dissim][layer]}"


rule fig6:
    input:
        expand(f"{OUT}/scores/fig6/{{method}}_dev.csv", method=FIG6_METHODS),
    output:
        f"{OUT}/figures/fig6.png",
    shell:
        "MPLBACKEND=Agg python -m word_circles.fig6_plot {output} {input}"


rule replication_report:
    input:
        notebook="notebooks/replication_report.py",
        roseg=expand(f"{OUT}/tables/roseg/{{est}}_{{model}}.csv", est=["cpca", "leace"], model=HUBERT_MODELS),
        sec2=f"{OUT}/tables/sec2_metrics.csv",
        fig3=f"{OUT}/figures/fig3.png",
        sec2_fits=expand(f"{OUT}/sec2/subspaces/{{model}}", model=HUBERT_MODELS),
        fig6=f"{OUT}/figures/fig6.png",
        gold=f"{OUT}/scores/gold_syllables_dev.csv",
        dissim=expand(f"{OUT}/scores/dissim_{{split}}.csv", split=["dev", "test"]),
        kaldi=f"{OUT}/scores/kaldi/dissim_test.csv",
        retune=[f"{OUT}/tables/dissim_retune.csv"] if config["dissim"]["retune"] else [],
    output:
        f"{OUT}/reports/replication_report.ipynb",
    params:
        parameters=lambda w: json.dumps({"out_dir": OUT, "random_models": HUBERT_MODELS[1:],
                                         "is_full_run": config["n_utts"] is None}),
    shell:
        "python -m word_circles.notebook {input.notebook} {output} --parameters '{params.parameters}'"
