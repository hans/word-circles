configfile: "config.yaml"

OUT = config["out_dir"]
DATA = config["data_dir"]
TP = config["third_party_dir"]
SPLITS = ["dev", "test", "train"]


rule all:
    input:
        f"{OUT}/tables/gold_syllables_dev.csv",
        expand(f"{OUT}/utts/{{split}}.txt", split=SPLITS),


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
        evaluator=f"{TP}/evaluation",
    output:
        scores=f"{OUT}/tables/gold_syllables_{{split}}.csv",
        empty=f"{OUT}/tables/gold_syllables_{{split}}.empty.json",
    shell:
        "python -m word_circles.gold {input.refs} {input.evaluator} {output.scores} {output.empty}"
