"""Throwaway oracle: which Fig 6 scoring semantics reproduces the gold-syllable curve?

Scores gold syllables per syllable-count group on dev-clean under the 2 x 2
candidates fixed by "Grilling: Fig 6 per-syllable-count scoring semantics":

  membership (i) strict start <= p <= end, (ii) +-1 frame
  scoring    (a) unmodified s-malan evaluator on per-group .list + TextGrid dirs
             (b) same boundary formulas, tokens counted only over the group's words

(b) token definition: ref tokens = (start, end) of each group word; seg tokens =
consecutive prediction pairs lying inside the same group word span.

Post-hoc candidate (c), added after all four missed: the evaluator's hit/OS/R/token
functions without its last-boundary drop (`eval_boundaries` deletes the final
pred/ref pair when they hit, which with ~2 refs per group-utterance inflates OS
toward n-1 instead of (n-1)/2), with `all` = groups 1-6 pooled (footnote 7
double-count) rather than a separate all-words group.

Run (the evaluator needs praat-textgrids, tqdm, numpy):

  uv run --with praat-textgrids --with tqdm python -I docs/research/fig6_gold_oracle.py \
      --tg-dir <flattened dev-clean TextGrids> --evaluator-dir <s-malan/evaluation @ c9074a4> \
      --work-dir <scratch> --targets docs/research/fig6-gold-targets.csv \
      --out docs/research/fig6-gold-oracle-results.csv
"""

import argparse
import csv
import importlib.util
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

GROUPS = ["all", "1", "2", "3", "4", "5", "6"]


def eval_boundaries_no_last_drop(seg, ref, tolerance):
    """`boundary_eval.eval_boundaries` (strict) minus the last-boundary deletion."""
    n_seg = n_ref = n_hit = 0
    for prediction, ground_truth in zip(seg, ref):
        prediction = list(prediction)
        n_seg += len(prediction)
        n_ref += len(ground_truth)
        for i_ref in ground_truth:
            for i, i_seg in enumerate(prediction):
                if abs(i_ref - i_seg) <= tolerance:
                    n_hit += 1
                    prediction.pop(i)
                    break
    return n_seg, n_ref, n_hit
METRICS = ["F1", "OS", "TokenF1", "R"]
TABLE_II_GOLD = {"F1": 85.5, "OS": 33.9, "TokenF1": 65.9, "R": 71.1}

INTERVAL_RE = re.compile(r'xmin = (\S+)\s+xmax = (\S+)\s+text = "(.*)"')


def parse_textgrid(path):
    """Return {tier name: [(xmin_str, xmax_str, text)]}, keeping the original time strings."""
    tiers = {}
    for chunk in re.split(r"item \[\d+\]:", path.read_text())[1:]:
        name = re.search(r'name = "(\w+)"', chunk).group(1)
        body = chunk.split("intervals: size", 1)[1]
        tiers[name] = INTERVAL_RE.findall(body)
    return tiers


def write_textgrid(path, boundaries):
    """Write a `words`-only TextGrid whose interval xmaxes are `boundaries` (time strings)."""
    edges = ["0.0"] + boundaries
    lines = [
        'File type = "ooTextFile"', 'Object class = "TextGrid"', "",
        "xmin = 0.0", f"xmax = {edges[-1]}", "tiers? <exists>", "size = 1", "item []:",
        "\titem [1]:", '\t\tclass = "IntervalTier"', '\t\tname = "words"',
        "\t\txmin = 0.0", f"\t\txmax = {edges[-1]}", f"\t\tintervals: size = {len(boundaries)}",
    ]
    for i, (lo, hi) in enumerate(zip(edges, edges[1:]), 1):
        lines += [f"\t\tintervals [{i}]:", f"\t\t\txmin = {lo}", f"\t\t\txmax = {hi}", '\t\t\ttext = "x"']
    path.write_text("\n".join(lines) + "\n")


def run_evaluator(evaluator, pred_dir, ref_dir):
    out = subprocess.run(
        [sys.executable, "-I", str(evaluator), str(pred_dir), str(ref_dir),
         "--alignment_type=words", "--tolerance", "1"],
        capture_output=True, text=True, check=True,
    ).stdout
    nums = dict(re.findall(r"([\w-]+(?: \w+)?): (-?[\d.e+-]+|-?inf)", out))
    return {
        "F1": 100 * float(nums["F1"]),
        "OS": 100 * float(nums["Over-segmentation"]),
        "TokenF1": 100 * float(nums["Token F1"]),
        "R": 100 * float(nums["R-value"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tg-dir", type=Path, required=True)
    ap.add_argument("--evaluator-dir", type=Path, required=True)
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--targets", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    evaluator = args.evaluator_dir / "boundary_eval.py"
    spec = importlib.util.spec_from_file_location("boundary_eval", evaluator)
    be = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(be)
    frame = lambda s: int(be.get_frame_num(float(s), ms_per_frame=20))

    # ---- parse + group words by syllable count ----
    utts = {}
    stats = Counter()
    for tg_path in sorted(args.tg_dir.glob("*.TextGrid")):
        tiers = parse_textgrid(tg_path)
        sylls = tiers["syllables"]
        words = []
        for lo, hi, text in tiers["words"]:
            if text == "":
                continue
            flo, fhi = float(lo), float(hi)
            n = sum(1 for slo, shi, st in sylls if st != "" and flo <= (float(slo) + float(shi)) / 2 <= fhi)
            stats["straddling_syllables"] += sum(
                1 for slo, shi, st in sylls if st != "" and (float(slo) < flo < float(shi) or float(slo) < fhi < float(shi))
            )
            stats["words_total"] += 1
            if not 1 <= n <= 6:
                stats[f"words_dropped_{'0' if n == 0 else '7plus'}"] += 1
                continue
            stats[f"words_{n}"] += 1
            words.append((lo, hi, n))
        utts[tg_path.stem] = dict(words=words, sylls=sylls, full=[hi for _, hi, _ in tiers["words"]])
    stats["words_all"] = sum(len(u["words"]) for u in utts.values())
    print(dict(stats))

    work = args.work_dir
    work.mkdir(parents=True, exist_ok=True)

    # ---- writer validation: regenerated full words tier must score identically ----
    pred_full, ref_regen = work / "full_pred", work / "full_ref_regen"
    for d in (pred_full, ref_regen):
        d.mkdir(exist_ok=True)
    for stem, u in utts.items():
        (pred_full / f"{stem}.list").write_text("\n".join(hi for _, hi, _ in u["sylls"]) + "\n")
        write_textgrid(ref_regen / f"{stem}.TextGrid", u["full"])
    orig = run_evaluator(evaluator, pred_full, args.tg_dir)
    regen = run_evaluator(evaluator, pred_full, ref_regen)
    print("full tier, original TextGrids:   ", orig)
    print("full tier, regenerated TextGrids:", regen)
    assert orig == regen, "TextGrid writer changes evaluator output"

    # ---- per-group scoring ----
    targets = {(r["group"], r["metric"]): float(r["value"]) for r in csv.DictReader(args.targets.open())}
    rows = []
    lists = {}
    for mem, slack in (("i", 0), ("ii", 1)):
        for g in GROUPS:
            pdir, rdir = work / f"pred_{mem}_{g}", work / f"ref_{g}"
            pdir.mkdir(exist_ok=True)
            rdir.mkdir(exist_ok=True)
            seg_list, ref_list, seg_tok, ref_tok = [], [], [], []
            for stem, u in utts.items():
                gw = [(lo, hi) for lo, hi, n in u["words"] if g == "all" or n == int(g)]
                if not gw:
                    continue
                bnd = {}
                for lo, hi in gw:
                    for s in (lo, hi):
                        if float(s) > 0:
                            bnd.setdefault(float(s), s)
                refs = [bnd[k] for k in sorted(bnd)]
                spans = [(frame(lo) - slack, frame(hi) + slack) for lo, hi in gw]
                preds = [hi for _, hi, _ in u["sylls"] if any(a <= frame(hi) <= b for a, b in spans)]
                (pdir / f"{stem}.list").write_text("\n".join(preds) + "\n")
                write_textgrid(rdir / f"{stem}.TextGrid", refs)

                # (b): in-process, mirroring boundary_eval.__main__ preprocessing
                pf = [frame(p) for p in (preds[1:] if float(preds[0]) == 0.0 else preds)]
                seg_list.append(pf)
                ref_list.append([frame(r) for r in refs])
                ref_tok.append([(frame(lo), frame(hi)) for lo, hi in gw])
                seg_tok.append([(x, y) for x, y in zip(pf, pf[1:]) if any(a <= x and y <= b for a, b in spans)])

            res_a = run_evaluator(evaluator, pdir, rdir)
            lists[(mem, g)] = (seg_list, ref_list)

            n_seg, n_ref, n_hit = be.eval_boundaries(seg_list, ref_list, tolerance=1.0)
            _, rec, f1 = be.get_p_r_f1(n_seg, n_ref, n_hit)
            os_ = be.get_os(n_seg, n_ref)
            t_seg = sum(map(len, seg_tok))
            t_ref = sum(map(len, ref_tok))
            t_hit = 0
            for st, rt in zip(seg_tok, ref_tok):
                rt = list(rt)
                for s0, s1 in st:
                    for k, (r0, r1) in enumerate(rt):
                        if abs(s0 - r0) <= 1 and abs(s1 - r1) <= 1:
                            t_hit += 1
                            rt.pop(k)
                            break
            res_b = {"F1": 100 * f1, "OS": 100 * os_, "TokenF1": 100 * be.get_p_r_f1(t_seg, t_ref, t_hit)[2],
                     "R": 100 * be.get_rvalue(os_, rec)}
            for m in ("F1", "OS", "R"):
                assert abs(res_a[m] - res_b[m]) < 1e-9, (mem, g, m, res_a[m], res_b[m])

            for sc, res in (("a", res_a), ("b", res_b)):
                for m in METRICS:
                    rows.append(dict(candidate=f"({sc})+({mem})", group=g, metric=m, value=round(res[m], 2),
                                     target=targets[(g, m)], dev=round(res[m] - targets[(g, m)], 2),
                                     n_utts=len(seg_list)))
            print(mem, g, {m: round(res_a[m], 1) for m in METRICS}, "b TokF1", round(res_b["TokenF1"], 1))

    def score_c(seg_list, ref_list):
        n_seg, n_ref, n_hit = eval_boundaries_no_last_drop(seg_list, ref_list, tolerance=1.0)
        _, rec, f1 = be.get_p_r_f1(n_seg, n_ref, n_hit)
        os_ = be.get_os(n_seg, n_ref)
        tok_f1 = be.get_p_r_f1(*be.eval_token_boundaries(seg_list, ref_list, tolerance=1.0))[2]
        return {"F1": 100 * f1, "OS": 100 * os_, "TokenF1": 100 * tok_f1, "R": 100 * be.get_rvalue(os_, rec)}

    for mem in ("i", "ii"):
        for g in GROUPS:
            if g == "all":
                seg_list = [s for k in GROUPS[1:] for s in lists[(mem, k)][0]]
                ref_list = [r for k in GROUPS[1:] for r in lists[(mem, k)][1]]
            else:
                seg_list, ref_list = lists[(mem, g)]
            res = score_c(seg_list, ref_list)
            for m in METRICS:
                rows.append(dict(candidate=f"(c)+({mem})", group=g, metric=m, value=round(res[m], 2),
                                 target=targets[(g, m)], dev=round(res[m] - targets[(g, m)], 2),
                                 n_utts=len(seg_list)))
            print("c", mem, g, {m: round(res[m], 1) for m in METRICS})

    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # ---- summary ----
    print("\ncandidate  matches/28  largest |dev| (group, metric)")
    for cand in sorted({r["candidate"] for r in rows}):
        rs = [r for r in rows if r["candidate"] == cand]
        ok = [abs(r["dev"]) <= (5 if r["metric"] == "OS" else 3) for r in rs]
        worst = max(rs, key=lambda r: abs(r["dev"]))
        print(f"{cand:9}  {sum(ok):2d}/28       {worst['dev']:+.1f} ({worst['group']}, {worst['metric']})")
    for cand in ("(a)+(ii)", "(c)+(i)"):
        alls = {r["metric"]: r["value"] for r in rows if r["candidate"] == cand and r["group"] == "all"}
        print(f"sanity {cand}: all-group F1/R above Table II gold row?",
              alls["F1"] > TABLE_II_GOLD["F1"] and alls["R"] > TABLE_II_GOLD["R"], alls)


if __name__ == "__main__":
    main()
