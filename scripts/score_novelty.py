#!/usr/bin/env python
"""NoveltyBench's same-story judge over every arm's coherent stories.

    python scripts/score_novelty.py <run-dir> [--limit N] [--subsets 500]

``<run-dir>`` is a run's output directory: the merged ``<model>/state.json`` a
sharded run leaves, or its ``shard*/`` directories, or a ``state.json`` itself.
Each arm's stories go through the coherence checks first, the same ones every
table and the story-review workbook use, and only the stories that pass are
judged. Results are written to ``<run-dir>/novelty.json`` and printed.

The judge is NoveltyBench's v1.0 classifier (DeBERTa-v3-large fine-tuned on
human same/different labels, ``yimingzhang/deberta-v3-large-generation-
similarity``), exactly as ``eval/creativity_metrics.ipynb`` runs it: the first
128 tokens of each story, a pair counted the same story above 0.102. On a GPU
every pair is scored in half precision and every pair near 0.102 is scored again
at full precision, with a random sample checked too, so no verdict rests on
rounding; the log reports how many the check would have changed.
It judges openings: the maintainers' later v1.1 reads whole responses.

Per arm:

- same-story pairs: the share of all pairs of its coherent stories the judge
  calls the same story. This project's measure, not one NoveltyBench reports.
- distinct of 10: NoveltyBench's distinct-k. Ten stories in a random order, each
  joining the first class whose first member the judge matches it with; the
  number of classes, averaged over random ten-story subsets.
- 95% intervals on each arm's difference from the untouched model and from
  top-p sampling at temperature 1.8, from 400 random half-size subsets of each
  arm. The spread of a statistic over random halves of a sample approximates
  its spread over fresh samples of the full size.

With more than one GPU the arms are split between them, one process each.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

THRESHOLD, MAX_TOKENS = 0.102, 128
JUDGE = "yimingzhang/deberta-v3-large-generation-similarity"
TOKENIZER = "microsoft/deberta-v3-large"


# --------------------------------------------------------------------------- #
#  Stories                                                                     #
# --------------------------------------------------------------------------- #

def state_files(path: Path) -> list:
    if path.is_file():
        return [path]
    merged = [p for p in path.glob("*/state.json") if not p.parent.name.startswith("shard")]
    if merged:
        return merged
    return sorted(path.glob("shard*/*/state.json"))


def arms(path: Path, extra: str = "") -> dict:
    """{run id: [story, ...]} in story order, every checkpoint pooled.

    ``extra`` is a JSON file of arms from earlier runs, ``{"arms": {run id:
    [story, ...]}}``, judged alongside so the new arms have them to be compared
    with. A run id the run itself wrote takes precedence.
    """
    runs: dict = {}
    for f in state_files(path):
        for rid, cells in json.loads(f.read_text()).get("runs", {}).items():
            runs.setdefault(rid, {}).update(cells)
    out = {rid: [c[k] for k in sorted(c, key=lambda x: int(x.split(":")[1]))]
           for rid, c in runs.items() if c}
    if extra:
        for rid, texts in json.loads(Path(extra).read_text())["arms"].items():
            out.setdefault(rid, list(texts))
    return out


def coherent(texts: list) -> list:
    from noiseegra.coherence import CoherenceFilter
    filt = CoherenceFilter()
    reps = [filt.check(t) for t in texts]
    return [r.text for r in reps if r.ok]


# --------------------------------------------------------------------------- #
#  The judge                                                                   #
# --------------------------------------------------------------------------- #

def load_judge(device: str):
    try:
        import sentencepiece  # noqa: F401
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                        "sentencepiece", "protobuf"], check=False)
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    model = AutoModelForSequenceClassification.from_pretrained(JUDGE).to(device).eval()
    return tok, model, torch


def _pair_probs(pairs, enc, tok, model, torch, device, batch: int, half: bool) -> list:
    """The judge's same-story probability for each (i, j), story i first."""
    out = []
    for s in range(0, len(pairs), batch):
        ids, tts = [], []
        for i, j in pairs[s:s + batch]:
            x = [tok.cls_token_id] + enc[i] + [tok.sep_token_id]
            first = len(x)
            x += enc[j] + [tok.sep_token_id]
            ids.append(x)
            tts.append([0] * first + [1] * (len(x) - first))
        width = max(len(x) for x in ids)
        att = [[1] * len(x) + [0] * (width - len(x)) for x in ids]
        ids = [x + [tok.pad_token_id] * (width - len(x)) for x in ids]
        tts = [t + [0] * (width - len(t)) for t in tts]
        with torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=half):
            lg = model(input_ids=torch.tensor(ids, device=device),
                       token_type_ids=torch.tensor(tts, device=device),
                       attention_mask=torch.tensor(att, device=device)).logits
        out += lg.float().softmax(-1)[:, 1].cpu().tolist()
    return out


# A pair whose half-precision probability is this close to the threshold is
# judged again at full precision, so no verdict rests on rounding. Half
# precision moves the probability by about 1e-3; the margin is thirty times that.
MARGIN = 0.03
CHECK_PAIRS = 400


def same_matrix(texts: list, tok, model, torch, device: str, batch: int = 64,
                report: dict = None) -> np.ndarray:
    """n x n: whether the judge calls stories i and j the same story.

    On a GPU every pair is scored in half precision, three to eight times
    faster on a T4, and then every pair near the threshold, plus a random
    sample of the rest, is scored again at full precision. The full-precision
    verdict is the one kept, and ``report`` records how many sampled verdicts
    half precision alone would have got wrong -- the evidence that it did not
    matter, measured on these stories.
    """
    enc = [tok.encode(t, truncation=True, max_length=MAX_TOKENS, add_special_tokens=False)
           for t in texts]
    n = len(texts)
    pairs = [(i, j) for i in range(n) for j in range(i)]
    half = str(device).startswith("cuda")
    probs = _pair_probs(pairs, enc, tok, model, torch, device, batch, half)
    if half and pairs:
        near = [k for k, p in enumerate(probs) if abs(p - THRESHOLD) < MARGIN]
        rng = random.Random(0)
        rest = [k for k in range(len(pairs)) if abs(probs[k] - THRESHOLD) >= MARGIN]
        sample = rng.sample(rest, min(CHECK_PAIRS, len(rest)))
        redo = near + sample
        full = _pair_probs([pairs[k] for k in redo], enc, tok, model, torch, device,
                           batch, False)
        flipped = sum((probs[k] > THRESHOLD) != (f > THRESHOLD)
                      for k, f in zip(sample, full[len(near):]))
        near_flipped = sum((probs[k] > THRESHOLD) != (f > THRESHOLD)
                           for k, f in zip(near, full[:len(near)]))
        for k, f in zip(redo, full):
            probs[k] = f
        if report is not None:
            report.update(pairs=len(pairs), near=len(near), near_flipped=near_flipped,
                          checked=len(sample), checked_flipped=flipped)
    same = np.eye(n, dtype=bool)
    for (i, j), p in zip(pairs, probs):
        same[i, j] = same[j, i] = p > THRESHOLD
    return same


# --------------------------------------------------------------------------- #
#  Measures                                                                    #
# --------------------------------------------------------------------------- #

def share_same(same: np.ndarray) -> float:
    n = same.shape[0]
    if n < 2:
        return float("nan")
    iu = np.triu_indices(n, 1)
    return float(same[iu].mean())


def distinct_of(same: np.ndarray, order) -> int:
    """Classes among ``order``: each joins the first class whose head it matches."""
    heads: list = []
    for i in order:
        if not any(same[i, h] for h in heads):
            heads.append(i)
    return len(heads)


def distinct_k(same: np.ndarray, k: int = 10, subsets: int = 500, seed: int = 0) -> float:
    n = same.shape[0]
    k = min(k, n)
    r = random.Random(seed)
    return float(np.mean([distinct_of(same, r.sample(range(n), k)) for _ in range(subsets)]))


def half_draws(same: np.ndarray, draws: int = 400, inner: int = 50, k: int = 10,
               seed: int = 0):
    """Same-story share and distinct-k over random half-size subsets."""
    n = same.shape[0]
    m = max(2, n // 2)
    rng = np.random.default_rng(seed)
    shares, dist = [], []
    for _ in range(draws):
        idx = rng.choice(n, m, replace=False)
        sub = same[np.ix_(idx, idx)]
        shares.append(share_same(sub))
        kk = min(k, m)
        dist.append(np.mean([distinct_of(sub, rng.choice(m, kk, replace=False))
                             for _ in range(inner)]))
    return np.array(shares), np.array(dist)


def interval(a: np.ndarray, b: np.ndarray, seed: int = 1) -> list:
    d = a - b[np.random.default_rng(seed).permutation(len(b))]
    return [float(a.mean() - b.mean()), float(np.percentile(d, 2.5)),
            float(np.percentile(d, 97.5))]


# --------------------------------------------------------------------------- #
#  Running                                                                     #
# --------------------------------------------------------------------------- #

def score(path: Path, rids: list, limit: int, subsets: int, device: str,
          extra: str = "") -> dict:
    stories = arms(path, extra)
    tok, model, torch = load_judge(device)
    out = {}
    for rid in rids:
        texts = stories[rid][:limit] if limit else stories[rid]
        kept = coherent(texts)
        t0 = time.time()
        rep = {}
        same = same_matrix(kept, tok, model, torch, device, report=rep)
        out[rid] = {"stories": len(texts), "coherent": len(kept),
                    "same": same.astype(int).tolist(),
                    "same_story_share": share_same(same),
                    "distinct10": distinct_k(same, subsets=subsets),
                    "precision_check": rep}
        extra = ""
        if rep:
            extra = (f"; half precision rechecked on {rep['near']} pairs near the line "
                     f"({rep['near_flipped']} changed) and {rep['checked']} others "
                     f"({rep['checked_flipped']} would have changed)")
        print(f"  judged {rid[-60:]}: {len(kept)} coherent, "
              f"{out[rid]['same_story_share']:.1%} same-story pairs, distinct of 10 "
              f"{out[rid]['distinct10']:.2f}  ({time.time() - t0:.0f}s{extra})", flush=True)
    return out


def summarise(path: Path, judged: dict) -> dict:
    from noiseegra.run_labels import label_run

    base = next((r for r in judged if r.endswith("__BASELINE")), None)
    topp = next((r for r in judged if "BASELINE__temp1p8__topp0p95" in r), None)
    draws = {r: half_draws(np.array(v["same"], dtype=bool)) for r, v in judged.items()}
    rows = {}
    for rid, v in judged.items():
        row = {k: v[k] for k in ("stories", "coherent", "same_story_share", "distinct10")}
        row["precision_check"] = v.get("precision_check", {})
        row["label"] = label_run(rid).text
        for name, ref in (("untouched", base), ("top_p", topp)):
            if ref and ref != rid:
                # The difference itself from the full sets; only its spread from
                # the half-size draws.
                s = interval(draws[rid][0], draws[ref][0])
                s[0] = v["same_story_share"] - judged[ref]["same_story_share"]
                d = interval(draws[rid][1], draws[ref][1])
                d[0] = v["distinct10"] - judged[ref]["distinct10"]
                row[f"same_vs_{name}"], row[f"distinct_vs_{name}"] = s, d
        rows[rid] = row
    (path if path.is_dir() else path.parent).joinpath("novelty.json").write_text(
        json.dumps({"judge": JUDGE, "threshold": THRESHOLD, "tokens": MAX_TOKENS,
                    "arms": rows, "pairs": {r: v["same"] for r, v in judged.items()}},
                   indent=1))
    ci = lambda x: f"{x[0]:+.1%} [{x[1]:+.1%}, {x[2]:+.1%}]"
    cd = lambda x: f"{x[0]:+.2f} [{x[1]:+.2f}, {x[2]:+.2f}]"
    print("\n" + "=" * 100)
    print("NoveltyBench judge (v1.0 classifier, first 128 tokens), coherent stories only")
    print("=" * 100)
    for rid, r in rows.items():
        print(f"{r['label'][:90]}\n    {r['coherent']}/{r['stories']} coherent | same-story pairs "
              f"{r['same_story_share']:.1%} | distinct of 10 {r['distinct10']:.2f}")
        for name in ("untouched", "top_p"):
            if f"same_vs_{name}" in r:
                print(f"    vs {name:9s} same-story {ci(r[f'same_vs_{name}'])}   "
                      f"distinct {cd(r[f'distinct_vs_{name}'])}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path")
    ap.add_argument("--limit", type=int, default=0, help="first N stories per arm (testing)")
    ap.add_argument("--subsets", type=int, default=500)
    ap.add_argument("--part", default="", help="i/n: judge every n-th arm from i (internal)")
    ap.add_argument("--device", default="")
    ap.add_argument("--extra", default="",
                    help="JSON of earlier arms to judge alongside: {\"arms\": {run id: [stories]}}")
    args = ap.parse_args()
    path = Path(args.path)
    if args.extra and not Path(args.extra).is_absolute():
        args.extra = str((ROOT / args.extra) if (ROOT / args.extra).is_file() else Path(args.extra))
    rids = sorted(arms(path, args.extra))
    if not rids:
        raise SystemExit(f"no stories under {path}")

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    gpus = torch.cuda.device_count() if device == "cuda" else 0
    if args.part:
        i, n = (int(x) for x in args.part.split("/"))
        judged = score(path, rids[i::n], args.limit, args.subsets, device, args.extra)
        (path if path.is_dir() else path.parent).joinpath(f"novelty_part{i}.json").write_text(
            json.dumps(judged))
        return
    if gpus > 1:
        print(f"judging {len(rids)} arms on {gpus} GPUs", flush=True)
        procs = [subprocess.Popen(
            [sys.executable, "-u", __file__, str(path), "--part", f"{i}/{gpus}",
             "--limit", str(args.limit), "--subsets", str(args.subsets),
             *(["--extra", args.extra] if args.extra else [])],
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(i))) for i in range(gpus)]
        codes = [p.wait() for p in procs]
        judged = {}
        for i in range(gpus):
            f = (path if path.is_dir() else path.parent) / f"novelty_part{i}.json"
            if f.is_file():
                judged.update(json.loads(f.read_text()))
                f.unlink()
        if any(codes):
            print(f"  some judging processes failed: exit codes {codes}", flush=True)
    else:
        judged = score(path, rids, args.limit, args.subsets, device, args.extra)
    if judged:
        summarise(path, judged)


if __name__ == "__main__":
    main()
