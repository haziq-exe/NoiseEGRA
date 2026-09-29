"""How different the stories are at their opening, middle and ending, not only their start.

    python scripts/segment_novelty.py PATH [--group 10]

``score_novelty.py`` judges the first 128 tokens of each story, about the first
half of a 150-word story. Noise that acts while the story is written, after the
prompt has set its course, can only show up later in the story, so the judge
never saw it. This scores each third of every coherent story (headings and
replies to the reader removed first) with the same NoveltyBench judge.

To keep it cheap, stories are compared only within fixed groups of ``--group``
consecutive coherent stories (45 pairs for a group of 10 instead of every pair
of the arm): the distinct-of-10 measure is itself a count within ten stories, so
each group gives one draw of it. Segments:

    whole    the story as score_novelty.py sees it (first 128 tokens)
    opening  its first third of words
    middle   its second third
    ending   its last third

Writes ``segment_novelty.json`` next to the stories: per arm and segment, the
distinct count and same-story share of every group.
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
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import score_novelty as S  # noqa: E402

SEGMENTS = ("whole", "opening", "middle", "ending")


def segments(text: str) -> dict:
    words = text.split()
    n = len(words)
    a, b = n // 3, 2 * n // 3
    return {"whole": text, "opening": " ".join(words[:a]),
            "middle": " ".join(words[a:b]), "ending": " ".join(words[b:])}


def grouped_same(texts: list, group: int, tok, model, torch, device: str) -> list:
    """For each full group of ``group`` consecutive texts, its same-story matrix."""
    enc = [tok.encode(t, truncation=True, max_length=S.MAX_TOKENS, add_special_tokens=False)
           for t in texts]
    blocks = [list(range(s, s + group)) for s in range(0, len(texts) - group + 1, group)]
    pairs = [(i, j) for b in blocks for x, i in enumerate(b) for j in b[:x]]
    half = str(device).startswith("cuda")
    probs = S._pair_probs(pairs, enc, tok, model, torch, device, 64, half)
    if half and pairs:
        # Borderline verdicts again at full precision, as score_novelty does.
        near = [k for k, p in enumerate(probs) if abs(p - S.THRESHOLD) < S.MARGIN]
        full = S._pair_probs([pairs[k] for k in near], enc, tok, model, torch, device, 64, False)
        for k, f in zip(near, full):
            probs[k] = f
    same = {}
    for (i, j), p in zip(pairs, probs):
        same[(i, j)] = same[(j, i)] = p > S.THRESHOLD
    out = []
    for b in blocks:
        m = np.eye(len(b), dtype=bool)
        for x, i in enumerate(b):
            for y, j in enumerate(b):
                if x != y:
                    m[x, y] = same[(i, j)]
        out.append(m)
    return out


def score_arm(texts: list, group: int, tok, model, torch, device: str) -> dict:
    kept = [S.strip_opening(t) for t in S.coherent(texts)]
    segs = [segments(t) for t in kept]
    res = {"stories": len(texts), "coherent": len(kept)}
    rng = random.Random(0)
    if group <= 0:
        # Every pair, for small pilots: the whole matrix is kept so intervals
        # can come from half-size draws, as score_novelty reports them.
        for name in SEGMENTS:
            m = S.same_matrix([s[name] for s in segs], tok, model, torch, device)
            res[name] = {"distinct10": S.distinct_k(m), "same_share": S.share_same(m),
                         "same": m.astype(int).tolist()}
        return res
    for name in SEGMENTS:
        mats = grouped_same([s[name] for s in segs], group, tok, model, torch, device)
        # The class count depends a little on the order stories are taken in:
        # averaged over 20 orders.
        dist = [float(np.mean([S.distinct_of(m, rng.sample(range(len(m)), len(m)))
                               for _ in range(20)])) for m in mats]
        res[name] = {"distinct": dist, "same": [S.share_same(m) for m in mats]}
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path")
    ap.add_argument("--group", type=int, default=10,
                    help="stories per comparison group; 0 compares every pair (small pilots)")
    ap.add_argument("--part", default="", help="i/n: every n-th arm from i (internal)")
    args = ap.parse_args()
    path = Path(args.path)
    arms = S.arms(path)
    rids = sorted(arms)
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpus = torch.cuda.device_count() if device == "cuda" else 0
    where = path if path.is_dir() else path.parent
    if gpus > 1 and not args.part:
        procs = [subprocess.Popen([sys.executable, "-u", __file__, str(path), "--group",
                                   str(args.group), "--part", f"{i}/{gpus}"],
                                  env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(i)))
                 for i in range(gpus)]
        codes = [p.wait() for p in procs]
        if any(codes):
            raise SystemExit(f"a part failed: {codes}")
        merged = {}
        for i in range(gpus):
            merged.update(json.loads((where / f"segment_part{i}.json").read_text()))
        (where / "segment_novelty.json").write_text(json.dumps(merged))
        print(f"wrote {where / 'segment_novelty.json'} ({len(merged)} arms)", flush=True)
        return
    if args.part:
        i, n = (int(x) for x in args.part.split("/"))
        rids = rids[i::n]
    tok, model, torch = S.load_judge(device)
    out = {}
    for rid in rids:
        t0 = time.time()
        out[rid] = r = score_arm(arms[rid], args.group, tok, model, torch, device)
        print(f"  {rid[-50:]}: {r['coherent']} coherent | " + "  ".join(
            f"{s} {np.mean(r[s]['distinct']) if 'distinct' in r[s] else r[s]['distinct10']:.2f}"
            for s in SEGMENTS)
              + f"  ({time.time() - t0:.0f}s)", flush=True)
    name = f"segment_part{args.part.split('/')[0]}.json" if args.part else "segment_novelty.json"
    (where / name).write_text(json.dumps(out))


if __name__ == "__main__":
    main()
