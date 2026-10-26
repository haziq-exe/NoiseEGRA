"""A second diversity reading that sees whole stories: long-text embedding similarity.

The NoveltyBench judge reads at most 253 tokens and was trained on about 128, so its
whole-story verdicts may lean towards "same". This scores every story in full with
BAAI/bge-m3 (8,192-token context) and reports, per version:
  mean pairwise cosine similarity (lower = more varied), with a 95% interval for the
  difference from a reference version, and the near-duplicate rate (share of pairs above
  a similarity threshold).

    python scripts/score_embed.py --extra eval/think_window_arms.json eval/x.json --out DIR
"""
import argparse, json, sys
from pathlib import Path

import numpy as np


def load_arms(paths):
    arms = {}
    for p in paths:
        arms.update(json.loads(Path(p).read_text())["arms"])
    return {k: [s for s in v if isinstance(s, str) and s.strip()] for k, v in arms.items()}


def pair_sims(E):
    S = E @ E.T
    iu = np.triu_indices(len(E), 1)
    return S, S[iu]


def distinct_of_k(S, thr, k=10, subsets=500, seed=0):
    """Average number of distinct stories in random sets of k: a story counts as new
    unless its similarity to a story already kept in the set is at least ``thr``."""
    rng = np.random.default_rng(seed)
    n = len(S)
    if n < k:
        return float("nan")
    tot = 0
    for _ in range(subsets):
        ix = rng.choice(n, k, replace=False)
        kept = []
        for i in ix:
            if all(S[i, j] < thr for j in kept):
                kept.append(i)
        tot += len(kept)
    return tot / subsets


def boot_diff(Sa, Sb, n=2000, seed=0):
    """95% interval for mean pairwise similarity of a minus b, resampling stories."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        ia = rng.integers(0, len(Sa), len(Sa)); ib = rng.integers(0, len(Sb), len(Sb))
        def m(S, ix):
            sub = S[np.ix_(ix, ix)]
            mask = ~np.eye(len(ix), dtype=bool) & (ix[:, None] != ix[None, :])
            return sub[mask].mean()
        out.append(m(Sa, ia) - m(Sb, ib))
    return np.percentile(out, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="BAAI/bge-m3")
    ap.add_argument("--dup", type=float, default=0.9, help="near-duplicate similarity threshold")
    ap.add_argument("--thresholds", nargs="*", type=float, default=[0.8, 0.85],
                    help="similarity at or above which two stories count as the same, for "
                         "a distinct-of-10 count at each")
    ap.add_argument("--reference", nargs="*", default=[],
                    help="name fragments of the versions to compare against (default: 'untouched')")
    args = ap.parse_args()
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.model)
    model.max_seq_length = 8192
    arms = load_arms(args.extra)
    res, sims = {}, {}
    for name, stories in arms.items():
        E = model.encode(stories, batch_size=8, normalize_embeddings=True, show_progress_bar=False)
        S, p = pair_sims(np.asarray(E, dtype=np.float64))
        sims[name] = S
        res[name] = {"n": len(stories), "mean_sim": float(p.mean()),
                     "near_dup": float((p >= args.dup).mean()),
                     "distinct10": {str(t): distinct_of_k(S, t) for t in args.thresholds},
                     "pct_pairs_above": {str(t): float((p >= t).mean()) for t in args.thresholds}}
        print(f"  {name:42s} n {len(stories):3d}  mean similarity {p.mean():.4f}  "
              + "  ".join(f"distinct of 10 @{t:g}: {res[name]['distinct10'][str(t)]:.2f}"
                          for t in args.thresholds), flush=True)
    refs = args.reference or ["untouched"]
    for name in arms:
        prefix = name.split(" ")[0]
        for frag in refs:
            ref = next((r for r in arms if r.startswith(prefix) and frag in r and r != name), None)
            if ref is None:
                continue
            lo, hi = boot_diff(sims[name], sims[ref])
            d = res[name]["mean_sim"] - res[ref]["mean_sim"]
            res[name][f"vs {ref}"] = [d, float(lo), float(hi)]
            print(f"  {name:42s} vs {ref:28s} similarity {d:+.4f} [{lo:+.4f}, {hi:+.4f}]", flush=True)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "embed_diversity.json").write_text(json.dumps({"model": args.model, "dup": args.dup,
                                                          "arms": res}, indent=1))


if __name__ == "__main__":
    main()
