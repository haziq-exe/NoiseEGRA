"""Length-matched Vendi, distinct-k and plot skeletons. No model downloads."""

from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from noiseegra.diversity import (  # noqa: E402
    DiversityScorer, calibrate_threshold, distinct_k, group_by_prompt,
    read_run_csv, truncate_words, vendi_from_embeddings, word_count,
)
from noiseegra.embeddings import (  # noqa: E402
    DEFAULT_EMBEDDING_MODEL, PAPER_EMBEDDING_MODEL, describe, resolve_embedding_model,
)
from noiseegra.plot_skeleton import (  # noqa: E402
    PLOT_FIELDS, PlotCache, extract_skeletons, parse_skeleton, skeleton_text,
)

failures = []


def check(name, cond, detail=""):
    print(f"{'[ok] ' if cond else '[FAIL] '}{name}" + (f"  {detail}" if detail else ""))
    if not cond:
        failures.append(name)


# --------------------------------------------------------------------------- #
print("== text helpers ==")
check("word_count", word_count("one two  three\nfour") == 4)
check("truncate cuts", truncate_words("a b c d e", 3) == "a b c")
check("truncate no-op when None", truncate_words("a b c", None) == "a b c")
check("truncate keeps short texts whole", truncate_words("a b", 10) == "a b")

# --------------------------------------------------------------------------- #
print("\n== vendi ==")
d = 16
ident = np.tile(np.eye(1, d), (5, 1))
check("identical set scores 1", abs(vendi_from_embeddings(ident) - 1.0) < 1e-6,
      f"{vendi_from_embeddings(ident):.4f}")
orth = np.eye(5, d)
check("orthogonal set scores n", abs(vendi_from_embeddings(orth) - 5.0) < 1e-6,
      f"{vendi_from_embeddings(orth):.4f}")
mixed = np.vstack([np.eye(3, d), np.eye(3, d)])
check("3 duplicated pairs score 3", abs(vendi_from_embeddings(mixed) - 3.0) < 1e-6,
      f"{vendi_from_embeddings(mixed):.4f}")

# --------------------------------------------------------------------------- #
print("\n== distinct-k ==")
check("duplicates collapse to 1", distinct_k(ident, 0.8) == 1)
check("orthogonal stay separate", distinct_k(orth, 0.8) == 5)
check("3 pairs give 3 classes", distinct_k(mixed, 0.8) == 3, str(distinct_k(mixed, 0.8)))
check("single item is 1 class", distinct_k(np.eye(1, d), 0.8) == 1)

# a cluster of 3 near-identical vectors plus 2 far ones
near = np.eye(1, d).repeat(3, axis=0) + 0.01 * np.random.RandomState(0).randn(3, d)
near /= np.linalg.norm(near, axis=1, keepdims=True)
far = np.eye(2, d, k=5)
mix2 = np.vstack([near, far])
check("near cluster + 2 far = 3 classes", distinct_k(mix2, 0.9) == 3, str(distinct_k(mix2, 0.9)))
check("distinct-k never exceeds n", distinct_k(mix2, 1.1) == 5)

# --------------------------------------------------------------------------- #
print("\n== threshold calibration ==")
rs = np.random.RandomState(1)
emb = rs.randn(20, 32)
emb /= np.linalg.norm(emb, axis=1, keepdims=True)
pidx = [i // 5 for i in range(20)]
thr = calibrate_threshold(emb, pidx, 99.0)
cross = (emb @ emb.T)[np.array(pidx)[:, None] != np.array(pidx)[None, :]]
check("threshold is a high cross-prompt quantile", thr > np.median(cross), f"{thr:.3f}")
check("single prompt gives nan", np.isnan(calibrate_threshold(emb, [0] * 20)))

# --------------------------------------------------------------------------- #
print("\n== grouping and run files ==")
check("groups split by prompt", group_by_prompt(["a"] * 6, [0, 0, 1, 1, 2, 2]) ==
      [[0, 1], [2, 3], [4, 5]])
check("small groups dropped", group_by_prompt(["a"] * 3, [0, 0, 1], min_group=2) == [[0, 1]])

with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    hdr = td / "with_header.csv"
    with hdr.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["prompt_index", "story_index", "story"])
        w.writerow([0, 0, "story one"])
        w.writerow([1, 0, "story two"])
        w.writerow([1, 1, ""])
    st, pi = read_run_csv(hdr)
    check("header CSV read", st == ["story one", "story two"] and pi == [0, 1], f"{st} {pi}")

    bare = td / "bare.csv"
    bare.write_text("story a\nstory b\n", encoding="utf-8")
    st, pi = read_run_csv(bare)
    check("headerless CSV read as one group", st == ["story a", "story b"] and pi == [0, 0])

# --------------------------------------------------------------------------- #
print("\n== scorer with a stub embedder ==")


class StubST:
    """Maps a text to a unit vector determined by its first word, so texts that
    share an opening are identical after truncation but not before."""

    def encode(self, texts, **kw):
        out = []
        for t in texts:
            v = np.zeros(8)
            words = t.split()
            v[hash(words[0]) % 8 if words else 0] = 1.0
            # later words nudge it, so full texts differ where truncated ones do not
            for extra in words[1:]:
                v[hash(extra) % 8] += 0.6
            out.append(v / np.linalg.norm(v))
        return np.stack(out)


stories = ["alpha " + w for w in ("one two", "three four", "five six", "seven eight")]
sc = DiversityScorer(model=StubST(), truncate_to=1)
res = sc.score(stories, [0, 0, 1, 1], threshold=0.99)
check("truncation collapses the shared opening", abs(res.vendi_matched - 1.0) < 1e-6,
      f"raw={res.vendi_raw:.3f} matched={res.vendi_matched:.3f}")
check("raw vendi above matched", res.vendi_raw > res.vendi_matched + 0.1)
check("distinct-k uses truncated text", res.distinct_mean == 1.0, str(res.distinct_mean))
check("mean words recorded", res.mean_words == 3.0, str(res.mean_words))
check("group count", res.n_groups == 2)
check("no truncation keeps full text",
      sc.score(stories, [0, 0, 1, 1], threshold=0.99, truncate_to=None).vendi_matched
      == res.vendi_raw)

# --------------------------------------------------------------------------- #
print("\n== embedding registry ==")
check("registry key resolves", resolve_embedding_model("bge-m3") == "BAAI/bge-m3")
check("hf id passes through", resolve_embedding_model("BAAI/bge-m3") == "BAAI/bge-m3")
check("default is qwen3", resolve_embedding_model(None).startswith("Qwen/Qwen3-Embedding"))
check("paper model still reachable", PAPER_EMBEDDING_MODEL == "bge-m3")
check("describe mentions the id", "Qwen" in describe(DEFAULT_EMBEDDING_MODEL))

# --------------------------------------------------------------------------- #
print("\n== plot skeletons ==")
good = '{"setting": "a farm", "protagonist": "Mira", "goal": "find the goat", ' \
       '"obstacle": "a storm", "turning_point": "she hears bleating", "resolution": "reunited"}'
sk = parse_skeleton("Sure, here it is:\n" + good + "\nHope that helps.")
check("json parsed", sk["setting"] == "a farm" and sk["resolution"] == "reunited", str(sk))
loose = 'setting: a ship\nprotagonist: Ines\ngoal: reach shore'
sk2 = parse_skeleton(loose)
check("loose key: value parsed", sk2["protagonist"] == "Ines" and sk2["obstacle"] == "none",
      str(sk2))
check("garbage yields all-none", set(parse_skeleton("???").values()) == {"none"})
check("skeleton_text has a fixed field order",
      skeleton_text(sk).startswith("setting: a farm.") and
      all(f + ":" in skeleton_text(sk) for f in PLOT_FIELDS))


class StubExtractor:
    tag = "stub"

    def __init__(self):
        self.calls = 0

    def extract(self, stories):
        self.calls += len(stories)
        return [{f: f"{f}-{s[:4]}" for f in PLOT_FIELDS} for s in stories]


with tempfile.TemporaryDirectory() as td:
    cache_path = Path(td) / "plots.json"
    ex = StubExtractor()
    texts = ["story one", "story two", "story one"]
    a = extract_skeletons(texts, ex, PlotCache(cache_path))
    check("duplicate stories extracted once", ex.calls == 2, f"calls={ex.calls}")
    ex2 = StubExtractor()
    b = extract_skeletons(texts, ex2, PlotCache(cache_path))
    check("cache reused on a second pass", ex2.calls == 0 and a == b)

# --------------------------------------------------------------------------- #
print("\n== the length confound, and the fix ==")


class LengthSensitiveST:
    """Stands in for a real encoder's length behaviour: the embedding is the
    story's topic vector plus a per-story drift that grows with word count, so a
    set of long stories spreads out more than a set of short ones saying the same
    things. This is the effect that produced r(words, Vendi) = +0.94 in the real
    alpha sweep."""

    def _unit(self, text, dim=32):
        rs = np.random.RandomState(abs(hash(text)) % (2 ** 31))
        v = rs.randn(dim)
        return v / np.linalg.norm(v)

    def encode(self, texts, **kw):
        out = []
        for t in texts:
            words = t.split()
            topic = self._unit(" ".join(words[:2]))
            drift = self._unit(t) * (0.02 * len(words))
            v = topic + drift
            out.append(v / np.linalg.norm(v))
        return np.stack(out)


HEAD = 40  # the length-matched budget used below


def make_condition(n_topics, n_stories, words, tag):
    """``n_stories`` drawn from ``n_topics`` topics, each ``words`` words long.

    The first HEAD words depend only on the topic, so stories sharing a topic open
    identically. Anything past HEAD is unique to the story. A short condition is
    therefore genuinely ``n_topics``-diverse, and a long one only looks more
    diverse because of material a reader never reaches in the short version.
    """
    texts, prompts = [], []
    for s in range(n_stories):
        topic = s % n_topics
        head = f"topic{topic} begins " + " ".join(f"w{i}" for i in range(HEAD - 2))
        tail = " ".join(f"{tag}{s}u{i}" for i in range(max(0, words - HEAD)))
        texts.append((head + " " + tail).strip())
        prompts.append(0)
    return texts, prompts


sc2 = DiversityScorer(model=LengthSensitiveST(), truncate_to=HEAD)
short_t, short_p = make_condition(3, 10, HEAD, "s")
long_t, long_p = make_condition(3, 10, 200, "l")
r_short = sc2.score(short_t, short_p, threshold=0.9)
r_long = sc2.score(long_t, long_p, threshold=0.9)

check("the stub reproduces the confound: long looks more diverse raw",
      r_long.vendi_raw > r_short.vendi_raw + 0.3,
      f"short={r_short.vendi_raw:.2f} long={r_long.vendi_raw:.2f}")
check("length matching removes it",
      abs(r_long.vendi_matched - r_short.vendi_matched) < 0.02,
      f"short@N={r_short.vendi_matched:.2f} long@N={r_long.vendi_matched:.2f}")
check("distinct-k recovers the true topic count in both",
      r_short.distinct_mean == 3.0 and r_long.distinct_mean == 3.0,
      f"short={r_short.distinct_mean} long={r_long.distinct_mean}")


print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    raise SystemExit(1)
print("all diversity tests passed")
