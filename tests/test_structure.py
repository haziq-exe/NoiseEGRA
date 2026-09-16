"""Structural and syntactic diversity, and the order-q Vendi score.

Structural diversity must move with what happens in the stories and not with how
they are phrased; syntactic diversity the other way around. The order-q Vendi
must discount a lone outlier as q rises. spaCy with an English model is required
(the constraint scorer already requires it); if it is missing the structure
tests are skipped and say so, and the Vendi tests still run.

python tests/test_structure.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from noiseegra.diversity import vendi_from_embeddings  # noqa: E402

failures = []


def check(name, cond, detail=""):
    print(f"{'[ok] ' if cond else '[FAIL] '}{name}" + (f"  {detail}" if detail else ""))
    if not cond:
        failures.append(name)


print("== the order-q Vendi score ==")
rng = np.random.default_rng(0)
# 19 near-identical items plus one orthogonal outlier.
base = rng.normal(size=32)
bulk = base + 0.01 * rng.normal(size=(19, 32))
out = rng.normal(size=32)
out -= out @ base / (base @ base) * base
emb = np.vstack([bulk, out])
emb /= np.linalg.norm(emb, axis=1, keepdims=True)

v1 = vendi_from_embeddings(emb, q=1.0)
v2 = vendi_from_embeddings(emb, q=2.0)
vinf = vendi_from_embeddings(emb, q=float("inf"))
check("q=1 matches the default", abs(v1 - vendi_from_embeddings(emb)) < 1e-9)
check("higher q discounts the outlier", v1 > v2 > vinf,
      f"q1={v1:.3f} q2={v2:.3f} qinf={vinf:.3f}")
check("identical items score 1 at every q",
      all(abs(vendi_from_embeddings(np.ones((5, 4)) / 2.0, q=q) - 1.0) < 1e-6
          for q in (0.5, 1.0, 2.0, float("inf"))))
orth = np.eye(6)
check("orthogonal items score n at every q",
      all(abs(vendi_from_embeddings(orth, q=q) - 6.0) < 1e-6
          for q in (0.5, 1.0, 2.0, float("inf"))))
try:
    vendi_from_embeddings(emb, q=-1.0)
    check("negative q is rejected", False)
except ValueError:
    check("negative q is rejected", True)

print("\n== structural and syntactic diversity ==")
from noiseegra.structure import _Nlp  # noqa: E402

if _Nlp.get() is None:
    print("spaCy English model not available; skipping the structure tests.")
    print("all structure tests passed (vendi only)")
    sys.exit(0)

from noiseegra.structure import (  # noqa: E402
    content_lemma_sets, rarefied_vendi, structural_vendi, syntactic_vendi,
    _set_vectors, pos_trigram_vectors,
)

# Three retellings of the same events, worded differently, against three
# stories in which different things happen.
SAME_STORY = [
    "The cat chases a red ball across the garden. It pounces and the ball rolls away.",
    "A red ball rolls through the garden and the cat is chasing it, pouncing as it rolls.",
    "Across the garden the cat pounced on the red ball, and away the ball rolled.",
]
DIFF_STORY = [
    "The cat chases a red ball across the garden and pounces on it.",
    "A sandcastle stands on the beach until the tide washes it away at dusk.",
    "Grandmother bakes warm bread while snow falls outside the kitchen window.",
]

s_same = structural_vendi(SAME_STORY, truncate=None)
s_diff = structural_vendi(DIFF_STORY, truncate=None)
check("retellings score structurally alike", s_same < s_diff,
      f"same={s_same:.2f} diff={s_diff:.2f}")
check("different plots approach the group size", s_diff > 2.5, f"{s_diff:.2f}")

# One sentence template repeated with different content words, against varied
# sentence shapes with the same content words: syntax must separate them the
# opposite way structure does.
SAME_SYNTAX = [
    "The cat chases the ball. The cat eats the fish. The cat sees the bird.",
    "The dog fetches the stick. The dog drinks the water. The dog digs the hole.",
    "The girl reads the book. The girl draws the horse. The girl sings the song.",
]
sy = syntactic_vendi(SAME_SYNTAX, truncate=None)
st = structural_vendi(SAME_SYNTAX, truncate=None)
check("one template, different topics: structure above syntax", st > sy,
      f"structure={st:.2f} syntax={sy:.2f}")

# A story with no content words must not crash either score.
lemmas = content_lemma_sets(["...", "The cat sleeps."], truncate=None)
check("empty story yields an empty set, not a crash", lemmas[0] == frozenset())
v = vendi_from_embeddings(_set_vectors(lemmas))
check("empty story still scores", v == v and v >= 1.0, f"{v:.2f}")

print("\n== rarefied Vendi ==")
vecs = pos_trigram_vectors(DIFF_STORY * 4, truncate=None)
full = vendi_from_embeddings(vecs)
m, sd = rarefied_vendi(vecs, 6, draws=8)
check("rarefied below full-group score", m <= full + 1e-9, f"{m:.2f} vs {full:.2f}")
m2, _ = rarefied_vendi(vecs, len(vecs) * 2, draws=8)
check("group size above n falls back to the full score", abs(m2 - full) < 1e-9)

print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    raise SystemExit(1)
print("all structure tests passed")
