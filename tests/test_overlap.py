"""The subspace-overlap measurement, on inputs whose answer is known in advance.

python tests/test_overlap.py
"""

import io, contextlib, shutil, sys, warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

from noiseegra.activation_basis import StoryAxes  # noqa: E402
from noiseegra.steering_vectors import SteeringVectorSet  # noqa: E402
from subspace_overlap import main as overlap_main, overlap, protected_basis  # noqa: E402

FAILURES = []


def check(name, cond, extra=""):
    if not cond:
        FAILURES.append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + extra) if extra else ''}")


DIM = 512
torch.manual_seed(0)
P = torch.linalg.qr(torch.randn(DIM, 20))[0]

print("== the measurement itself ==")
s, c = overlap(P[:, :5] @ torch.linalg.qr(torch.randn(5, 5))[0], P)
check("a basis built inside the protected span reads 100%", abs(s - 1) < 1e-4, f"{s:.4f}")
check("and its principal cosine is 1", abs(c - 1) < 1e-4, f"{c:.4f}")

C = torch.linalg.qr(torch.randn(DIM, 8))[0]
C = torch.linalg.qr(C - P @ (P.t() @ C))[0]
s, c = overlap(C, P)
check("one built in the complement reads 0%", s < 1e-6, f"{s:.3g}")
check("and its principal cosine is 0", c < 1e-3, f"{c:.3g}")

s, _ = overlap(torch.linalg.qr(torch.randn(DIM, 24))[0], P)
check("a random basis sits at chance, k/dim", abs(s - 20 / DIM) < 0.02,
      f"{s:.4f} vs {20 / DIM:.4f}")

print("\n== the protected span is built the way the plan builds it ==")
NAMES = ["present_tense", "simple_register", "dialogue", "terse", "varied_openers"]
LAYERS = [14, 15]
vecs = SteeringVectorSet(
    vectors={n: {l: torch.randn(DIM) for l in LAYERS} for n in NAMES},
    components={n: {l: torch.linalg.qr(torch.randn(DIM, 8))[0] for l in LAYERS}
                for n in NAMES},
)
pb = protected_basis(vecs, NAMES, LAYERS[0], 8)
check("five directions plus eight components each span 45 dimensions",
      pb.shape == (DIM, 45), str(tuple(pb.shape)))
check("and the span is orthonormal",
      float((pb.t() @ pb - torch.eye(45)).abs().max()) < 1e-4)
check("protect-rank 0 leaves only the five directions",
      protected_basis(vecs, NAMES, LAYERS[0], 0).shape[1] == 5)

print("\n== end to end on files ==")
tmp = Path("/tmp/_overlap_test")
shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir(parents=True)
vecs.save(tmp / "steering_Tiny.pt")
# A basis deliberately half inside the protected span, so the printed number has
# a value this test knows: 24 directions, 12 of them taken from the span itself.
pb15 = protected_basis(vecs, NAMES, LAYERS[1], 8)
mixed = {}
for l, p in ((LAYERS[0], pb), (LAYERS[1], pb15)):
    half = p[:, :12] @ torch.linalg.qr(torch.randn(12, 12))[0]
    rest = torch.randn(DIM, 12)
    rest = rest - p @ (p.t() @ rest)
    mixed[l] = torch.linalg.qr(torch.cat([half, torch.linalg.qr(rest)[0]], dim=1))[0]
torch.save(StoryAxes(basis=mixed, mean={l: torch.zeros(DIM) for l in LAYERS},
                     explained=0.9, n_stories=32),
           tmp / "actpcs_story_Tiny.pt")

argv = sys.argv
sys.argv = ["subspace_overlap.py", "--input-dir", str(tmp), "--protect-rank", "8"]
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    overlap_main()
sys.argv = argv
out = buf.getvalue()
check("the script runs and reports both layers",
      out.count("   story ") >= 2, out[-400:])
rows = "\n     ".join(l for l in out.splitlines() if l.strip().startswith("story"))
check("a half-inside basis reads about 50%", " 50.00%" in out, rows)
check("chance is reported next to it", " 8.79%" in out, rows)
shutil.rmtree(tmp, ignore_errors=True)

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
    sys.exit(1)
print("all overlap tests passed")
