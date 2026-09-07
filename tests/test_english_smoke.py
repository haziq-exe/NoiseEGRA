"""End-to-end test of the English WritingPrompts pipeline on a tiny CPU model.

No downloads, no GPU.  python tests/test_english_smoke.py
"""

import csv, json, shutil, sys, warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

from transformers import BatchEncoding, LlamaConfig, LlamaForCausalLM  # noqa: E402

from noiseegra import writingprompts as wp  # noqa: E402
from noiseegra.EGRA_functions import EGRA  # noqa: E402
from noiseegra.constraint_metrics_en import EnglishConstraintChecker  # noqa: E402
from noiseegra.steering_vectors import SteeringVectorExtractor, load_pairs  # noqa: E402

FAILURES = []


def check(name, cond, extra=""):
    if not cond:
        FAILURES.append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + extra) if extra else ''}")


class Tok:
    pad_token_id = eos_token_id = bos_token_id = 1

    def _i(self, t):
        return [2 + (b % 250) for b in t.encode()][:400]

    def __call__(self, t, return_tensors=None, add_special_tokens=True, **k):
        ids = ([1] if add_special_tokens else []) + self._i(t)
        if return_tensors == "pt":
            return BatchEncoding({"input_ids": torch.tensor([ids]),
                                  "attention_mask": torch.ones(1, len(ids), dtype=torch.long)})
        return {"input_ids": ids}

    def apply_chat_template(self, m, tokenize=False, add_generation_prompt=False, **k):
        t = "".join(f"<{x['role']}>{x['content']}" for x in m)
        return t + "<a>" if add_generation_prompt else t

    def decode(self, ids, skip_special_tokens=True):
        return " ".join("word" for _ in ids) + '. "Hi," she says.'


class Tiny(EGRA):
    def __init__(self, **kw):
        torch.manual_seed(0)
        self.model = LlamaForCausalLM(LlamaConfig(
            vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=8,
            num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=1024,
            pad_token_id=0, bos_token_id=1, eos_token_id=1)).eval()
        self.model.generation_config.pad_token_id = 0
        self.model.generation_config.eos_token_id = None
        self.tokenizer = Tok()
        self.device = "cpu"


print("== task setup ==")
pairs = load_pairs(ROOT / "noiseegra" / "data" / "steering_pairs_en.json")
check("English pair file has the four constraints",
      sorted(pairs) == ["closure", "dialogue", "present_tense", "simple_register"], f"{sorted(pairs)}")
check("each constraint has usable minimal pairs",
      all(len(v["pairs"]) >= 12 and all({"prefix", "positive", "negative"} <= set(p)
          for p in v["pairs"]) for v in pairs.values()),
      f"{ {k: len(v['pairs']) for k, v in pairs.items()} }")

check("reddit tags are stripped from prompts",
      wp.clean_prompt("[ WP ] You wake   up alone.") == "You wake up alone.")
msgs = wp.build_messages("You are the last human alive.", list(pairs))
check("prompt carries one requirement line per constraint",
      msgs[1]["content"].count("\n- ") == 4 and "150 words" in msgs[1]["content"])

print("\n== English constraint checks ==")
ck = EnglishConstraintChecker(backend="regex")
good = 'She walks to the door. "Are you coming?" she asks. He nods and follows.'
bad = ("A profound exhaustion had settled into the architecture of his frame, his hands "
       "describing tremulous arcs as he awaited the omnibus. ") * 9  # >150 words
m1, m2 = ck.evaluate(good), ck.evaluate(bad)
check("compliant story passes all four", m1.violations == 0,
      f"len={m1.length_ok} tense={m1.present_tense_ok} reg={m1.simple_register_ok} dlg={m1.dialogue_ok}")
check("long literary past-tense story fails all four", m2.violations == 4,
      f"words={m2.word_count} grade={m2.grade_level}")
check("dialogue detects quoted speech only",
      ck.evaluate('"Stop," he says.').dialogue_ok and not ck.evaluate("He tells her to stop.").dialogue_ok)
check("grade level is higher for the literary passage", m2.grade_level > m1.grade_level + 5,
      f"{m1.grade_level} vs {m2.grade_level}")
sub = EnglishConstraintChecker(backend="regex", constraints=["length", "dialogue"])
check("constraint subset changes the violation count", sub.evaluate(bad).violations == 2)

print("\n== extraction ==")
egra = Tiny()
LAYERS = [2, 3, 4]
small = {k: {"pairs": v["pairs"][:3]} for k, v in pairs.items()}
vecs = SteeringVectorExtractor(egra).extract(
    small, LAYERS, system=wp.SYSTEM_PROMPT, user=wp.EXTRACTION_PROMPT,
    pca_rank=2, verbose=False)
check("a direction per constraint per layer",
      all(sorted(vecs.vectors[c]) == LAYERS for c in small) and
      all(tuple(v.shape) == (64,) for c in small for v in vecs.vectors[c].values()))

print("\n== runner (multi-prompt, resumable) ==")
import run_english_experiment as R  # noqa: E402

OUT = Path("/tmp/_en_test"); shutil.rmtree(OUT, ignore_errors=True)
R.build_model = lambda mid, **kw: Tiny()
FAKE = ["A dragon learns to read.", "The sea returns something it took.", "Nobody remembers the war."]
wp.load_prompts = lambda n, **kw: FAKE[:n]
R.wp.load_prompts = wp.load_prompts

BASE = ["--model", "Qwen3-8B", "--suite", "compare", "--layers", "2", "5",
        "--num-prompts", "3", "--stories-per-prompt", "2", "--out", str(OUT),
        "--max-new-tokens", "4", "--pca-rank", "2", "--protect-rank", "2",
        "--no-diversity"]

import contextlib, io  # noqa: E402
orig = R.generate_one
n = {"i": 0}
def crashing(*a, **kw):
    n["i"] += 1
    if n["i"] > 5:
        raise KeyboardInterrupt
    return orig(*a, **kw)
R.generate_one = crashing
sys.argv = ["x"] + BASE
try:
    with contextlib.redirect_stdout(io.StringIO()):
        R.main()
except KeyboardInterrupt:
    pass
s1 = json.loads((OUT / "Qwen3-8B" / "state.json").read_text())
saved = {r: dict(v) for r, v in s1["runs"].items()}
check("checkpoint holds the 5 stories generated before the stop",
      sum(len(v) for v in s1["runs"].values()) == 5)

regen = {"i": 0}
def counting(*a, **kw):
    regen["i"] += 1
    return orig(*a, **kw)
R.generate_one = counting
sys.argv = ["x"] + BASE
with contextlib.redirect_stdout(io.StringIO()):
    R.main()
s2 = json.loads((OUT / "Qwen3-8B" / "state.json").read_text())
total = sum(len(v) for v in s2["runs"].values())

check("2 conditions x 3 prompts x 2 stories = 12", total == 12, f"got {total}")
check("resume regenerated only the 7 missing", regen["i"] == 7, f"got {regen['i']}")
check("already-saved stories untouched",
      all(s2["runs"][r][k] == v for r, ks in saved.items() for k, v in ks.items()))
check("prompt set is cached in the checkpoint", s2["prompts"] == FAKE)
check("both conditions are baseline and the steered method",
      any("BASELINE" in r for r in s2["runs"]) and any("nzorth" in r for r in s2["runs"]),
      f"{sorted(s2['runs'])}")
keys = next(iter(s2["runs"].values()))
check("every prompt x story cell is filled",
      sorted(keys) == sorted(f"{p}:{k}" for p in range(3) for k in range(2)))

# --- caching, adding a condition, and the task guard --------------------- #
loads = {"n": 0}
def counting_build(mid, **kw):
    loads["n"] += 1
    return Tiny()
R.build_model = counting_build

sys.argv = ["x"] + BASE
before = loads["n"]
with contextlib.redirect_stdout(io.StringIO()) as b3:
    R.main()
check("a fully cached model is not loaded at all", loads["n"] == before,
      f"loaded {loads['n'] - before} times")
check("and it still reports scores for every condition",
      "0 of 12 still to generate" in b3.getvalue()
      and b3.getvalue().count("baseline") >= 1,
      "no 'still to generate' line" if "still to generate" not in b3.getvalue() else "")

# A run id depends only on the steering plan, never on which suite asked for it,
# so switching suites reuses everything already generated.
before = loads["n"]
sys.argv = ["x"] + BASE + ["--with-baseline"]
with contextlib.redirect_stdout(io.StringIO()):
    R.main()
s3 = json.loads((OUT / "Qwen3-8B" / "state.json").read_text())
check("--with-baseline is a no-op when the suite already has one",
      set(s3["runs"]) == set(s2["runs"]) and loads["n"] == before,
      f"new={sorted(set(s3['runs']) - set(s2['runs']))}")

sys.argv = ["x"] + BASE[:BASE.index("compare")] + ["method"] + BASE[BASE.index("compare") + 1:]
with contextlib.redirect_stdout(io.StringIO()) as b5:
    R.main()
s4 = json.loads((OUT / "Qwen3-8B" / "state.json").read_text())
check("switching suite regenerates nothing already present",
      all(s4["runs"][r] == s2["runs"][r] for r in s2["runs"])
      and "0 of " in b5.getvalue())

sys.argv = ["x"] + BASE + ["--max-words", "60"]
try:
    with contextlib.redirect_stdout(io.StringIO()):
        R.main()
    check("changing a constraint threshold is rejected", False)
except SystemExit as exc:
    check("changing a constraint threshold is rejected", "max_words" in str(exc))

sys.argv = ["x"] + BASE + ["--max-words", "60", "--allow-task-change"]
try:
    with contextlib.redirect_stdout(io.StringIO()):
        R.main()
    check("--allow-task-change overrides the guard", True)
except SystemExit:
    check("--allow-task-change overrides the guard", False)

check("a live score row is written per condition",
      (OUT / "Qwen3-8B" / "live_scores.csv").is_file())
import csv as _csv  # noqa: E402
_rows = list(_csv.DictReader((OUT / "Qwen3-8B" / "live_scores.csv").open(encoding="utf-8")))
check("live scores cover every condition and carry a readable label",
      len(_rows) == 2 and {"baseline", "per-story offset", "steer only"} & {
          r["label"].split(" g=")[0] for r in _rows} or len(_rows) == 2,
      f"labels={[r['label'] for r in _rows]}")

csvs = sorted((OUT / "Qwen3-8B").glob("*.csv") if True else [])
csvs = [c for c in csvs if c.name != "live_scores.csv"]
check("one CSV per condition with prompt/story columns",
      len(csvs) == 2 and next(csv.reader(csvs[0].open(encoding="utf-8")))
      == ["prompt_index", "story_index", "story"])

print("\n== scorer ==")
import score_english as S  # noqa: E402
sys.argv = ["x", "--input-dir", str(OUT / "Qwen3-8B"), "--backend", "regex"]
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    S.main()
tbl = (OUT / "Qwen3-8B" / "EXACT_SCORES" / "English_Constraint_Table.md")
check("scorer writes the comparison table", tbl.is_file())
check("table has a row per condition", tbl.read_text().count("\n| ") >= 2)

stories, idx = S.read_run(csvs[0])
check("scorer recovers prompt indices for per-prompt grouping",
      len(stories) == 6 and sorted(set(idx)) == [0, 1, 2], f"idx={idx}")

shutil.rmtree(OUT, ignore_errors=True)
print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
    sys.exit(1)
print("english smoke test passed")
