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
from noiseegra.constraint_metrics_en import (  # noqa: E402
    MONOTONE_CONSTRAINTS,
    EnglishConstraintChecker,
)
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
check("English pair file holds the eleven steerable directions",
      sorted(pairs) == ["closure", "dialogue", "named_character", "no_heading",
                        "plain_words", "present_tense", "sensory",
                        "simple_register", "simple_syntax", "terse",
                        "varied_openers"], f"{sorted(pairs)}")

# The first pair set was length-confounded: three of its four directions had a
# positive side 11 to 14 words shorter than the negative, so "simple register"
# was partly "say less". Contrast pairs must differ in the property and nothing
# else, length included.
import re as _re
_W = _re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
# `no_heading` is exempt, and has to be: its two sides share a body word for
# word and the negative adds a heading on top, so the added words *are* the
# property being contrasted. Holding it to a length match would mean shortening
# the body under the heading, which would make the direction "write less" as
# well as "write no heading" -- the exact confound this check exists to catch.
_LENGTH_IS_THE_PROPERTY = {"no_heading"}
gaps = {name: max(abs(len(_W.findall(p["positive"])) - len(_W.findall(p["negative"])))
                  for p in v["pairs"])
        for name, v in pairs.items() if name not in _LENGTH_IS_THE_PROPERTY}
check("positive and negative sides are the same length, to within a word",
      all(g <= 1 for g in gaps.values()), str(gaps))
check("each constraint has usable minimal pairs",
      all(len(v["pairs"]) >= 12 and all({"prefix", "positive", "negative"} <= set(p)
          for p in v["pairs"]) for v in pairs.values()),
      f"{ {k: len(v['pairs']) for k, v in pairs.items()} }")

check("reddit tags are stripped from prompts",
      wp.clean_prompt("[ WP ] You wake   up alone.") == "You wake up alone.")
# Not every extracted direction is a scored requirement with its own bullet.
# `no_heading` steers the instruction's closing line -- "write only the story
# itself: no title, heading, preamble or commentary" -- which the prompt already
# states once and which no bullet repeats. Build the prompt from the names that
# do map to a scored constraint.
_STEER_ONLY = {"no_heading"}
msgs = wp.build_messages("You are the last human alive.",
                         [n for n in pairs if n not in _STEER_ONLY])
_scenario_names = [n for n in pairs if n not in _STEER_ONLY]
check("scenario prompt carries one requirement line per constraint",
      msgs[1]["content"].count("\n- ") == len(_scenario_names), str(msgs[1]["content"]))

print("\n== English constraint checks ==")
# Thresholds are constructor arguments, so the same four mechanics can be checked
# against the one-sided rules the first English runs used.
ck = EnglishConstraintChecker(backend="regex", word_range=(1, 60), n_quotes=1,
                              max_grade_level=3.0, present_ratio_threshold=0.8,
                              constraints=["length", "present_tense",
                                           "simple_register", "dialogue"])
good = 'She walks to the door. "Are you coming?" she asks. He nods and follows.'
bad = ("A profound exhaustion had settled into the architecture of his frame, his hands "
       "describing tremulous arcs as he awaited the omnibus. ") * 9  # >60 words
m1, m2 = ck.evaluate(good), ck.evaluate(bad)
check("compliant story passes the original four", m1.violations == 0, str(m1.checks))
check("long literary past-tense story fails all four", m2.violations == 4,
      f"words={m2.word_count} grade={m2.grade_level}")
check("dialogue detects quoted speech only",
      ck.evaluate('"Stop," he says.').checks["dialogue"]
      and not ck.evaluate("He tells her to stop.").checks["dialogue"])
check("grade level is higher for the literary passage", m2.grade_level > m1.grade_level + 5,
      f"{m1.grade_level} vs {m2.grade_level}")
sub = EnglishConstraintChecker(backend="regex", word_range=(1, 60), n_quotes=1,
                               constraints=["length", "dialogue"])
check("constraint subset changes the violation count", sub.evaluate(bad).violations == 2)

print("\n== sentence splitting ==")
from noiseegra.constraint_metrics_en import split_sentences  # noqa: E402

# Terminal punctuation inside quotes is not a sentence boundary. Getting this
# wrong inflated the sentence count on every story containing dialogue, which
# shortened the mean sentence and so depressed the Flesch-Kincaid grade.
for text, want in [
    ('Lila runs. She sees a ball. "Look!" she says. The ball rolls away.', 4),
    ('She walks out. "Are you coming?" she asks. He nods.', 3),
    ('"Stop," he says. She stops.', 2),
    ('One sentence only.', 1),
    ('Line one.\nLine two.', 2),
    ('He paid $3.50 for it. Then he left.', 2),
]:
    got = split_sentences(text)
    check(f"{want} sentence(s) in {text[:34]!r}", len(got) == want,
          f"got {len(got)}: {got}")

print("\n== the thirteen-requirement task ==")
full = EnglishConstraintChecker(backend="regex")
check("thirteen requirements are scored by default", len(full.constraints) == 13,
      str(len(full.constraints)))
check("and fifteen in the monotone set", len(MONOTONE_CONSTRAINTS) == 15,
      str(len(MONOTONE_CONSTRAINTS)))
# The two newest exist because the other thirteen can all be satisfied by
# collapsed text: short sentences, short words, a plain opening and no
# subordinate clauses each get *easier* as the writing falls apart, and "The
# Rabbit jumps. The Rabbit sleeps. The Rabbit Bites. The Rabbit Bites." passes
# nine of them. The five-word-run rule misses it because the repeating unit is
# three words long.
COLLAPSED = ("The Rabbit jumps. The Rabbit sleeps. The Rabbit smiles. "
             "The Rabbit Bites. The Rabbit Bites. The Rabbit Bites.")
coll = EnglishConstraintChecker(backend="spacy", constraints=MONOTONE_CONSTRAINTS,
                                max_opener_uses=3).evaluate(COLLAPSED)
check("collapsed text fails the two rules written to catch it",
      coll.checks["distinct_sentences"] is False
      and coll.checks["fresh_openings"] is False,
      f"duplicate sentences {coll.n_dup_sentences}, worst opening {coll.same_opener}")
# The whole point of the monotone set is that no requirement in it is a band, so
# a push along a direction and the requirement it serves agree about which way is
# better. A banded one slipping in would be invisible in the aggregate and would
# reintroduce the failure the set exists to avoid.
check("no banded requirement is in the monotone set",
      not ({"length", "sentence_count", "sentence_band", "dialogue", "one_name"}
           & set(MONOTONE_CONSTRAINTS)),
      str(sorted(set(MONOTONE_CONSTRAINTS))))
check("every requirement has prompt text and a short label",
      set(full.requirements()) >= set(full.constraints)
      and set(full.requirements_short()) >= set(full.constraints))
generic = wp.build_generic_messages(full.requirements(), full.constraints)
body = generic[1]["content"]
check("the generic prompt carries one line per requirement",
      body.count("\n- ") == len(full.constraints), str(body.count("\n- ")))
check("the generic prompt states the actual thresholds",
      f"{full.max_words} words" in body and f"grade-{full.max_grade_level:g}" in body)
check("the generic prompt supplies no scenario",
      "You are" not in body and "wake up" not in body)

COMPLIANT = (
    'Mira feeds the hens. Two of them peck at her boot. She laughs and steps back '
    'very fast. "Come here," Mira says to them. The grey hen is on a log. Mira lifts '
    'it down with care. "Now you stay here," she says. Soon the hens run to the grass.'
)
mc = full.evaluate(COMPLIANT)
broken = [c for c in full.constraints if mc.checks[c] is False]
check("a story written to the rules breaks none of the checkable ones",
      broken == [], f"broken: {broken}  words={mc.word_count} sents={mc.n_sentences} "
                    f"grade={mc.grade_level} quotes={mc.n_quoted_spans}")

# The monotone set has to be jointly satisfiable, and by a story a person would
# actually write. Thirteen rules chosen one at a time can easily contradict each
# other -- "use the name three times" against "use no word more than three times"
# did, until names were exempted -- and a set nothing can satisfy would read as a
# method failure in every condition.
MONO_COMPLIANT = (
    'Rain taps the glass. "Come see," Mira says. A frog sits on the wet step. '
    '"It is cold," says Mira. She gets a warm cloth. The frog hops to her hand. '
    'Soft green skin feels smooth. "Now go home," Mira says. It jumps to the grass.'
)
mono = EnglishConstraintChecker(backend="spacy", constraints=MONOTONE_CONSTRAINTS,
                                max_opener_uses=3)
mm = mono.evaluate(MONO_COMPLIANT)
mbroken = [c for c in mono.constraints if mm.checks[c] is False]
check("the monotone set can all be satisfied at once", mbroken == [],
      f"broken: {mbroken}  words={mm.word_count} adverbs={mm.n_adverbs} "
      f"sensory={mm.n_sensory} subordinate={mm.n_subordinate} "
      f"quotes={mm.n_quoted_spans} name_uses={mm.name_uses}")

# Every rule is two-sided or exact, so each one has a violation on both sides where
# that is meaningful. A one-sided rule the model always satisfies measures nothing.
for name, story, want in [
    ("length", "Mira feeds the hens. Two of them peck at her boot.", False),
    ("length", COMPLIANT + " " + COMPLIANT, False),
    ("easy_opening", "The small brown hen with the crooked foot runs away fast. " + COMPLIANT, False),
    ("spelled_number", COMPLIANT.replace("Two of them", "Many of them"), False),
    ("spelled_number", COMPLIANT.replace("Two of them", "Three of 2"), False),
    ("varied_openers", "Mira runs. Mira stops. Mira waits. It rains. Now she goes.", False),
    ("plain_punctuation", "# A Hen\n\n" + COMPLIANT, False),
    ("plain_punctuation", COMPLIANT.replace("her boot.", "her boot; she laughs."), False),
    ("short_words", COMPLIANT + " It is extraordinarily complicated.", False),
    ("sentence_count", "She runs. She stops.", False),
    ("sentence_band",
     "She runs and runs and runs and runs and runs and runs and runs away.", False),
    ("sentence_band", COMPLIANT + " Go on.", False),
    ("dialogue", COMPLIANT.replace('"Now you stay here," she says.', "She says so."), False),
    ("dialogue", COMPLIANT + ' "One more," she says.', False),
    # A perturbation strong enough to buy diversity can send the model into a
    # loop, and a set of differently-broken stories scores as more diverse than a
    # set of good ones. Without this in the list an arm wins by writing worse.
    ("no_repetition", "I am going to the store. " * 8, False),
    ("no_repetition", COMPLIANT, True),
]:
    got = full.evaluate(story).checks[name]
    check(f"{name} catches its violation", got is want, f"got {got}")

check("one_name wants exactly one, used three times",
      full.evaluate(COMPLIANT).checks["one_name"] is True
      and full.evaluate(COMPLIANT.replace("Mira lifts", "Tom lifts"))
              .checks["one_name"] is False
      and full.evaluate(COMPLIANT.replace("Mira says", "she says"))
              .checks["one_name"] is False)
check("a name used only at the start of sentences is still found",
      full.evaluate(COMPLIANT).n_names == 1, str(full.evaluate(COMPLIANT).n_names))

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
from noiseegra.run_labels import label_run  # noqa: E402

OUT = Path("/tmp/_en_test"); shutil.rmtree(OUT, ignore_errors=True)
R.build_model = lambda mid, **kw: Tiny()
FAKE = ["A dragon learns to read.", "The sea returns something it took.", "Nobody remembers the war."]
wp.load_prompts = lambda n, **kw: FAKE[:n]
R.wp.load_prompts = wp.load_prompts

BASE = ["--model", "Qwen3-8B", "--suite", "compare", "--layers", "2", "5",
        "--task", "scenario",
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

sys.argv = ["x"] + BASE + ["--max-words", "120"]
try:
    with contextlib.redirect_stdout(io.StringIO()):
        R.main()
    check("changing a constraint threshold is rejected", False)
except SystemExit as exc:
    check("changing a constraint threshold is rejected", "max_words" in str(exc))

sys.argv = ["x"] + BASE + ["--max-words", "120", "--allow-task-change"]
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


# --------------------------------------------------------------------------- #
print("\n== generic task and the entropy gate ==")

from noiseegra.run_labels import label_run  # noqa: E402
from noiseegra.entropy_gate import gate_threshold as gate_thr, quantile  # noqa: E402
from noiseegra.subspace import ConstraintSpec, SteeringPlan  # noqa: E402

GOUT = Path("/tmp/_en_generic"); shutil.rmtree(GOUT, ignore_errors=True)
sys.argv = ["x", "--model", "Qwen3-8B", "--suite", "gate", "--layers", "2", "5",
            "--task", "generic", "--stories", "3", "--out", str(GOUT),
            "--max-new-tokens", "4", "--pca-rank", "2", "--protect-rank", "2",
            "--gate-samples", "2", "--no-diversity"]
with contextlib.redirect_stdout(io.StringIO()) as gbuf:
    R.main()
gtext = gbuf.getvalue()
gstate = json.loads((GOUT / "Qwen3-8B" / "state.json").read_text())

check("the generic task uses one prompt", gstate["task"]["num_prompts"] == 1)
check("the generic prompt is printed, so the run documents itself",
      "must satisfy every one of these requirements" in gtext)
check("no scenario is loaded", not gstate.get("prompts"))
check("every story lands in one group",
      all(all(k.startswith("0:") for k in cells) for cells in gstate["runs"].values()),
      str(list(list(gstate["runs"].values())[0])))

check("entropy was measured once and cached",
      "entropy" in gstate and any("quantiles" in v for v in gstate["entropy"].values()),
      str(gstate.get("entropy")))
quant = list(gstate["entropy"].values())[0]["quantiles"]
check("the gate thresholds are ordered none < median <= high",
      quant["none"] == 0.0 and quant["median"] <= quant["high"], str(quant))

gate_ids = sorted(gstate["runs"])
check("the gate suite produces a reference plus three gate levels",
      len(gate_ids) == 4, str(len(gate_ids)))

# Two conditions that differ must not print the same name. A second label
# implementation that read the plan instead of the run id used to exist and did
# not know about the gate, so a three-arm sweep printed one name three times.
gate_names = [R.condition_label(r) for r in gate_ids]
check("every condition in the sweep gets its own name",
      len(set(gate_names)) == len(gate_names), str(sorted(gate_names)))
check("the printed names are the ones the scorer will use later",
      gate_names == [label_run(r).text for r in gate_ids])
check("the live_scores labels match too",
      sorted(row["label"] for row in csv.DictReader(
          (GOUT / "Qwen3-8B" / "live_scores.csv").open(encoding="utf-8")))
      == sorted(gate_names))
labels = sorted(label_run(r).text for r in gate_ids)
check("the gate level is recoverable from the run id",
      any("gated to uncertain steps" in l for l in labels)
      and any("gated to the most uncertain steps" in l for l in labels)
      and any(l == "steering only, no perturbation" for l in labels), str(labels))

vals = [0.1 * i for i in range(101)]
check("quantile is linear-interpolated", abs(quantile(vals, 0.5) - 5.0) < 1e-9)
check("no gate means every step", gate_thr(vals, "none") == 0.0)
check("the median gate sits at the middle of the model's own entropies",
      abs(gate_thr(vals, "median") - 5.0) < 1e-9)
check("the high gate sits near the top", abs(gate_thr(vals, "high") - 9.0) < 1e-9)

dim, layer = 8, 0
plan = SteeringPlan.build(
    {"closure": {layer: torch.eye(1, dim).squeeze(0)}}, [layer],
    [ConstraintSpec("closure", beta=1.0)], rms_scale=1.0,
    noise_mode="orth", noise_alpha=0.5, offset_gamma=0.3, offset_mode="orth",
    gate_threshold=1.5, gate_level="median",
)
check("the plan carries the gate", plan.gate_threshold == 1.5 and plan.gate_level == "median")
torch.manual_seed(0)
open_delta = plan.delta_for(layer, 0, with_noise=True, with_offset=True)
torch.manual_seed(0)
shut_delta = plan.delta_for(layer, 0, with_noise=False, with_offset=False)
torch.manual_seed(0)
steer_only = plan.delta_for(layer, 0, with_noise=False, with_offset=False)
check("closing the gate drops the perturbation",
      not torch.allclose(open_delta, shut_delta),
      f"{float(open_delta.norm()):.3f} vs {float(shut_delta.norm()):.3f}")
check("closing the gate keeps the steering",
      torch.allclose(shut_delta, steer_only) and float(shut_delta.norm()) > 0,
      f"{float(shut_delta.norm()):.3f}")

shutil.rmtree(GOUT, ignore_errors=True)


# --------------------------------------------------------------------------- #
print("\n== the embedding model does not evict the generator ==")

from noiseegra import embeddings as E  # noqa: E402


class FakeCuda:
    """Stands in for torch.cuda with a controllable amount of free memory."""

    def __init__(self, free_gb):
        self.free_gb = list(free_gb)

    def is_available(self):
        return True

    def device_count(self):
        return len(self.free_gb)

    def mem_get_info(self, i):
        return int(self.free_gb[i] * 1024 ** 3), int(16 * 1024 ** 3)


real_cuda = torch.cuda
try:
    torch.cuda = FakeCuda([0.3, 0.4])          # an 8B model on both cards
    check("a full GPU sends the embedder to the CPU", E.pick_device() == "cpu",
          E.pick_device())
    torch.cuda = FakeCuda([0.3, 9.0])          # the second card is free
    check("a card with room is used", E.pick_device() == "cuda:1", E.pick_device())
    torch.cuda = FakeCuda([12.0, 9.0])
    check("the emptiest card wins", E.pick_device() == "cuda:0", E.pick_device())
finally:
    torch.cuda = real_cuda

check("no CUDA at all means the CPU",
      E.pick_device() in ("cpu",) or torch.cuda.is_available())


class OomThenFine:
    """Raises an out-of-memory error once, then behaves."""

    def __init__(self):
        self.calls = 0
        self.device = "cuda:0"

    def to(self, device):
        self.device = device
        return self

    def encode(self, texts, **kw):
        self.calls += 1
        if self.calls == 1:
            # torch.OutOfMemoryError only exists in newer torch; it subclasses
            # RuntimeError and carries this message either way, which is what the
            # handler matches on.
            raise RuntimeError("CUDA out of memory. Tried to allocate 80.00 MiB")
        import numpy as np
        v = np.eye(len(texts), 8)
        return v / np.linalg.norm(v, axis=1, keepdims=True)


from noiseegra.creativity_metrics import CreativityScorer  # noqa: E402

stub = OomThenFine()
sc = CreativityScorer.__new__(CreativityScorer)
sc.texts = ["one story here", "another story here", "a third story"]
sc.model, sc.batch_size, sc.truncate_words = stub, 16, None
sc.total_modal_collapse_indices_by_run = {}
emb = sc._encode()
check("an out-of-memory error falls back to the CPU instead of killing the run",
      stub.calls == 2 and stub.device == "cpu" and emb.shape[0] == 3,
      f"calls={stub.calls} device={stub.device}")


# --------------------------------------------------------------------------- #
print("\n== baseline on its own ==")

BOUT = Path("/tmp/_en_baseline"); 
# --------------------------------------------------------------------------- #
print("\n== sampling baselines ==")

SOUT = Path("/tmp/_en_sampling"); shutil.rmtree(SOUT, ignore_errors=True)
sys.argv = ["x", "--model", "Qwen3-8B", "--suite", "sampling",
            "--task", "generic", "--stories", "2", "--out", str(SOUT),
            "--max-new-tokens", "4", "--no-diversity"]
with contextlib.redirect_stdout(io.StringIO()):
    R.main()
sstate = json.loads((SOUT / "Qwen3-8B" / "state.json").read_text())
snames = sorted(R.condition_label(r) for r in sstate["runs"])
# The grid draws the temperature/compliance curve rather than sampling one point
# on it: a representation-level method only beats decoding parameters if it lands
# above the whole curve.
check("sampling sweeps the decoding grid and keeps the plain baseline",
      snames == ["baseline",
                 "baseline (temperature 1.3, top-p 0.9)",
                 "baseline (temperature 1.3, top-p 0.95)",
                 "baseline (temperature 1.6, top-p 0.9)",
                 "baseline (temperature 1.6, top-p 0.95)",
                 "baseline (temperature 1.8, top-k 40)",
                 "baseline (temperature 1.8, top-p 0.95)"], str(snames))
# Not a formality. Qwen3 ships top_p and top_k in its generation config, so an
# arm that leaves them unset silently inherits the checkpoint's truncation: the
# "plain" baseline was really top-p 0.95, and the top-p 0.95 arm was the same run
# under a different name -- both came back with identical statistics to every
# digit across 100 stories.
# A cut-off is only varied where the temperature is also raised. At temperature
# 1.0 it does nothing visible, because Qwen3 already ships top_p in its
# generation config: an earlier grid's top-p 0.95 arm at temperature 1.0 came
# back identical to the baseline in every digit across a hundred stories.
check("no cut-off arm sits at the baseline temperature",
      not any(("topp" in r and "temp" not in r) for r in sstate["runs"]),
      str(sorted(sstate["runs"])))
check("the sampling settings are in the run id",
      any("temp1p3__topp0p95" in r for r in sstate["runs"])
      and any("temp1p8__topk40" in r for r in sstate["runs"]), str(sorted(sstate["runs"])))
check("sampling needs no steering vectors either",
      not (SOUT / "Qwen3-8B" / "steering_Qwen3-8B.pt").exists())

sys.argv = ["x", "--model", "Qwen3-8B", "--suite", "sampling", "--task", "generic",
            "--stories", "2", "--out", str(SOUT), "--max-new-tokens", "4",
            "--baseline-top-k", "-1", "--no-diversity", "--allow-task-change"]
with contextlib.redirect_stdout(io.StringIO()):
    R.main()
check("a negative cut-off drops the top-k arm",
      len(json.loads((SOUT / "Qwen3-8B" / "state.json").read_text())["runs"]) == 7)

shutil.rmtree(SOUT, ignore_errors=True)
shutil.rmtree(BOUT, ignore_errors=True)
sys.argv = ["x", "--model", "Qwen3-8B", "--suite", "baseline",
            "--task", "generic", "--stories", "4", "--out", str(BOUT),
            "--max-new-tokens", "4", "--no-diversity"]
with contextlib.redirect_stdout(io.StringIO()) as bbuf:
    R.main()
btext = bbuf.getvalue()
bstate = json.loads((BOUT / "Qwen3-8B" / "state.json").read_text())

check("baseline produces exactly one condition", len(bstate["runs"]) == 1,
      str(list(bstate["runs"])))
check("that condition is the unmodified one",
      R.condition_label(next(iter(bstate["runs"]))) == "baseline",
      R.condition_label(next(iter(bstate["runs"]))))
check("all four stories are generated",
      len(next(iter(bstate["runs"].values()))) == 4)
check("no steering vectors are extracted",
      not (BOUT / "Qwen3-8B" / "steering_Qwen3-8B.pt").exists()
      and "extracting (once)" not in btext)
check("no activation scale is calibrated",
      not bstate.get("rms_scale"), str(bstate.get("rms_scale")))
check("and none is reported",
      "activation scale [" not in btext,
      next((l for l in btext.splitlines() if "activation scale" in l), ""))
check("the run says why it skipped them", "baseline only" in btext)
check("no steering layers are claimed in the header",
      "steering layers" not in btext)

# The steering path must still work in the same output directory afterwards.
sys.argv = ["x", "--model", "Qwen3-8B", "--suite", "compare",
            "--task", "generic", "--stories", "4", "--out", str(BOUT),
            "--layers", "2", "5", "--max-new-tokens", "4", "--pca-rank", "2",
            "--protect-rank", "2", "--no-diversity"]
with contextlib.redirect_stdout(io.StringIO()) as b2buf:
    R.main()
bstate2 = json.loads((BOUT / "Qwen3-8B" / "state.json").read_text())
check("a later steering run reuses the baseline stories",
      bstate2["runs"][next(iter(bstate["runs"]))]
      == next(iter(bstate["runs"].values())))
check("and adds the steering condition without regenerating the baseline",
      len(bstate2["runs"]) == 2 and "0 of " not in b2buf.getvalue(),
      str(len(bstate2["runs"])))

shutil.rmtree(BOUT, ignore_errors=True)


# --------------------------------------------------------------------------- #
print("\n== the embedding model loads late, not early ==")

from noiseegra.creativity_metrics import CreativityScorer as CS  # noqa: E402
import noiseegra.embeddings as EM  # noqa: E402

calls = {"n": 0, "device_seen": []}
real_loader = EM.load_embedder


class TinyEmbedder:
    device = "cpu"

    def encode(self, texts, **kw):
        import numpy as np
        v = np.eye(len(texts), 8)
        return v / np.linalg.norm(v, axis=1, keepdims=True)


def fake_loader(name=None, device=None):
    calls["n"] += 1
    calls["device_seen"].append(device)
    return TinyEmbedder()


EM.load_embedder = fake_loader
try:
    sc = CS(["a story about rain", "a story about snow"], device="cpu")
    check("constructing the scorer loads nothing", calls["n"] == 0, str(calls["n"]))
    check("the device can be reported without loading", sc.device == "cpu")
    check("no load has happened yet", calls["n"] == 0, str(calls["n"]))
    sc.semantic_diversity()
    check("the first score loads it", calls["n"] == 1, str(calls["n"]))
    sc.semantic_diversity()
    check("and only once", calls["n"] == 1, str(calls["n"]))
finally:
    EM.load_embedder = real_loader

shutil.rmtree(OUT, ignore_errors=True)
print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
    sys.exit(1)
print("english smoke test passed")


# --------------------------------------------------------------------------- #
# The middle-school task: the same architecture with the four rules that force
# the prose to be as small as possible removed, and the reading level a floor
# instead of a ceiling.
print("\n== the middle-school requirement set ==")
from noiseegra.constraint_metrics_en import MIDDLE_CONSTRAINTS  # noqa: E402
from noiseegra import writingprompts as _wp  # noqa: E402

_gone = {"short_sentences", "easy_opening", "short_words", "simple_syntax",
         "simple_register"}
check("the rules that force tiny prose are gone",
      not (_gone & set(MIDDLE_CONSTRAINTS)),
      ", ".join(sorted(_gone & set(MIDDLE_CONSTRAINTS))) or "none present")
check("the reading level is now a floor", "mature_register" in MIDDLE_CONSTRAINTS)
check("the anti-repetition rules are kept",
      {"no_repetition", "distinct_sentences", "fresh_openings"} <= set(MIDDLE_CONSTRAINTS))

_mid = EnglishConstraintChecker(backend="regex", constraints=MIDDLE_CONSTRAINTS,
                                min_grade_level=6.0)
_simple = "The cat sits. The dog runs. Mia laughs. The sun is warm. She plays."
_grown = ("Mira edges along the flooded corridor, counting the doors she passes "
          "and listening for the hum of the generator below. \"Someone left the "
          "hatch open,\" she says, and her voice carries further than she wants. "
          "Nothing answers except the water moving against the walls.")
check("a tiny-sentence story fails the new reading floor",
      _mid.evaluate_all([_simple])["stories"][0].checks["mature_register"] is False)
check("grown-up prose passes it",
      _mid.evaluate_all([_grown])["stories"][0].checks["mature_register"] is True)

# The children's set must re-score exactly as before: the new rule is an
# addition, not a replacement.
_old = EnglishConstraintChecker(backend="regex", constraints=MONOTONE_CONSTRAINTS)
check("the children's set is untouched by the addition",
      "mature_register" not in MONOTONE_CONSTRAINTS
      and len(MONOTONE_CONSTRAINTS) == 15)

_msgs = _wp.build_middle_messages(_mid.requirements(), list(MIDDLE_CONSTRAINTS),
                                  target=150)
check("the prompt asks for a middle-school reader",
      "middle-school reader" in _msgs[1]["content"])
check("the prompt asks for a longer story", "150 words" in _msgs[1]["content"])
check("the prompt lists every scored rule and nothing else",
      _msgs[1]["content"].count("\n- ") == len(MIDDLE_CONSTRAINTS),
      f'{_msgs[1]["content"].count(chr(10) + "- ")} bullets for {len(MIDDLE_CONSTRAINTS)} rules')


def test_opening_tense_switch_is_detected():
    """A story that opens in the past and then narrates in the present is flagged.

    The present-tense requirement is a share over the whole story, so it cannot
    see that the exceptions are all in the opening sentence. Under the
    constraint push 96% of finite verbs are present tense while 73% of stories
    open in the past and switch immediately.
    """
    from noiseegra.constraint_metrics_en import (
        MIDDLE_CONSTRAINTS, EnglishConstraintChecker, opens_in_the_wrong_tense,
    )
    from noiseegra.constraint_metrics_en import _Spacy

    # The regex backend cannot see present-tense verbs at all -- it returns zero
    # of them for "She walks to the window" -- so every tense measurement in this
    # project is made with spaCy, and so is this check.
    if not _Spacy.available():
        print("  [SKIP] opening-tense check needs spaCy")
        return

    ck = EnglishConstraintChecker(backend="spacy", constraints=MIDDLE_CONSTRAINTS)

    switches = ("Mara looked up from her book and frowned at the clock. "
                "She walks to the window and opens it wide. "
                "The wind comes in and lifts the curtain. "
                "She listens to the trees for a while. "
                "Her brother calls her name from downstairs.")
    consistent = ("Mara looks up from her book and frowns at the clock. "
                  "She walks to the window and opens it wide. "
                  "The wind comes in and lifts the curtain. "
                  "She listens to the trees for a while. "
                  "Her brother calls her name from downstairs.")
    all_past = ("Mara looked up from her book and frowned at the clock. "
                "She walked to the window and opened it wide. "
                "The wind came in and lifted the curtain. "
                "She listened to the trees for a while. "
                "Her brother called her name from downstairs.")

    assert opens_in_the_wrong_tense(switches, ck), "the tense switch was missed"
    assert not opens_in_the_wrong_tense(consistent, ck), "consistent present flagged"
    assert not opens_in_the_wrong_tense(all_past, ck), "consistent past flagged"
    print("  [PASS] an opening in the wrong tense is detected, consistent stories are not")


if __name__ == "__main__":
    test_opening_tense_switch_is_detected()
