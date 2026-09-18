#!/usr/bin/env python
"""Orthogonal constraint steering on English WritingPrompts. Resumable.

    python scripts/run_english_experiment.py --model Qwen3-8B --suite compare \
        --num-prompts 10 --stories-per-prompt 5

Generates K stories for each of P prompts under every condition. Diversity has to
be measured within a prompt -- stories from different prompts are trivially
different -- so the scorer groups by prompt before averaging.

Every story is written into state.json as it is produced; re-running the same
command resumes and regenerates nothing.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from noiseegra import writingprompts as wp  # noqa: E402
from noiseegra.activation_basis import (  # noqa: E402
    StoryAxes, collect_block_pcs, collect_prompt_pcs, collect_story_pcs,
)
from noiseegra.constraint_metrics_en import (  # noqa: E402
    DEFAULT_MAX_ADVERBS,
    DEFAULT_MAX_OPENER_USES,
    DEFAULT_MAX_WORD_USES,
    DEFAULT_MIN_SENSORY,
    DEFAULT_PRESENT_RATIO,
    DEFAULT_MIN_GRADE_LEVEL,
    EnglishConstraintChecker,
)
from noiseegra.defaults import (  # noqa: E402
    EN_MIDDLE_CONSTRAINTS,
    EN_MIDDLE_MAX_NEW_TOKENS,
    EN_MIDDLE_MAX_ADVERBS,
    EN_MIDDLE_MAX_OPENER_USES,
    EN_MIDDLE_MAX_WORDS,
    EN_MIDDLE_MAX_WORD_USES,
    EN_MIDDLE_MIN_SENSORY,
    EN_MIDDLE_PRESENT_RATIO,
    EN_MIDDLE_REGISTER_STEER_VECTORS,
    EN_MIDDLE_STEER_VECTORS,
    EN_MONOTONE_CONSTRAINTS,
    EN_MONOTONE_MAX_OPENER_USES,
    EN_MONOTONE_STEER_VECTORS,
    EN_STEER_VECTORS,
    EN_TASK_CONSTRAINTS,
    EN_MAX_GRADE_LEVEL,
    EN_MAX_WORDS,
    EN_MIN_WORDS,
    EN_MODEL_HF_IDS,
    EN_MODEL_LAYER_RANGES,
    RMS_ALPHA,
)
from noiseegra.RMS_std import RMSCalibrator  # noqa: E402
from noiseegra.setup_experiment import _spec_mode, _spec_to_run_id, make_specs  # noqa: E402
from noiseegra.steering_vectors import (  # noqa: E402
    SteeringVectorExtractor,
    SteeringVectorSet,
    load_pairs,
)

from build_steering_vectors import build_model  # noqa: E402
from kaggle_orthosteer import generate_one, load_state, save_state  # noqa: E402
from run_orthosteer_experiment import build_suite  # noqa: E402
from score_english import (  # noqa: E402
    constraint_legend, live_table, score_condition,
)

from noiseegra.run_labels import label_run, plan_summary  # noqa: E402
from noiseegra.entropy_gate import (  # noqa: E402
    GATE_LEVELS, collect_decode_entropies, describe as describe_gate, gate_threshold,
)

_DATA = Path(__file__).resolve().parents[1] / "noiseegra" / "data"
EN_PAIRS = _DATA / "steering_pairs_en.json"

# Which set of contrast pairs the directions are extracted from. Both sets use
# the same within-item design; they differ in the register both sides are
# written in. "children" is the original: every pair, positive and negative
# alike, is a sentence from a book for a five-year-old. "middle" rewrites all of
# them at twelve to twenty words a sentence and adds a direction for developed
# prose, because directions measured inside a four-word-sentence world were
# being applied to a task that asks for real sentences.
PAIR_SETS = {
    "children": EN_PAIRS,
    "middle": _DATA / "steering_pairs_en_middle.json",
}

# Any suite that can draw a per-story offset from an estimated basis must be
# listed here, or the basis is never built and the offset silently falls back to
# an isotropic draw while the run id still says `obstory`. That fault ran
# unnoticed through r16-spread and r17-headline: `spread` and `frontier` were
# missing, so every "story-difference basis" arm in those runs was isotropic
# noise. `tests/test_suites_build.py` asserts every suite whose run ids name an
# estimated basis is in this set, and the runner crashes at generation time if a
# basis-naming run id survives with no basis built.
# Any suite that reads args.gate_thresholds to set a per-arm gate must be listed
# here, or every level silently resolves to "no gate" and the sweep reads as a
# null while the run ids faithfully record gates that were never applied.
# tests/test_suites_build.py asserts a suite whose run ids name a gate is here.
GATE_SUITES = {"gate", "gatedwrite"}

BASIS_SUITES = {"offset", "story", "prompt", "main", "pareto", "select", "feedback",
                "assemble", "headtohead", "closure", "control", "ablate",
                "controls", "tame", "core4", "combine", "amplify", "spread", "frontier", "siting", "prefill", "asymmetric", "boundary", "bands", "whilewriting", "gatedwrite", "promptbudget", "opening", "framing",
                "constdose"}


def weights_are_cached(model_id: str) -> bool:
    """True if the full snapshot is already on local disk."""
    if Path(model_id).is_dir():
        return True
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(model_id, local_files_only=True)
        return True
    except Exception:
        return False

# Every steered direction is a property of every sentence, so every schedule is
# flat. The earlier list ramped `closure` up over the story, which only made sense
# for a direction meaning "wrap it up"; that direction is gone, because it was not
# one of the scored requirements.
EN_SCHEDULES = {name: "constant" for name in EN_STEER_VECTORS}


def condition_label(run_id: str) -> str:
    """Readable name for a condition.

    Derived from the run id rather than from the plan, because the run id is the
    one thing that is guaranteed to distinguish two conditions -- that is what it
    is for. A second implementation reading the plan directly existed here and
    drifted: it never learned about the entropy gate, so a three-arm gate sweep
    printed the same name three times.
    """
    return label_run(run_id).text


def seed_for(prompt_idx: int, story_idx: int) -> int:
    """Stable per (prompt, story) seed, shared across conditions."""
    return (42 + prompt_idx * 100003 + story_idx * 7919) % (2 ** 31)


def write_csvs(out: Path, state: dict) -> None:
    for run_id, cells in state["runs"].items():
        rows = []
        for key, text in cells.items():
            p_idx, s_idx = (int(x) for x in key.split(":"))
            rows.append((p_idx, s_idx, text))
        rows.sort()
        with (out / f"{run_id}.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["prompt_index", "story_index", "story"])
            w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen3-8B", choices=sorted(EN_MODEL_HF_IDS))
    ap.add_argument("--model-id", help="HF id or local snapshot dir; overrides --model's id")
    ap.add_argument("--layers", nargs=2, type=int, metavar=("LO", "HI"))
    ap.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16"])
    ap.add_argument("--suite", nargs="+", default=["compare"],
                    choices=["baseline", "sampling", "compare", "method", "noise",
                             "offset", "story", "prompt", "main", "pareto", "directions", "select", "budget", "feedback", "assemble", "headtohead", "closure", "control",
                             "ablate", "controls", "tame", "fsc", "core4", "combine", "dose", "siting", "dropone", "prefill", "asymmetric", "boundary", "bands", "whilewriting", "gatedwrite", "promptbudget", "opening", "framing",
                             "amplify", "constdose", "spread", "frontier",
                             "window", "decay", "core", "ortho", "alpha", "gate",
                             "beta", "loo", "all"])
    ap.add_argument("--task", default="generic", choices=["generic", "scenario"],
                    help="'generic' is the published design: one instruction with no "
                         "scenario, many requirements, and every story in one group, so "
                         "the measure is how many different stories the model invents. "
                         "'scenario' draws WritingPrompts scenarios, which supply the "
                         "content and cap how different the stories can be")
    ap.add_argument("--stories", type=int, default=100,
                    help="stories for the generic task, all in one group")
    ap.add_argument("--gate", default="none", choices=sorted(GATE_LEVELS),
                    help="entropy gate for every perturbed condition outside --suite "
                         "gate: 'none' perturbs every decode step, 'median' the more "
                         "uncertain half, 'high' the most uncertain tenth")
    ap.add_argument("--gate-samples", type=int, default=6,
                    help="unsteered generations used to measure the model's own entropy "
                         "distribution before setting the gate threshold")
    ap.add_argument("--allow-task-change", action="store_true",
                    help="proceed even though the checkpoint was written under different "
                         "constraint thresholds or a different prompt selection")
    ap.add_argument("--no-diversity", dest="diversity", action="store_false",
                    help="skip Vendi/Self-BLEU while scoring conditions as they finish "
                         "(avoids downloading the embedding model)")
    ap.add_argument("--with-baseline", action="store_true",
                    help="prepend an unsteered baseline condition to whichever suite is run "
                         "(already included in `compare` and `noise`)")
    ap.add_argument("--constraint-set", choices=("mixed", "monotone", "middle"), default="mixed",
                    help="'monotone': the thirteen one-sided requirements, steered along "
                         "nine directions. 'mixed': the earlier set, which includes the "
                         "banded requirements (word count, sentence count, exactly-N "
                         "quoted lines). Sets --constraints and --steer-vectors unless "
                         "those are given explicitly.")
    ap.add_argument("--constraints", nargs="*", default=list(EN_TASK_CONSTRAINTS),
                    help="what the prompt asks for and the scorer checks")
    ap.add_argument("--steer-vectors", nargs="*", default=list(EN_STEER_VECTORS),
                    help="which of those steering is applied along; the rest are asked "
                         "for in the prompt only")
    ap.add_argument("--num-prompts", type=int, default=10)
    ap.add_argument("--stories-per-prompt", type=int, default=5)
    ap.add_argument("--prompt-seed", type=int, default=0)
    ap.add_argument("--prompt-split", default="test")
    ap.add_argument("--min-words", type=int, default=EN_MIN_WORDS)
    ap.add_argument("--max-words", type=int, default=EN_MAX_WORDS)
    ap.add_argument("--max-grade", type=float, default=EN_MAX_GRADE_LEVEL)
    ap.add_argument("--min-grade", type=float, default=DEFAULT_MIN_GRADE_LEVEL,
                    help="reading-level floor for --constraint-set middle: the story "
                         "must be at least this Flesch-Kincaid grade")
    ap.add_argument("--max-word-uses", type=int, default=DEFAULT_MAX_WORD_USES,
                    help="how often a word of four letters or more may be reused. "
                         "A longer story reuses more, so the middle-school task "
                         "needs this above the children's task's 3")
    ap.add_argument("--story-target", type=int, default=150,
                    help="the word count --constraint-set middle asks the model to "
                         "aim for in the prompt")
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--record-uncertainty", action="store_true",
                    help="record each story's mean next-token entropy, read off the "
                         "raw scores before temperature or any cut-off. Raising the "
                         "temperature leaves this untouched, so it distinguishes an "
                         "intervention that makes the model less certain from one "
                         "that moves it somewhere else while leaving it as certain")
    ap.add_argument("--opening-at-prompt", action="store_true",
                    help="also add the per-story perturbation to the prompt, not "
                         "only to the opening decode steps")
    ap.add_argument("--prompt-head", type=int, default=0,
                    help="how many opening prompt positions to leave unperturbed")
    ap.add_argument("--head-sweep", nargs="*", type=int, default=[8, 16, 24],
                    help="how many opening prompt positions to leave unperturbed")
    ap.add_argument("--opening-steps", nargs="*", type=int, default=[12, 30, 60],
                    help="how many opening decode steps the perturbation lasts")
    ap.add_argument("--gate-sweep", nargs="*", default=None,
                    help="which entropy gates to cross the perturbation with")
    ap.add_argument("--prompt-tail", type=int, default=8,
                    help="how many final prompt positions to leave unperturbed")
    ap.add_argument("--tail-sweep", nargs="*", type=int, default=[2, 4, 8],
                    help="how many of the final prompt positions to leave "
                         "unperturbed -- the chat template's own tokens, which "
                         "mark that the instruction has ended")
    ap.add_argument("--prefill-gains", nargs="*", type=float, default=[2.0, 4.0],
                    help="prompt-side push strengths, as multiples of the strength "
                         "used while writing")
    ap.add_argument("--max-adverbs", type=int, default=DEFAULT_MAX_ADVERBS,
                    help="at most this many adverbs (the verbs-not-adverbs rule)")
    ap.add_argument("--min-sensory", type=int, default=DEFAULT_MIN_SENSORY,
                    help="at least this many words for how something looks, sounds, "
                         "feels, smells or tastes")
    ap.add_argument("--present-ratio", type=float, default=DEFAULT_PRESENT_RATIO,
                    help="share of finite verbs that must be in the present tense "
                         "for the tense requirement to count as met")
    ap.add_argument("--pairs", default="children", choices=sorted(PAIR_SETS),
                    help="which contrast-pair file the steering directions are "
                         "extracted from. 'children' is the original set, whose "
                         "every pair is written on both sides in the register of a "
                         "book for a five-year-old. 'middle' is the same design "
                         "rewritten at middle-school sentence length, with a "
                         "direction for developed prose that the children's set "
                         "has no counterpart for")
    ap.add_argument("--extraction-context", default="task", choices=["task", "generic"],
                    help="the conversation the contrast activations are read in. "
                         "'task' uses the real instruction the stories are written "
                         "to, which is what steering_vectors.py says it does and "
                         "what it did not do: directions were measured under 'You "
                         "are a creative writer / write a short story' and applied "
                         "under a twelve-requirement children's-reading prompt. "
                         "Steering directions are known to degrade out of "
                         "distribution. 'generic' is the old behaviour, kept so the "
                         "difference can be measured rather than assumed")
    ap.add_argument("--beta-sweep", nargs="*", type=float, default=[0.25, 0.5, 1.0, 2.0, 4.0],
                    help="in --suite directions, multipliers on the calibrated "
                         "coefficients for the whole block")
    ap.add_argument("--gamma-sweep", nargs="*", type=float, default=[0.05, 0.15, 0.4],
                    help="per-story offset magnitudes used by --suite offset")
    ap.add_argument("--offset-rank", type=int, default=64,
                    help="how many activation principal components offsets may use")
    ap.add_argument("--offset-basis", dest="offset_basis_kind", default="prompt",
                    choices=["step", "story", "prompt"],
                    help="which directions a per-story offset is drawn from. 'prompt' "
                         "takes the principal components of the instruction's own "
                         "hidden states in a single forward pass, so nothing has to "
                         "be generated first. 'story' samples stories and takes the "
                         "components across their mean activations, so the offset "
                         "moves along an axis the model's own stories already differ "
                         "on -- stronger, but it costs a sampling pass. 'step' takes "
                         "them over individual decode steps, whose leading directions "
                         "describe token position rather than content")
    ap.add_argument("--offset-basis-stories", type=int, default=32,
                    help="unsteered stories sampled to estimate the story-level basis")
    ap.add_argument("--offset-basis-tokens", type=int, default=120,
                    help="tokens generated per sample while estimating either basis")
    ap.add_argument("--noise-horizon", type=int, default=24,
                    help="decode steps the perturbation covers in --suite window, and "
                         "the cosine horizon it fades out over in --suite decay. Kept "
                         "separate from --horizon, which drives the constraint "
                         "schedules: a 24-token noise window must not also compress "
                         "the closure ramp into 24 tokens")
    ap.add_argument("--random-directions", action="store_true",
                    help="replace every extracted constraint direction, and the "
                         "principal components that build the protected subspace, "
                         "with Gaussian draws of the same norm. The control for "
                         "'does the extracted direction mean anything, or would any "
                         "push of that size do the same?' -- if the random arm moves "
                         "the requirements as much as the real one, the extraction "
                         "is not what is doing the work")
    ap.add_argument("--read-window", default="all", choices=["all", "first", "last"],
                    help="which continuation positions the contrast activations are "
                         "averaged over. 'all' is what has been used so far and "
                         "mixes the position where the property is decided with the "
                         "content that follows it. 'first' takes the opening tokens, "
                         "where 'walks' or 'walked' is actually chosen, which is "
                         "closer to standard CAA. 'last' takes the final position")
    ap.add_argument("--read-tokens", type=int, default=4,
                    help="how many opening tokens --read-window first averages over")
    ap.add_argument("--keep", nargs="*", default=["simple_register=+1",
                                                  "varied_openers=-1"],
                    help="NAME=SIGN for each direction --suite select keeps, with "
                         "the sign measured by --suite directions rather than "
                         "guessed from what the direction was extracted for")
    ap.add_argument("--tune-betas", nargs="*", type=float, default=[1.5, 3.0, 4.5, 6.0],
                    help="coefficient magnitudes --suite select sweeps each kept "
                         "direction over")
    ap.add_argument("--combo-beta", type=float, default=3.0,
                    help="coefficient the kept directions are combined at")
    ap.add_argument("--thinking", default="off", choices=["off", "on", "default"],
                    help="whether the model is allowed to emit a <think> block "
                         "before the story. Qwen3's chat template leaves this on, "
                         "and a small model can spend its whole token budget "
                         "reasoning and never reach the story -- slow to generate, "
                         "and the reasoning text is then what gets scored. 'off' "
                         "switches it off in the template, 'default' leaves the "
                         "template alone. A reasoning block that appears anyway is "
                         "stripped before scoring either way")
    ap.add_argument("--control-betas", nargs="*", type=float, default=[1.0, 2.0],
                    help="gain on the constraint error in --suite control")
    ap.add_argument("--closure-betas", nargs="*", type=float, default=[2.0, 4.0],
                    help="strength of the closure push in --suite closure")
    ap.add_argument("--closure-horizon", type=int, default=80,
                    help="decode steps before the closure push starts. A story of "
                         "50 to 65 words is about 75 tokens, so the default leaves "
                         "a legal story untouched and presses only on one running "
                         "over")
    ap.add_argument("--feedback-betas", nargs="*", type=float, default=[0.5, 1.0],
                    help="gain on the shortfall for --suite feedback. 1.0 closes "
                         "the whole gap between where the story sits on a "
                         "constraint axis and where compliant text sits, in one "
                         "step at every steered layer")
    ap.add_argument("--feedback-cap", type=float, default=0.1,
                    help="ceiling on one feedback correction, as a fraction of the "
                         "hidden state's own length. A story far from compliant on "
                         "several axes at once would otherwise receive a correction "
                         "large enough to break the text")
    ap.add_argument("--steer-budget", type=float, default=None,
                    help="total length of the constraint push, in units of the "
                         "model's own activation scale, held fixed however many "
                         "constraints are steered. Without it, steering k "
                         "constraints at coefficient beta gives a push of "
                         "beta*sqrt(k)*rms, so adding a constraint raises the dose "
                         "as a side effect and the number of constraints and the "
                         "strength of the intervention are the same knob")
    ap.add_argument("--budget-sweep", nargs="*", type=float, default=[2.0, 3.0, 4.5],
                    help="budgets --suite budget tries")
    ap.add_argument("--realloc-kappa", type=float, default=0.6,
                    help="spread of the per-story reallocation of the budget across "
                         "constraints. The total push is unchanged; only how it is "
                         "divided between the constraints moves, so the "
                         "perturbation never leaves the constraint subspace")
    ap.add_argument("--peek-stories", type=int, default=8,
                    help="print the opening of each condition's first N stories to "
                         "the log as they are generated, so a broken arm is visible "
                         "from the live output instead of after the run. 0 disables.")
    ap.add_argument("--abort-broken-arms", action="store_true",
                    help="after a condition's first few stories, run the cheap "
                         "coherence checks on them; if nearly all are broken, skip "
                         "the rest of that condition instead of paying for 100 "
                         "stories of rubbish. The check point and threshold are "
                         "--abort-check-at and --abort-flag-frac.")
    ap.add_argument("--abort-check-at", type=int, default=12,
                    help="how many stories to generate before judging an arm")
    ap.add_argument("--abort-flag-frac", type=float, default=0.9,
                    help="abort when at least this fraction of the first stories "
                         "fail the coherence checks. 0.9 of 12 means 11 broken; "
                         "ordinary imperfect arms (a third broken) are never touched")
    ap.add_argument("--steer-horizon", type=int, default=32,
                    help="tokens the steering decays over when a schedule is used")
    ap.add_argument("--probe-beta", nargs="*", type=float, default=[3.0],
                    help="how hard --suite directions pushes a single direction "
                         "when it checks whether that direction moves its own "
                         "requirement. More than one magnitude separates 'this "
                         "direction does not help' from 'this much of it is too "
                         "much': a single large probe cannot tell them apart, and "
                         "round 4 could not say whether beta 3 was simply past the "
                         "point where the text starts to suffer")
    ap.add_argument("--kappa-sweep", nargs="*", type=float, default=[0.1, 0.15],
                    help="f(S_c) sideways-step magnitudes used by --suite pareto")
    ap.add_argument("--main-gamma", type=float, default=0.15,
                    help="per-story offset magnitude used by --suite main")
    ap.add_argument("--main-kappa-perp", type=float, default=0.15,
                    help="f(S_c) sideways step, read on the same scale as gamma: the "
                         "step's length as a fraction of the hidden state's own length")
    ap.add_argument("--main-kappa-rotate", type=float, default=1.0,
                    help="f(S_c) rotation: the constraint vector is turned by "
                         "atan(kappa) and keeps its length, so 1.0 is 45 degrees")
    ap.add_argument("--main-kappa-gain", type=float, default=0.5,
                    help="f(S_c) per-constraint gain jitter: the spread of a "
                         "lognormal with mean 1 applied to each constraint's beta")
    ap.add_argument("--lambda-sweep", nargs="*", type=float, default=[1.5, 2.0, 3.0],
                    help="how far --suite amplify stretches a story's own deviation "
                         "from the average story. 1 is a no-op; 2 doubles it")
    ap.add_argument("--alpha-sweep", nargs="*", type=float,
                    default=[0.0, 0.0875, 0.175, 0.35, 0.7],
                    help="noise strengths used by --suite alpha")
    ap.add_argument("--alpha", type=float, default=RMS_ALPHA)
    ap.add_argument("--protect-rank", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=200)
    ap.add_argument("--steer-prefill", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=400)
    ap.add_argument("--word-budget", type=int, default=None,
                    help="stop a generation once it has written this many words and "
                         "record it as it stands. A perturbation strong enough to buy "
                         "diversity is often strong enough to stop the model "
                         "terminating, and such a sample runs to the token cap every "
                         "time -- about six times slower than every other condition, "
                         "for output already scored as failed. Default is half again "
                         "the longest legal story: at that length the word-count rule "
                         "is already broken past recovery, so continuing cannot turn "
                         "the sample into a pass. 0 disables it")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--offset-draw", choices=("iid", "spread"), default="iid",
                    help="'iid' draws each story's perturbation independently, which is "
                         "what every round so far did. 'spread' lays the whole set out "
                         "in advance so they repel one another: same length, same "
                         "subspace, same expected direction, but no two stories get "
                         "near-identical perturbations. A hundred independent draws from "
                         "a rank-8 basis contain pairs 95% alike.")
    ap.add_argument("--top-p", type=float, default=None,
                    help="nucleus cut-off for EVERY condition, steered ones included. "
                         "Unset leaves the checkpoint's own generation config in force, "
                         "which for Qwen3 is not untruncated sampling.")
    ap.add_argument("--top-k", type=int, default=None,
                    help="top-k cut-off for every condition, steered ones included")
    ap.add_argument("--turn-sweep", nargs="*", type=float, default=[0.15, 0.3, 0.5],
                    help="how far the constraint push is turned, in --suite constdose: "
                         "the angle is atan(kappa), so 0.3 is 17 degrees. The length of "
                         "the push is unchanged whatever this is.")
    ap.add_argument("--realloc-sweep", nargs="*", type=float, default=[0.3, 0.6, 1.0],
                    help="spread of the per-story re-weighting across requirements, in "
                         "--suite constdose. The total push is renormalised afterwards, "
                         "so this changes the split and not the dose.")
    ap.add_argument("--frame-sweep", nargs="*", type=float, default=[0.1, 0.2],
                    help="how far each direction is moved before the directions are made "
                         "orthogonal, in --suite constdose")
    ap.add_argument("--sampling-grid", nargs="*", default=None,
                    help="decoding settings for --suite sampling, as TEMP or TEMP:TOP_P "
                         "(e.g. 1.0 1.0:0.95 1.3 1.6:0.95). Default sweeps six, which is "
                         "enough to draw the temperature/compliance curve a "
                         "representation-level method has to sit above.")
    ap.add_argument("--baseline-temperature", type=float, default=1.8,
                    help="temperature for the sampling baselines in --suite sampling")
    ap.add_argument("--baseline-top-p", type=float, default=0.95,
                    help="nucleus cut-off for the high-temperature baseline; "
                         "pass a negative value to skip that arm")
    ap.add_argument("--baseline-top-k", type=int, default=40,
                    help="top-k cut-off for the high-temperature baseline; "
                         "pass a negative value to skip that arm")
    ap.add_argument("--pca-rank", type=int, default=8)
    ap.add_argument("--embedding-model", default=None,
                    help="diversity embedding model: registry key or HF id "
                         "(default qwen3-0.6b; use bge-m3 for the published Arabic setup)")
    ap.add_argument("--truncate-words", type=int, default=None,
                    help="cut every story to its first N words before scoring diversity")
    ap.add_argument("--embedding-device", default="auto",
                    help="where to put the diversity embedding model: 'auto' picks a "
                         "GPU with room and falls back to the CPU, which is what you "
                         "want while an 8B model is holding the card")
    ap.add_argument("--shard", default=None, metavar="I/N",
                    help="run only conditions I, I+N, I+2N ... of the suite. A "
                         "Kaggle kernel has two T4s and one model uses one of "
                         "them, so two shards pinned to a GPU each halve the wall "
                         "clock for the same quota. Each shard writes its own "
                         "output directory; the tables are merged when they are "
                         "scored")
    ap.add_argument("--out", default="/kaggle/working/english")
    ap.add_argument("--dry-run", action="store_true",
                    help="parse the arguments, print the conditions the suite would "
                         "run, and stop. A typo in a flag name costs a Kaggle "
                         "session start, a model download and several minutes "
                         "before argparse rejects it; this catches it locally in a "
                         "second")
    args = ap.parse_args()

    # How many sentences one word may begin. Each rule set below sets its own
    # level; this is the value for the original mixed set, which sets none.
    args.max_opener_uses = DEFAULT_MAX_OPENER_USES

    if args.constraint_set == "monotone":
        # Only override what the caller left at its default, so an explicit
        # --constraints or --steer-vectors still wins.
        if list(args.constraints) == list(EN_TASK_CONSTRAINTS):
            args.constraints = list(EN_MONOTONE_CONSTRAINTS)
        if list(args.steer_vectors) == list(EN_STEER_VECTORS):
            args.steer_vectors = list(EN_MONOTONE_STEER_VECTORS)
        args.max_opener_uses = EN_MONOTONE_MAX_OPENER_USES

    if args.constraint_set == "middle":
        # The middle-school task: the same architecture with the four rules that
        # force the prose to be as small as possible removed, and the reading
        # level a floor instead of a ceiling. The steered set drops the three
        # directions whose requirement is gone (short sentences, simple
        # register, simple syntax); the rest still apply.
        if list(args.constraints) == list(EN_TASK_CONSTRAINTS):
            args.constraints = list(EN_MIDDLE_CONSTRAINTS)
        if list(args.steer_vectors) == list(EN_STEER_VECTORS):
            # With the middle-school pairs there are two more directions to
            # steer: the reading floor itself, which the children's pairs cannot
            # express, and verbs-rather-than-adverbs, whose requirement is in
            # this rule list.
            args.steer_vectors = list(EN_MIDDLE_REGISTER_STEER_VECTORS
                                      if args.pairs == "middle"
                                      else EN_MIDDLE_STEER_VECTORS)
        args.max_opener_uses = EN_MIDDLE_MAX_OPENER_USES
        # Four settings that have to move together with this rule set, and were
        # passed by hand on the command line for the first three rounds of it.
        # Leaving one of them out silently produces a different task: the story
        # length the model is allowed, the point at which an over-long
        # generation is cut off, and how often a word may be reused all reach
        # the prompt or the generation. A run that omitted `--max-words 200`
        # stopped every story at 97 words while the instruction asked for 150,
        # and its numbers were not comparable with the rounds before it.
        # Anything given explicitly still wins; these only fill in defaults.
        if args.max_words == EN_MAX_WORDS:
            args.max_words = EN_MIDDLE_MAX_WORDS
        if args.max_word_uses == DEFAULT_MAX_WORD_USES:
            args.max_word_uses = EN_MIDDLE_MAX_WORD_USES
        if args.max_new_tokens == ap.get_default("max_new_tokens"):
            args.max_new_tokens = EN_MIDDLE_MAX_NEW_TOKENS
        if args.max_adverbs == DEFAULT_MAX_ADVERBS:
            args.max_adverbs = EN_MIDDLE_MAX_ADVERBS
        if args.min_sensory == DEFAULT_MIN_SENSORY:
            args.min_sensory = EN_MIDDLE_MIN_SENSORY
        if args.present_ratio == DEFAULT_PRESENT_RATIO:
            args.present_ratio = EN_MIDDLE_PRESENT_RATIO


    if args.dry_run:
        # Parsing the flags is the easy half. Two runs have now reached Kaggle,
        # started a session and downloaded a model before dying on a suite that
        # passed an argument make_plan does not take -- an error that costs
        # minutes there and milliseconds here. So build the suite too, against
        # stand-in vectors of the right shape, which exercises make_plan and
        # SteeringPlan.build all the way to a run id.
        import torch as _t

        from noiseegra.steering_vectors import SteeringVectorSet as _SVS

        print("arguments parse. suites requested: " + " ".join(args.suite))
        print(f"model {args.model}, {args.stories} stories, "
              f"constraints {len(args.constraints)}, "
              f"steer vectors {list(args.steer_vectors)}")
        # The settings that reach the prompt or the generation but are not named
        # by any flag on a typical command line. A launch that left one of them
        # at the children's-task default cut every story off at 97 words while
        # the instruction asked for 150, and nothing said so until the run was
        # twenty minutes in.
        print(f"directions from the {args.pairs} contrast pairs; "
              f"story target {args.story_target} words, length rule "
              f"{args.max_words} words, generation stopped at "
              f"{int(1.5 * args.max_words)} words or {args.max_new_tokens} tokens, "
              f"a word may be reused {args.max_word_uses} times")
        _lo, _hi = args.layers if args.layers else EN_MODEL_LAYER_RANGES[args.model]
        _layers = list(range(_lo, _hi))
        _dim, _rank = 64, max(args.protect_rank, 1)
        _t.manual_seed(0)
        _vecs = _SVS(
            vectors={n: {l: _t.randn(_dim) for l in _layers} for n in args.steer_vectors},
            components={n: {l: _t.linalg.qr(_t.randn(_dim, _rank))[0] for l in _layers}
                        for n in args.steer_vectors},
            positives={n: {l: _t.randn(_dim) for l in _layers} for n in args.steer_vectors},
        )
        args.targets = _vecs.positives
        args.offset_basis = {l: _t.linalg.qr(_t.randn(_dim, 24))[0] for l in _layers}
        args.amplify_basis, args.amplify_mean = args.offset_basis, None
        args.gate_thresholds = {"none": 0.0}
        args.direction_source = "extracted"
        from noiseegra.constraint_control import ConstraintController as _CC

        args.controller = _CC()
        _total = 0
        for _suite in (["core", "ortho", "alpha", "gate", "beta", "loo"]
                       if "all" in args.suite else args.suite):
            _built, _desc = build_suite(_suite, _vecs, _layers,
                                        list(args.steer_vectors), 1.5, args)
            _ids = {_spec_to_run_id(args.model, sp) for sp in make_specs(
                *[({"mode": it} if isinstance(it, str) else dict(it)) for it in _built])}
            if len(_ids) != len(_built):
                raise SystemExit(
                    f"suite {_suite}: {len(_built)} conditions collapse to "
                    f"{len(_ids)} run ids, so two of them would share a file")
            _total += len(_built)
            print(f"  suite {_suite}: builds, {len(_built)} conditions")
        print(f"{_total} conditions x {args.stories} stories = "
              f"{_total * args.stories} generations")
        return

    out = Path(args.out) / args.model
    out.mkdir(parents=True, exist_ok=True)
    state_path = out / "state.json"
    state = load_state(state_path)

    hf_id = EN_MODEL_HF_IDS[args.model]
    model_id = args.model_id or hf_id
    lo, hi = args.layers if args.layers else EN_MODEL_LAYER_RANGES[args.model]
    layers = list(range(lo, hi))
    dtype_arg = None if args.dtype == "auto" else getattr(torch, args.dtype)

    # The constraint thresholds and the prompt selection are baked into the text
    # the model is given, but not into the run id. Reusing a checkpoint under
    # different values would silently mix stories written to different
    # instructions, so the task setup is pinned on first write.
    task = {
        "task": args.task,
        "constraints": list(args.constraints),
        "steer_vectors": list(args.steer_vectors),
        "min_words": args.min_words,
        "max_words": args.max_words,
        "max_grade": args.max_grade,
        "num_prompts": args.num_prompts if args.task == "scenario" else 1,
        "stories": args.stories if args.task == "generic" else args.stories_per_prompt,
        "prompt_seed": args.prompt_seed,
        "prompt_split": args.prompt_split,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        # Not part of the prompt, but part of what the steered stories are: two
        # pair sets give different directions, so stories from each are no more
        # comparable than stories written to different instructions.
        "pairs": args.pairs,
        # Every threshold below is written into the requirement list the model is
        # given -- "no word begins more than N sentences" names its N in the
        # prompt. Changing one changes the instruction, so stories written before
        # and after are not comparable. They were missing from this dict, so that
        # change would have been made silently.
        "max_opener_uses": args.max_opener_uses,
        "max_word_uses": args.max_word_uses,
        "min_grade": args.min_grade,
        "story_target": args.story_target,
        "max_adverbs": args.max_adverbs,
        "min_sensory": args.min_sensory,
        "present_ratio": args.present_ratio,
    }
    # A checkpoint written before a setting was recorded does not carry it, and
    # there is no way to know what that run used. Comparing such a key would
    # refuse every resume of an older run the first time a key is added, so only
    # keys the checkpoint actually holds are compared. The merged setup is
    # written back, so the first run after a key is added records it and every
    # run after that is held to it.
    prev = state.get("task")
    if prev is None:
        state["task"] = task
        save_state(state_path, state)
    elif any(prev[k] != v for k, v in task.items() if k in prev) \
            and not args.allow_task_change:
        changed = [f"    {k}: {prev[k]!r} -> {task[k]!r}"
                   for k in task if k in prev and prev[k] != task[k]]
        raise SystemExit(
            f"{state_path} holds stories generated under a different task setup:\n"
            + "\n".join(changed)
            + "\n\n  These values go into the prompt itself, so old and new stories are not\n"
              "  comparable. Either point --out at a fresh directory, or pass\n"
              "  --allow-task-change if you are certain you want them mixed."
        )
    else:
        # Record any setting this checkpoint predates, so it is pinned from now on.
        if any(k not in prev for k in task):
            state["task"] = {**prev, **task}
            save_state(state_path, state)

    args.keep_directions = {}
    for item in args.keep:
        name, _, sign = item.partition("=")
        args.keep_directions[name] = float(sign or 1.0)
    # --keep only matters to the suites that read it. Its default names two
    # directions, so a run that steers a different set was refused outright for a
    # flag it never uses.
    unknown = [n for n in args.keep_directions if n not in args.steer_vectors]
    if unknown and "select" in args.suite:
        raise SystemExit(f"--keep names {unknown} are not in --steer-vectors "
                         f"{list(args.steer_vectors)}")
    for n in unknown:
        args.keep_directions.pop(n)

    word_budget = (args.word_budget if args.word_budget is not None
                   else int(1.5 * args.max_words))
    word_budget = word_budget if word_budget > 0 else None

    print(f"model       : {model_id}")
    print(f"layers      : {layers}")
    print(f"constraints : {args.constraints}")
    print(f"word budget : "
          + (f"{word_budget} words, then the generation is stopped where it is "
             f"({3 * args.max_words // args.max_words}x the longest legal story)"
             if word_budget else "none; generations run to the token cap"))
    print(f"out         : {out}")
    print(f"resuming    : {sum(len(v) for v in state['runs'].values())} stories already saved\n")

    # ---- the task -------------------------------------------------------- #
    # The checker is built first and the prompt is generated from its
    # `requirements()`, so the sentence the model is given and the rule it is
    # scored against are the same object. A threshold cannot change in one place
    # and not the other.
    checker = EnglishConstraintChecker(
        min_words=args.min_words, max_words=args.max_words,
        max_grade_level=args.max_grade,
        min_grade_level=args.min_grade,
        max_word_uses=args.max_word_uses,
        max_opener_uses=getattr(args, "max_opener_uses", DEFAULT_MAX_OPENER_USES),
        max_adverbs=args.max_adverbs,
        min_sensory=args.min_sensory,
        present_ratio_threshold=args.present_ratio,
        constraints=list(args.constraints),
    )

    if args.task == "generic":
        prompts = ["<generic instruction>"]
        messages = [
            wp.build_middle_messages(checker.requirements(), args.constraints,
                                     target=args.story_target)
            if args.constraint_set == "middle" else
            wp.build_generic_messages(checker.requirements(), args.constraints)]
        stories_per_prompt = args.stories
        print(f"task: one generic instruction, {len(args.constraints)} requirements, "
              f"{stories_per_prompt} stories in a single group")
        print("\n" + messages[0][1]["content"] + "\n")
    else:
        prompts = state.get("prompts")
        if not prompts or len(prompts) != args.num_prompts:
            prompts = wp.load_prompts(
                args.num_prompts, seed=args.prompt_seed, split=args.prompt_split,
                cache=out / "prompts.json",
            )
            state["prompts"] = prompts
            save_state(state_path, state)
        messages = [
            wp.build_messages(t, args.constraints,
                              requirements=checker.requirements())
            for t in prompts
        ]
        stories_per_prompt = args.stories_per_prompt
        print(f"task: {len(prompts)} WritingPrompts scenarios, e.g.:")
        for t in prompts[:3]:
            print(f"   - {t[:110]}{'...' if len(t) > 110 else ''}")

    # Loading is deferred: with the steering vectors and the activation scale both
    # cached, a fully generated model needs no weights at all and exits in seconds.
    holder = {"model": None, "dtype": None}

    def get_model():
        if holder["model"] is None:
            if weights_are_cached(model_id):
                print("\nloading model from local cache ...", flush=True)
            else:
                # Kaggle wipes ~/.cache between sessions even when /kaggle/working is
                # restored, so a resumed run re-downloads the weights. Say so, because
                # otherwise this looks like a hang.
                print(
                    f"\n{model_id} is not in the local cache. Downloading the weights "
                    "(~14-24 GB) before anything can run.\n"
                    "  This is a download, not a hang. Progress appears below; if it "
                    "falls under ~1 MB/s, interrupt and re-run -- it restarts at full "
                    "speed and no generated stories are lost.",
                    flush=True,
                )
            print("", flush=True)
            m = build_model(model_id, dtype=dtype_arg, wrapper_for=hf_id)
            if args.thinking != "default":
                m.enable_thinking = (args.thinking == "on")
                print(f"  reasoning blocks: {'allowed' if m.enable_thinking else 'off'}")
            d = next(m.model.parameters()).dtype
            if torch.cuda.is_available() and d not in (torch.float16, torch.bfloat16):
                raise SystemExit(f"model loaded in {d}; upgrade transformers (>=4.56).")
            print(f"  dtype={d}")
            holder["model"], holder["dtype"] = m, d
        return holder["model"]

    suites_req = (["core", "ortho", "alpha", "gate", "beta", "loo"]
                  if "all" in args.suite else list(args.suite))

    # A baseline-only run is unmodified generation: no steering vectors, no
    # activation scale, no entropy measurement. Skipping all three means it starts
    # generating immediately, and it runs on a model the extraction pair file has
    # never been tried on.
    if args.baseline_top_p is not None and args.baseline_top_p < 0:
        args.baseline_top_p = None
    if args.baseline_top_k is not None and args.baseline_top_k < 0:
        args.baseline_top_k = None
    steering_needed = any(name not in ("baseline", "sampling") for name in suites_req)
    vectors, rms_scale = None, 0.0

    # ---- steering vectors ------------------------------------------------- #
    if not steering_needed:
        print("\nbaseline only: skipping steering-vector extraction and activation "
              "scale calibration.")
    if steering_needed:
        # The context is part of what the direction *is*, so it is part of the
        # cache key. Reusing a file extracted under a different conversation would
        # silently answer a question nobody asked.
        ctx_tag = "" if args.extraction_context == "generic" else f"_{args.extraction_context}"
        win_tag = "" if args.read_window == "all" else f"_{args.read_window}{args.read_tokens}"
        # The pair file is part of what the direction is, exactly as the context
        # is. Without it in the name, a run asking for the middle-school pairs
        # would load a cached file extracted from the children's pairs and report
        # the result under the new name.
        pair_tag = "" if args.pairs == "children" else f"_{args.pairs}"
        vec_path = out / f"steering{ctx_tag}{win_tag}{pair_tag}_{args.model}.pt"
        if vec_path.is_file():
            vectors = SteeringVectorSet.load(vec_path)
            print(f"steering vectors: loaded from {vec_path.name}")
        else:
            print("steering vectors: extracting (once) ...")
            if args.extraction_context == "task":
                ex_system = messages[0][0]["content"]
                ex_user = messages[0][1]["content"]
            else:
                ex_system, ex_user = wp.SYSTEM_PROMPT, wp.EXTRACTION_PROMPT
            print(f"  read in the {args.extraction_context} context")
            vectors = SteeringVectorExtractor(get_model()).extract(
                load_pairs(PAIR_SETS[args.pairs]), layers,
                system=ex_system, user=ex_user,
                window=args.read_window, window_tokens=args.read_tokens,
                pca_rank=args.pca_rank, only=args.steer_vectors, verbose=False,
            )
            vectors.save(vec_path)
            print(f"steering vectors: saved to {vec_path.name}")
        args.direction_source = "random" if args.random_directions else "extracted"
        if args.random_directions:
            gen = torch.Generator().manual_seed(1234)
            for name, per in vectors.vectors.items():
                for layer, vec in per.items():
                    draw = torch.randn(vec.shape, generator=gen, dtype=torch.float32)
                    per[layer] = draw / draw.norm().clamp_min(1e-12) * vec.norm()
            for name, per in vectors.components.items():
                for layer, comp in per.items():
                    draw = torch.randn(comp.shape, generator=gen, dtype=torch.float32)
                    per[layer] = torch.linalg.qr(draw)[0]
            print("  directions replaced with random draws of the same norm "
                  "(control arm; the extraction is not being used)")

        for name in args.steer_vectors:
            cons = [vectors.diagnostics[name][l]["consistency"] for l in layers]
            flag = "" if min(cons) > 0.3 else "   <-- weak, direction may be mostly noise"
            print(f"  {name:<16} agreement across pairs: {min(cons):.2f}-{max(cons):.2f}{flag}")

        # ---- activation scale, keyed by model+layers -------------------------- #
        cal_key = f"{args.model}|{lo}-{hi}"
        if cal_key not in state["rms_scale"]:
            print("\ncalibrating activation scale ...")
            rms = RMSCalibrator(get_model()).collect_block_rms(messages[0], layers=layers)
            state["rms_scale"][cal_key] = float(np.median(list(rms.values())))
            save_state(state_path, state)
        rms_scale = state["rms_scale"][cal_key]
        if not (rms_scale > 0) or rms_scale != rms_scale:
            raise SystemExit(
                f"calibration returned {rms_scale}; the forward pass produced non-finite "
                "activations. This model probably cannot run in float16 -- retry with "
                "--dtype bfloat16 (slower on a T4, but correct)."
            )
        print(f"activation scale [{cal_key}] = {rms_scale:.4g}  "
              f"(noise {args.alpha * rms_scale:.4g}, steering {args.beta * rms_scale:.4g})")
        if holder["dtype"] == torch.float16 and rms_scale > 100:
            print(f"[warn] activation scale {rms_scale:.4g} is large for float16 "
                  "(max representable 65504). If the stories come out empty or garbled, "
                  "rerun with --dtype bfloat16.")

    # Where text that satisfies each constraint sits on its own axis, taken from
    # the positive side of that constraint's contrast pairs. Only feedback steering
    # needs it; older vector files predate it and say so rather than failing later.
    args.targets = getattr(vectors, "positives", None) if vectors is not None else None
    if "feedback" in suites_req and not args.targets:
        raise SystemExit(
            "--suite feedback needs the positive-side activations, which this "
            "steering-vector file predates. Delete it and let the run re-extract."
        )

    # The controller reads the same thresholds the checker scores against, so the
    # quantity being steered and the quantity being reported cannot drift apart.
    from noiseegra.constraint_control import ConstraintController  # noqa: E402

    args.controller = ConstraintController(
        word_range=checker.word_range,
        sentence_range=checker.sentence_range,
        sentence_word_range=checker.sentence_word_range,
        n_quotes=checker.n_quotes,
    )

    # ---- directions a per-story offset is allowed to use -------------------- #
    # Cached under the basis kind, because the two are different sets of
    # directions and a run that mixed them would be unreadable.
    args.offset_basis = None
    args.amplify_basis = None
    args.amplify_mean = None
    if BASIS_SUITES & set(suites_req):
        kind = args.offset_basis_kind
        pc_path = out / f"actpcs_{kind}_{args.model}.pt"
        legacy = out / f"actpcs_{args.model}.pt"
        if kind == "step" and not pc_path.is_file() and legacy.is_file():
            pc_path = legacy
        cached = None
        if pc_path.is_file():
            cached = torch.load(pc_path, map_location="cpu", weights_only=False)
            if kind in ("story", "prompt") and not isinstance(cached, StoryAxes):
                # Written before the estimate carried the mean its directions are
                # measured from. The directions themselves are still right, but
                # amplification cannot use them, so take it again.
                print(f"activation basis: {pc_path.name} predates the stored mean "
                      "activation; re-estimating")
                cached = None
            else:
                print(f"activation basis ({kind}): loaded from {pc_path.name}")
        if cached is not None:
            pass
        elif kind == "prompt":
            print("activation basis: reading the instruction's own hidden states "
                  "(one forward pass, nothing generated) ...", flush=True)
            cached = collect_prompt_pcs(
                get_model(), messages[0], layers, rank=args.offset_rank,
            )
            torch.save(cached, pc_path)
            print(f"activation basis (prompt): saved to {pc_path.name}")
        elif kind == "story":
            print(f"activation basis: sampling {args.offset_basis_stories} unsteered "
                  "stories to find the directions they differ along (once) ...",
                  flush=True)
            cached = collect_story_pcs(
                get_model(), messages[0], layers,
                n_stories=args.offset_basis_stories,
                rank=args.offset_rank,
                max_new_tokens=args.offset_basis_tokens,
            )
            torch.save(cached, pc_path)
            print(f"activation basis (story): saved to {pc_path.name}")
        else:
            print("activation basis: estimating decode-step principal components "
                  "(once) ...", flush=True)
            cached = collect_block_pcs(
                get_model(),
                (messages * 4)[:4],
                layers, rank=args.offset_rank,
                max_new_tokens=args.offset_basis_tokens,
            )
            torch.save(cached, pc_path)
            print(f"activation basis (step): saved to {pc_path.name}")

        # The story-level estimate carries the mean the directions are measured
        # from, which amplification needs; the decode-step one is a bare basis and
        # is centred on zero.
        if isinstance(cached, StoryAxes):
            args.offset_basis, args.amplify_basis = cached.basis, cached.basis
            args.amplify_mean = cached.mean
        else:
            args.offset_basis = args.amplify_basis = cached
        rank0 = args.offset_basis[sorted(args.offset_basis)[0]].shape[1]
        print(f"  the perturbation may move along {rank0} directions per layer, "
              "before the constraint directions are projected out")
        if "amplify" in suites_req and args.amplify_mean is None:
            raise SystemExit(
                "--suite amplify needs the average activation its directions are "
                "measured from, which only the story-level basis carries. Re-run "
                "with --offset-basis story."
            )

    # ---- entropy gate threshold, measured once per model ------------------- #
    # In nats, and nats are not comparable across models or tokenisers, so the
    # threshold is a quantile of this model's own decode entropy -- the same
    # reasoning as calibrating the noise scale to the model's own block RMS.
    # Measured unsteered, so it describes the model and not the condition.
    args.gate_thresholds = {"none": 0.0}
    # Any suite that sets a per-arm gate needs the thresholds measured, not just
    # the one called "gate". `gatedwrite` did not, so every level resolved to
    # 0.0 and three arms that differed only in their gate generated identical
    # stories under run ids that named the gate. Same silent null as a suite
    # missing from BASIS_SUITES.
    if GATE_SUITES & set(suites_req) or args.gate != "none":
        gate_key = f"{args.model}|{lo}-{hi}"
        store = state.setdefault("entropy", {})
        if gate_key not in store:
            print("\nmeasuring the model's own decode entropy ...", flush=True)
            ent = collect_decode_entropies(
                get_model(), messages[0], n_samples=args.gate_samples,
                max_new_tokens=args.max_new_tokens, temperature=args.temperature,
            )
            store[gate_key] = {
                "n_steps": len(ent),
                "quantiles": {lv: gate_threshold(ent, lv) for lv in GATE_LEVELS},
            }
            save_state(state_path, state)
        rec = store[gate_key]
        args.gate_thresholds.update(rec["quantiles"])
        print(f"decode entropy over {rec['n_steps']} unsteered steps: "
              + ", ".join(f"{lv} gate at {v:.3f} nats" for lv, v in
                          sorted(rec["quantiles"].items()) if v))

    # ---- conditions -------------------------------------------------------- #
    suites = suites_req
    items = ["baseline"] if args.with_baseline else []
    for suite in suites:
        built, desc = build_suite(suite, vectors, layers, args.steer_vectors, rms_scale, args)
        items.extend(built)
        print(f"  suite {suite}: {desc} ({len(built)} runs)")

    # A gate asked for on the command line applies to every condition that
    # actually perturbs. --suite gate sets its own per-arm gates and is left alone.
    if args.gate != "none":
        thr = args.gate_thresholds.get(args.gate, 0.0)
        for it in items:
            plan = it.get("plan") if isinstance(it, dict) else None
            if plan is not None and (plan.noise_alpha > 0 or plan.offset_gamma > 0):
                plan.gate_threshold = thr
                plan.gate_level = args.gate
        print(f"  {describe_gate(args.gate, thr)}")

    normalised = []
    for it in items:
        d = {"mode": it} if isinstance(it, str) else dict(it)
        d.setdefault("temperature", args.temperature)
        # Decoding settings apply to the steered conditions too. Every steered run
        # so far was at temperature 1.0, where this model loops on 32% of stories;
        # steering was being measured in the degenerate regime, against a baseline
        # that was also in it. Raising the temperature both stops the looping and
        # triples the diversity, so the honest question is what steering adds on
        # top of a decoding setting worth using, not on top of the default.
        if args.top_p is not None:
            d.setdefault("top_p", args.top_p)
        if args.top_k is not None:
            d.setdefault("top_k", args.top_k)
        d.setdefault("max_new_tokens_plan", args.max_new_tokens)
        d.setdefault("max_new_tokens_story", args.max_new_tokens)
        normalised.append(d)

    if args.shard:
        try:
            idx, total = (int(x) for x in args.shard.split("/"))
        except ValueError:
            raise SystemExit(f"--shard wants I/N, got {args.shard!r}")
        if not (0 <= idx < total):
            raise SystemExit(f"--shard {args.shard}: I must be in [0, N)")
        kept = normalised[idx::total]
        print(f"shard {idx} of {total}: {len(kept)} of {len(normalised)} conditions")
        normalised = kept

    specs, run_ids, seen = [], [], set()
    for spec in make_specs(*normalised):
        rid = _spec_to_run_id(args.model, spec)
        if rid in seen:
            continue
        seen.add(rid)
        specs.append(spec)
        run_ids.append(rid)
        state["runs"].setdefault(rid, {})

    # A run id that names an estimated basis (obstory/obprompt) while no basis was
    # built is the silent fault that made r16 and r17 isotropic: the offset falls
    # back to a random draw and the id lies about it. Crash instead. `obiso` and
    # the bare `step` basis are legitimately basis-free and do not trip this.
    if args.offset_basis is None:
        lying = [r for r in run_ids if "__obstory" in r or "__obprompt" in r]
        if lying:
            raise SystemExit(
                f"{len(lying)} condition(s) ask for an estimated offset basis but none "
                f"was built, so the offset would silently be isotropic. First: "
                f"{lying[0]}. Add the suite to BASIS_SUITES.")

    P, K = len(prompts), stories_per_prompt
    total = len(specs) * P * K
    print(f"\n{len(specs)} conditions x {P} prompts x {K} stories = {total} generations")
    for rid in run_ids:
        print(f"  [{len(state['runs'][rid]):>4}/{P * K}] {rid}")

    remaining = sum(
        1 for rid in run_ids for k in range(K) for p_idx in range(P)
        if f"{p_idx}:{k}" not in state["runs"][rid]
    )
    print(f"\n{remaining} of {total} still to generate.")

    # Conditions run one at a time and are scored the moment they finish, so the
    # sweep produces readable results as it goes instead of only at the end.
    scorer = None
    if args.diversity:
        from noiseegra.creativity_metrics import CreativityScorer

        # Built now, loaded on first use: loading it here, before the language
        # model, would take the empty GPU and leave the generator to share it.
        scorer = CreativityScorer(
            ["placeholder one", "placeholder two"],
            embedding_model=args.embedding_model,
            truncate_words=args.truncate_words,
            device=args.embedding_device,
        )
        print(f"diversity scoring: {scorer.embedding_model}, "
              f"loaded on first use", flush=True)

    done, t0 = total - remaining, time.time()
    started = done
    rows = []

    live_header, live_row = live_table(checker.constraints, args.diversity)
    print("\n" + live_header)
    print("-" * len(live_header), flush=True)

    # The cheap coherence heuristics (no model, milliseconds a story), used to
    # peek at arms as they generate and to abort ones that are plainly dead. The
    # judgement is deliberately blunt: it only ever fires on an arm where nearly
    # every story is broken, never on an ordinarily imperfect one.
    from noiseegra.coherence import CoherenceFilter as _CoherenceFilter
    _arm_filter = _CoherenceFilter()

    def _arm_is_dead(rid):
        """True when almost every story generated so far in this arm is broken."""
        texts = list(state["runs"][rid].values())
        if len(texts) < args.abort_check_at:
            return False
        flagged = sum(1 for t in texts if not _arm_filter.check(t).ok)
        frac = flagged / len(texts)
        if frac >= args.abort_flag_frac:
            print(f"  [ABORTED] {rid[-60:]}: {flagged} of {len(texts)} opening "
                  f"stories fail the coherence checks; skipping the rest of this "
                  f"condition", flush=True)
            return True
        return False

    for spec, rid in zip(specs, run_ids):
        # Lay this condition's whole set of per-story perturbations out before any
        # of them is used, so they can be chosen to cover the subspace instead of
        # colliding by chance. A no-op unless the plan asked for it.
        plan = getattr(spec, "steering_plan", None)
        if plan is not None and hasattr(plan, "plan_offsets"):
            plan.plan_offsets(K, seed=0)
        missing = [(p_idx, k) for k in range(K) for p_idx in range(P)
                   if f"{p_idx}:{k}" not in state["runs"][rid]]
        # A resumed arm that already failed the abort check stays aborted.
        if missing and args.abort_broken_arms and _arm_is_dead(rid):
            missing = []
        if missing:
            model = get_model()
            mode = _spec_mode(spec)
            # The model's own next-token uncertainty, averaged over the story.
            # Read off the raw scores, so raising the temperature does not move
            # it: it separates an intervention that makes the model less certain
            # from one that moves it elsewhere while leaving it as certain.
            probes = [] if args.record_uncertainty else None
            for p_idx, k in missing:
                text = generate_one(model, spec, mode, messages[p_idx],
                                    seed_for(p_idx, k), args.max_new_tokens,
                                    max_words=word_budget, story_index=k,
                                    entropy_out=probes)
                state["runs"][rid][f"{p_idx}:{k}"] = text
                if probes is not None and probes[-1].get("history"):
                    state.setdefault("uncertainty", {}).setdefault(rid, []).append(
                        float(np.mean(probes[-1]["history"])))
                save_state(state_path, state)
                done += 1
                rate = (time.time() - t0) / max(done - started, 1)
                print(f"  [{done:>4}/{total}] {rid[-34:]:<34} prompt {p_idx:>2} story {k:>2}"
                      f" | {rate:5.1f}s | ETA {(total - done) * rate / 60:6.1f} min", flush=True)
                if args.peek_stories and p_idx == 0 and k < args.peek_stories:
                    body = " ".join(text.split())
                    print(f"  [peek] {rid[-34:]} story {k}: {body[:170]}", flush=True)
                if (args.abort_broken_arms
                        and len(state["runs"][rid]) == args.abort_check_at
                        and _arm_is_dead(rid)):
                    break
            torch.cuda.empty_cache()

        cells = state["runs"][rid]
        keys = sorted(cells, key=lambda x: tuple(int(i) for i in x.split(":")))
        stories = [cells[key] for key in keys]
        pidx = [int(key.split(":")[0]) for key in keys]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            row = score_condition(stories, pidx, checker, scorer)
        except Exception as exc:
            # Every story is already on disk. Losing the scores for one condition
            # is an inconvenience; losing the run at story 300 of 400 is not.
            import traceback

            print(f"\n  [warn] diversity scoring failed for {rid}:", flush=True)
            traceback.print_exc()
            print("  The stories themselves are saved. Score them afterwards with\n"
                  f"    python scripts/score_english.py --input-dir {out} --diversity\n",
                  flush=True)
            row = score_condition(stories, pidx, checker, None)
        row["run"] = rid
        rows.append(row)
        write_csvs(out, state)
        print(live_row(condition_label(rid), row), flush=True)

    print("\n" + "=" * len(live_header))
    print(f"  {args.model}: {P} prompt{'s' if P != 1 else ''} x {K} stories per condition")
    for line in plan_summary([r["run"] for r in rows]):
        print(f"  {line}")
    print("  perturbation magnitudes are multiples of the model's own activation scale")
    print("=" * len(live_header))
    print(live_header)
    print("-" * len(live_header))
    for row in rows:
        print(live_row(condition_label(row["run"]), row))
    print("\nwhat each requirement column means, and what a story has to do to pass it")
    for line in constraint_legend(checker):
        print("  " + line)

    summary = out / "live_scores.csv"
    with summary.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["run", "label"] + [k for k in rows[0] if k != "run"])
        w.writeheader()
        for row in rows:
            w.writerow({**row, "label": condition_label(row["run"])})
    print(f"\nwrote {summary}")


if __name__ == "__main__":
    main()
