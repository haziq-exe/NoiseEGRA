#!/usr/bin/env python
"""One workbook holding every method worth keeping, its stories, and its scores.

    python scripts/build_method_workbook.py --out method_review.xlsx

A table of scores says how often a rule is broken and nothing about whether the
writing is any good, and three results in this project were wrong until somebody
read the stories. So this puts both in one file: what each method does and how to
run it, every number measured the same way, and the stories themselves beside the
untouched model's and both raised-temperature baselines' so they can be read
against each other.

Everything here comes from one model, one instruction and one sampling
temperature, and every run listed shares a byte-identical prompt -- checked, not
assumed, because the project has two prompt families and mixing them would make
the comparison meaningless.

Each story index is seeded the same way in every run, so story 7 of one method
and story 7 of the baseline are the same draw under different conditions and can
be read side by side.

Numbers follow the project's usual order: drop the stories the coherence checks
reject, cut each story to its first 40 words, trim the ones sitting far from the
rest of their own set, and pool every condition at the size of the smallest.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from openpyxl import Workbook  # noqa: E402
from openpyxl.styles import Alignment, Font, PatternFill  # noqa: E402
from openpyxl.utils import get_column_letter  # noqa: E402

from compare_conditions import opener_share, stories_from_run  # noqa: E402
from noiseegra.coherence import (  # noqa: E402
    CoherenceFilter, opens_with_a_title, trim_lead, trim_title,
)
from noiseegra.constraint_metrics_en import (  # noqa: E402
    MIDDLE_CONSTRAINTS, EnglishConstraintChecker, opens_in_the_wrong_tense,
)
from noiseegra.defaults import (  # noqa: E402
    EN_MIDDLE_MAX_ADVERBS, EN_MIDDLE_MAX_OPENER_USES, EN_MIDDLE_MAX_WORD_USES,
    EN_MIDDLE_MAX_WORDS, EN_MIDDLE_MIN_SENSORY, EN_MIDDLE_PRESENT_RATIO,
)
from noiseegra.readability import uncommon_word_share  # noqa: E402

import re  # noqa: E402

SENTENCE = re.compile(r"[^.!?]+[.!?]")
from noiseegra.structure import (  # noqa: E402
    _set_vectors, content_lemma_sets, pos_trigram_vectors, rarefied_vendi,
    trim_isolated,
)

TRUNCATE = 40
INTERVAL_DRAWS = 200

COMMON = (
    "--model Qwen3-1.7B --task generic --constraint-set middle "
    "--stories {stories} --layers 6 14 --temperature 1.0 --story-target 150 "
    "--truncate-words 40 --pairs children --offset-basis story"
)

# ---------------------------------------------------------------------------
#  Everything worth keeping, in the order it is presented.
#
#  `family` groups methods that differ only in one setting, so a reader can see
#  at a glance which rows are a sweep of one idea and which are separate ideas.
# ---------------------------------------------------------------------------
BASELINES = [
    dict(
        id="B1", label="Untouched model", run="r42-final",
        frag="Qwen3-1.7B__BASELINE", family="Baseline", stories=200,
        summary="The model answering the instruction, nothing changed.",
        works=(
            "Ordinary sampling at temperature 1.0. No steering, no perturbation, "
            "no change to decoding. This is what the instruction alone gets you "
            "and the floor every other row has to beat."
        ),
        different="Nothing is different -- this is the thing to be different from.",
        command="(the BASELINE arm of the run below)",
    ),
    dict(
        id="B2", label="Raised temperature, nucleus sampling", run="r42-final",
        frag="BASELINE__temp1p8__topp0p95", family="Baseline", stories=200,
        summary="Temperature 1.8 with nucleus (top-p) sampling at 0.95.",
        works=(
            "The standard way to buy variety: flatten the next-token distribution "
            "by raising the temperature, then cut the tail so the flattening does "
            "not let genuinely bad tokens through. Costs nothing to run."
        ),
        different=(
            "Acts on the probabilities over the next token, after the model has "
            "finished computing. It cannot know anything about the story so far "
            "beyond what the distribution already says."
        ),
        command="(the temp1.8 top-p arm of the run below)",
    ),
    dict(
        id="B3", label="Raised temperature, top-k sampling", run="r42-final",
        frag="BASELINE__temp1p8__topk40", family="Baseline", stories=200,
        summary="Temperature 1.8 with top-k sampling at k=40.",
        works=(
            "As above but the tail is cut at a fixed count rather than a "
            "probability mass. On this task it is the stronger of the two "
            "baselines for variety of what happens, so it is the one to beat."
        ),
        different=(
            "Same as nucleus: a decoding-time change, with no access to the "
            "model's internal state."
        ),
        command="(the temp1.8 top-k arm of the run below)",
    ),
]

METHODS = [
    dict(
        id="M01", label="Per-token noise 0.1, no displacement", run="r44-pertoken",
        frag="a0p1__k36__bud2__tail8", family="Per-token noise", stories=100,
        summary=(
            "The constraint push, plus fresh Gaussian noise added to the hidden "
            "state at every decode step. No per-story displacement."
        ),
        works=(
            "Four constraint directions are extracted by contrast (see 'How the "
            "push works' on the read-me sheet), made mutually orthogonal, summed "
            "to a fixed total strength of 2, and added to the residual stream at "
            "layers 6 to 13 -- both at the prompt positions and at every decode "
            "step. On top of that, independent noise of size 0.1 is drawn afresh "
            "at each step. This is the mechanism from the published workshop "
            "paper, run under the push for the first time."
        ),
        different=(
            "Changes the model's internal state rather than its output "
            "probabilities. Gives the best requirement compliance of anything "
            "measured -- but buys no variety at all, because fresh noise at every "
            "step averages out over a story, so the stories it produces are no "
            "more different FROM EACH OTHER than the untouched model's."
        ),
        command=(
            "scripts/kaggle_run.sh r44-pertoken --profile <account> --shards 2 -- \\\n"
            "  " + COMMON.format(stories=100) + " \\\n"
            "  --peek-stories 4 --abort-broken-arms --offset-draw-shape manifold \\\n"
            "  --suite pertoken --steer-vectors present_tense sensory named_character no_heading \\\n"
            "  --steer-budget 2 --main-gamma 0.15 --alpha-sweep 0.1 0.2"
        ),
    ),
    dict(
        id="M02", label="Round 37: push + displacement 0.15 (old unit)",
        run="r37-noheading", frag="g0p15orth", family="Displacement, old unit",
        stories=100,
        summary=(
            "The first configuration that kept every story coherent. Push plus one "
            "per-story displacement of fixed size."
        ),
        works=(
            "The push as above. Then one displacement per story, drawn from the "
            "subspace the model's own stories differ along (32 sampled stories, "
            "rank 31), projected clear of the constraint directions, added at the "
            "prompt positions only, with the last eight prompt positions left "
            "alone. The size is fixed at 0.15 of the hidden state's own length."
        ),
        different=(
            "The displacement is drawn once per story and held for the whole "
            "story, so it shifts where the story starts from rather than jiggling "
            "each token. That is what makes stories differ FROM EACH OTHER, which "
            "is what a diversity score measures."
        ),
        command=(
            "scripts/kaggle_run.sh r37-noheading --profile <account> --shards 2 -- \\\n"
            "  " + COMMON.format(stories=100) + " \\\n"
            "  --peek-stories 6 --abort-broken-arms --suite boundary \\\n"
            "  --steer-vectors present_tense sensory named_character no_heading \\\n"
            "  --steer-budget 2 --tail-sweep 8 --gamma-sweep 0.125 0.15"
        ),
    ),
    dict(
        id="M03", label="Round 42: same, shaped draw, 200 stories", run="r42-final",
        frag="bud2__odmanifold", family="Displacement, old unit", stories=200,
        summary="Round 37's configuration with the draw shaped like a real story difference.",
        works=(
            "As M02, but the displacement's coefficients are weighted by how far "
            "the sampled stories actually spread along each direction, instead of "
            "treating every direction alike. Run at 200 stories in one run with "
            "all three baselines, which is why the baselines on this sheet come "
            "from here."
        ),
        different=(
            "Same as M02. The shaping makes a displacement of a given size look "
            "more like a real difference between two of the model's own stories."
        ),
        command=(
            "scripts/kaggle_run.sh r42-final --profile <account> --shards 2 -- \\\n"
            "  " + COMMON.format(stories=200) + " \\\n"
            "  --peek-stories 4 --abort-broken-arms --suite final \\\n"
            "  --steer-vectors present_tense sensory named_character no_heading \\\n"
            "  --steer-budget 2 --main-gamma 0.15 --prompt-tail 8 --offset-draw-shape manifold"
        ),
    ),
    dict(
        id="M04", label="Displacement size varying between stories",
        run="r48-varysize", frag="gs0p5", family="Displacement, old unit", stories=100,
        summary="As M03, but each story gets its own displacement size.",
        works=(
            "The size is drawn per story, uniformly within plus or minus 50% of "
            "the nominal size, so the stories fill a ball around the unperturbed "
            "state instead of sitting on a shell around it. The average size is "
            "unchanged."
        ),
        different=(
            "The one property of the displacement that had never been varied. "
            "Spread is what a diversity score measures and a shell has less of it "
            "than a ball."
        ),
        command=(
            "scripts/kaggle_run.sh r48-varysize --profile <account> --shards 2 -- \\\n"
            "  " + COMMON.format(stories=100) + " \\\n"
            "  --peek-stories 4 --abort-broken-arms --offset-draw-shape manifold \\\n"
            "  --suite varysize --steer-vectors present_tense sensory named_character no_heading \\\n"
            "  --steer-budget 2 --main-gamma 0.15 --spread-sweep 0.5 1.0"
        ),
    ),
    dict(
        id="M05", label="Story-register shield, fixed size", run="r50-shield",
        frag="g0p15orth", family="Shield", stories=100,
        summary=(
            "As M03, with the displacement forbidden from moving along the "
            "direction that separates telling a story from talking about the task."
        ),
        works=(
            "Reading the stories that fail shows they are not garbled: the model "
            "refuses, or narrates its own planning ('The user wants a short "
            "story...'). That is a direction, so a contrast set was written for it "
            "-- twelve pairs, positives a line of story, negatives a line of "
            "planning or refusal, matched on word count and sentence length -- and "
            "its direction extracted like any other. It is then never pushed: it "
            "is removed from the subspace the displacement is drawn from, and "
            "every draw is projected clear of it."
        ),
        different=(
            "Nothing else in the project forbids a direction rather than pushing "
            "one. The constraint push is byte-identical with and without it; the "
            "only thing that changes is what the displacement is allowed to do. "
            "Took refusals and leaked planning to zero at this size."
        ),
        command=(
            "scripts/kaggle_run.sh r50-shield --profile <account> --shards 2 -- \\\n"
            "  " + COMMON.format(stories=100) + " \\\n"
            "  --peek-stories 6 --abort-broken-arms --suite boundary --tail-sweep 8 \\\n"
            "  --gamma-sweep 0.15 0.25 0.35 --offset-gamma-spread 0.5 --offset-draw-shape manifold \\\n"
            "  --steer-vectors present_tense sensory named_character no_heading \\\n"
            "  --shield-vectors in_story --steer-budget 2\n"
            "# NOTE: this run's 0.15 arm carries no size variation (the spread flag\n"
            "# did not reach this suite until a later fix); M06 is the fixed version."
        ),
    ),
    dict(
        id="M06", label="Shield + varying size (old unit 0.15)", run="r51-shieldsize",
        frag="g0p15orth", family="Shield", stories=100,
        summary=(
            "The shield and the varying displacement size together. Every story "
            "coherent, compliance well ahead of both baselines."
        ),
        works="M05's shield with M04's per-story size variation, at a nominal size of 0.15.",
        different=(
            "The first arm to reach every story coherent while beating both "
            "raised-temperature baselines on requirement compliance. Variety of "
            "what happens ties nucleus sampling and sits below top-k."
        ),
        command=(
            "scripts/kaggle_run.sh r51-shieldsize --profile <account> --shards 2 -- \\\n"
            "  " + COMMON.format(stories=100) + " \\\n"
            "  --peek-stories 6 --abort-broken-arms --suite boundary --tail-sweep 8 \\\n"
            "  --gamma-sweep 0.15 0.2 0.25 --offset-gamma-spread 0.5 --offset-draw-shape manifold \\\n"
            "  --steer-vectors present_tense sensory named_character no_heading \\\n"
            "  --shield-vectors in_story --steer-budget 2"
        ),
    ),
    dict(
        id="M07", label="Size measured in stories: 0.5 out", run="r56-storyunits",
        frag="g0p5orth", family="Size in story units", stories=100,
        summary="Half as far from the average as one of the model's own stories sits.",
        works=(
            "Identical to M06 except for what the size means. Until this point the "
            "size was a fraction of the hidden state's own length, which has no "
            "relation to how far the model's stories sit from one another: on this "
            "model a real story sits 6.9 from their average at layer 6, and a size "
            "of 0.15 in the old unit works out at 15.0 -- twice as far out as any "
            "story the model wrote. Measured against the stories instead, 1.0 is "
            "one story's distance."
        ),
        different=(
            "Best requirement compliance of any arm with a displacement, every "
            "story coherent, and not one heading or preamble in a hundred "
            "stories. The variety is below both baselines: this is deliberately "
            "the conservative end of the sweep."
        ),
        command=(
            "scripts/kaggle_run.sh r56-storyunits --profile <account> --shards 2 -- \\\n"
            "  " + COMMON.format(stories=100) + " \\\n"
            "  --peek-stories 6 --abort-broken-arms --suite boundary --tail-sweep 8 \\\n"
            "  --gamma-sweep 0.5 1.0 1.5 --offset-norm story --offset-gamma-spread 0.5 \\\n"
            "  --offset-draw-shape manifold \\\n"
            "  --steer-vectors present_tense sensory named_character no_heading \\\n"
            "  --shield-vectors in_story --steer-budget 2"
        ),
    ),
    dict(
        id="M08", label="Size in stories: 1.0 out", run="r56-storyunits",
        frag="g1orth", family="Size in story units", stories=100,
        summary="As far from the average as one of the model's own stories sits.",
        works="As M07 at a size of 1.0.",
        different="As M07. Every story coherent, no formatting failure of any kind.",
        command="(same run as M07, the 1.0 arm)",
    ),
    dict(
        id="M09", label="Size in stories: 1.5 out", run="r56-storyunits",
        frag="g1p5orth", family="Size in story units", stories=100,
        summary="Half again as far out as a real story.",
        works="As M07 at a size of 1.5.",
        different=(
            "Where the drawn displacement starts to show headings again, which is "
            "the first sign of the state leaving the region the model writes from."
        ),
        command="(same run as M07, the 1.5 arm)",
    ),
    dict(
        id="M10", label="Aimed at a real story: 1.0 out", run="r56-anchorunits",
        frag="g1orth", family="Aimed at a story", stories=100,
        summary=(
            "The displacement points at one of the model's own sampled stories "
            "instead of at a random point in the space they span."
        ),
        works=(
            "The 32 sampled stories' activations were previously used to compute "
            "principal components and then thrown away. Kept instead, each story's "
            "displacement aims at a different one of them. The direction is then "
            "one real story's deviation from the average rather than a blend of "
            "thirty-one of them."
        ),
        different=(
            "Worth three to five points of variety of what happens over a drawn "
            "displacement at the same distance. Both move the state equally far; "
            "only one moves it somewhere the model has been."
        ),
        command=(
            "scripts/kaggle_run.sh r56-anchorunits --profile <account> --shards 2 -- \\\n"
            "  " + COMMON.format(stories=100) + " \\\n"
            "  --peek-stories 6 --abort-broken-arms --suite boundary --tail-sweep 8 \\\n"
            "  --gamma-sweep 0.5 1.0 1.5 --offset-norm story --offset-gamma-spread 0.5 \\\n"
            "  --offset-draw-shape anchor \\\n"
            "  --steer-vectors present_tense sensory named_character no_heading \\\n"
            "  --shield-vectors in_story --steer-budget 2"
        ),
    ),
    dict(
        id="M11", label="Aimed at a real story: 1.5 out (100 stories)",
        run="r56-anchorunits", frag="g1p5orth", family="Aimed at a story", stories=100,
        summary="As M10 at one and a half stories out.",
        works="As M10 at a size of 1.5.",
        different="As M10. This is the configuration measured again at 200 stories as M18.",
        command="(same run as M10, the 1.5 arm)",
    ),
    dict(
        id="M12", label="Measured against steered stories: drawn, 1.0 out",
        run="r59-pushdrawn", frag="g1orth", family="Steered frame", stories=100,
        summary=(
            "The 32 stories the displacement is measured against are sampled while "
            "the constraint push is applied, not from the untouched model."
        ),
        works=(
            "The displacement acts during steered generation, so the cloud of "
            "states it is moving within is the cloud of steered stories. Sampling "
            "the basis without the push measures it from the wrong centre, along "
            "directions describing a cloud the state is never in. The push is "
            "applied during sampling exactly as the generation hook applies it."
        ),
        different=(
            "Takes refusals and leaked planning to zero, and gives the first arm in "
            "the project with no formatting failure of any kind: no heading and no "
            "preamble in a hundred stories, every story coherent."
        ),
        command=(
            "scripts/kaggle_run.sh r59-pushdrawn --profile <account> --shards 2 -- \\\n"
            "  " + COMMON.format(stories=100) + " --basis-under-push \\\n"
            "  --peek-stories 6 --abort-broken-arms --suite boundary --tail-sweep 8 \\\n"
            "  --gamma-sweep 1.0 1.5 --offset-norm story --offset-gamma-spread 0.5 \\\n"
            "  --offset-draw-shape manifold \\\n"
            "  --steer-vectors present_tense sensory named_character no_heading \\\n"
            "  --shield-vectors in_story --steer-budget 2"
        ),
    ),
    dict(
        id="M13", label="Steered frame: drawn, 1.5 out", run="r59-pushdrawn",
        frag="g1p5orth", family="Steered frame", stories=100,
        summary="As M12 at one and a half stories out.",
        works="As M12 at a size of 1.5.",
        different="Every story coherent, with variety of wording level with the baselines.",
        command="(same run as M12, the 1.5 arm)",
    ),
    dict(
        id="M14", label="Steered frame: drawn, 1.75 out (200 stories)",
        run="r63-drawnmid", frag="g1p75orth", family="Steered frame", stories=200,
        summary="The point between M13 and M15, measured at 200 stories.",
        works="As M12 at a size of 1.75, run at 200 stories for a tighter interval.",
        different=(
            "Within one story of the baselines on coherence while breaking almost "
            "a whole requirement fewer."
        ),
        command=(
            "scripts/kaggle_run.sh r63-drawnmid --profile <account> --shards 2 -- \\\n"
            "  " + COMMON.format(stories=200) + " --basis-under-push \\\n"
            "  --peek-stories 6 --abort-broken-arms --suite boundary --tail-sweep 8 \\\n"
            "  --gamma-sweep 1.75 --offset-norm story --offset-gamma-spread 0.5 \\\n"
            "  --offset-draw-shape manifold \\\n"
            "  --steer-vectors present_tense sensory named_character no_heading \\\n"
            "  --shield-vectors in_story --steer-budget 2"
        ),
    ),
    dict(
        id="M15", label="Steered frame: drawn, 2.0 out", run="r61-drawnfar",
        frag="g2orth", family="Steered frame", stories=100,
        summary="Two stories out, measured against the steered cloud.",
        works="As M12 at a size of 2.0.",
        different=(
            "Beats top-k on variety of wording on an interval clear of zero, ties "
            "it on variety of what happens, and beats both baselines on compliance."
        ),
        command=(
            "scripts/kaggle_run.sh r61-drawnfar --profile <account> --shards 2 -- \\\n"
            "  " + COMMON.format(stories=100) + " --basis-under-push \\\n"
            "  --peek-stories 6 --abort-broken-arms --suite boundary --tail-sweep 8 \\\n"
            "  --gamma-sweep 2.0 2.5 3.0 --offset-norm story --offset-gamma-spread 0.5 \\\n"
            "  --offset-draw-shape manifold \\\n"
            "  --steer-vectors present_tense sensory named_character no_heading \\\n"
            "  --shield-vectors in_story --steer-budget 2"
        ),
    ),
    dict(
        id="M16", label="Steered frame, aimed at a story: 1.0 out",
        run="r58-pushbasis", frag="g1orth", family="Steered frame", stories=100,
        summary="M12's frame with M10's aiming.",
        works="The displacement aims at one of the stories sampled under the push.",
        different=(
            "Zero refusals and zero leaked plans, 98 stories of 100 coherent. The "
            "cost is variety of what happens: the push makes stories alike, which "
            "is its job, so the cloud it produces has less left to amplify."
        ),
        command=(
            "scripts/kaggle_run.sh r58-pushbasis --profile <account> --shards 2 -- \\\n"
            "  " + COMMON.format(stories=100) + " --basis-under-push \\\n"
            "  --peek-stories 6 --abort-broken-arms --suite boundary --tail-sweep 8 \\\n"
            "  --gamma-sweep 1.0 1.5 --offset-norm story --offset-gamma-spread 0.5 \\\n"
            "  --offset-draw-shape anchor \\\n"
            "  --steer-vectors present_tense sensory named_character no_heading \\\n"
            "  --shield-vectors in_story --steer-budget 2"
        ),
    ),
    dict(
        id="M17", label="Steered frame, aimed at a story: 1.25 out",
        run="r59-pushsweep", frag="g1p25orth", family="Steered frame", stories=100,
        summary="As M16 at one and a quarter stories out.",
        works="As M16 at a size of 1.25.",
        different="Variety of wording well ahead of both baselines at 95 of 100 coherent.",
        command=(
            "scripts/kaggle_run.sh r59-pushsweep --profile <account> --shards 2 -- \\\n"
            "  " + COMMON.format(stories=100) + " --basis-under-push \\\n"
            "  --peek-stories 6 --abort-broken-arms --suite boundary --tail-sweep 8 \\\n"
            "  --gamma-sweep 0.75 1.25 --offset-norm story --offset-gamma-spread 0.5 \\\n"
            "  --offset-draw-shape anchor \\\n"
            "  --steer-vectors present_tense sensory named_character no_heading \\\n"
            "  --shield-vectors in_story --steer-budget 2"
        ),
    ),
    dict(
        id="M18", label="Aimed at a real story: 1.5 out (200 stories)",
        run="r64-plain15", frag="g1p5orth", family="Aimed at a story", stories=200,
        summary=(
            "The only arm that beats top-k on BOTH variety measures on intervals "
            "clear of zero. Its weakness is coherence."
        ),
        works=(
            "M11 measured at 200 stories against both baselines at 200 stories. "
            "Aimed at one of the untouched model's own sampled stories, at one and "
            "a half stories' distance, with the register shield on and the size "
            "varying between stories."
        ),
        different=(
            "Beats top-k -- the stronger baseline on that axis -- on variety of "
            "what happens and on variety of wording, while breaking fewer "
            "requirements than either baseline. It keeps 176 stories of 200, which "
            "is the one axis it loses."
        ),
        command=(
            "scripts/kaggle_run.sh r64-plain15 --profile <account> --shards 2 -- \\\n"
            "  " + COMMON.format(stories=200) + " \\\n"
            "  --peek-stories 6 --abort-broken-arms --suite boundary --tail-sweep 8 \\\n"
            "  --gamma-sweep 1.5 --offset-norm story --offset-gamma-spread 0.5 \\\n"
            "  --offset-draw-shape anchor \\\n"
            "  --steer-vectors present_tense sensory named_character no_heading \\\n"
            "  --shield-vectors in_story --steer-budget 2"
        ),
    ),
]

ALL = BASELINES + METHODS

HEAD = PatternFill("solid", fgColor="1F3864")
HEADFONT = Font(color="FFFFFF", bold=True)
GOOD = PatternFill("solid", fgColor="C6EFCE")
WARN = PatternFill("solid", fgColor="FFEB9C")
BAD = PatternFill("solid", fgColor="FFC7CE")


def repool(S, ids, interval=INTERVAL_DRAWS):
    """Re-pool a subset among themselves.

    Every condition is pooled at the size of the smallest, so putting a
    hundred-story arm in the same table as a two-hundred-story one costs the
    larger one half its sample. The conditions run at 200 stories are therefore
    also compared among themselves, where the pool is twice as big and the
    intervals correspondingly tighter.
    """
    sub = {i: dict(S[i]) for i in ids}
    pool = min(v["kept"] for v in sub.values())
    take = max(int(pool * 0.7), 2)
    rng = np.random.default_rng(0)
    for v in sub.values():
        v["happens"] = rarefied_vendi(v["happens_vectors"], pool, draws=200)[0]
        v["wording"] = rarefied_vendi(v["wording_vectors"], pool, draws=200)[0]
        got = {"happens": [], "wording": []}
        for _ in range(interval):
            for key in ("happens", "wording"):
                vecs = v[f"{key}_vectors"]
                if len(vecs) < take:
                    continue
                idx = rng.choice(len(vecs), take, replace=False)
                got[key].append(rarefied_vendi(vecs[idx], take, draws=1,
                                               seed=int(rng.integers(1_000_000)))[0])
        v["draws"] = {k: np.array(x) for k, x in got.items()}
    return sub, pool, take


def score_all(entries, interval=INTERVAL_DRAWS):
    checker = EnglishConstraintChecker(
        backend="spacy", constraints=list(MIDDLE_CONSTRAINTS),
        max_opener_uses=EN_MIDDLE_MAX_OPENER_USES,
        max_word_uses=EN_MIDDLE_MAX_WORD_USES, min_grade_level=3.0,
        max_words=EN_MIDDLE_MAX_WORDS, max_adverbs=EN_MIDDLE_MAX_ADVERBS,
        min_sensory=EN_MIDDLE_MIN_SENSORY,
        present_ratio_threshold=EN_MIDDLE_PRESENT_RATIO,
    )
    coherence = CoherenceFilter()
    out = {}
    for e in entries:
        rid, texts = stories_from_run(e["run"], e["frag"])
        lead = [trim_lead(t)[1] > 0 for t in texts]
        titled = [opens_with_a_title(trim_lead(t)[0]) for t in texts]
        scored_text = [trim_title(trim_lead(t)[0])[0] for t in texts]
        reports = [coherence.check(t) for t in scored_text]
        kept = [r.text for r in reports if r.ok]
        kept_raw = [t for t, r in zip(texts, reports) if r.ok]
        agg = checker.evaluate_all(kept_raw)
        per = agg["stories"]
        e2 = dict(e)
        e2.update(
            rid=rid, texts=texts, reports=reports, n=len(texts), kept=len(kept),
            per_story=per,
            broken=agg["mean_violations"],
            broken_each=np.array([s.violations for s in per], float),
            grade=agg["mean_grade_level"],
            preambled=float(np.mean(lead)), titled=float(np.mean(titled)),
            usable=float(np.mean([r.ok for r in reports])),
            uncommon=float(np.mean(uncommon_word_share(kept))),
            opener=float(np.mean([opener_share(t) for t in kept])),
            wrong_open=float(np.mean([opens_in_the_wrong_tense(t, checker) for t in kept])),
            words=float(np.mean([len(t.split()) for t in kept])),
            words_per_sentence=float(
                np.mean([len(t.split()) for t in kept])
                / max(np.mean([len(SENTENCE.findall(t)) or 1 for t in kept]), 1e-9)),
            sentences=float(np.mean([len(SENTENCE.findall(t)) or 1 for t in kept])),
            rejected=Counter(r.reason for r in reports if not r.ok),
            happens_vectors=trim_isolated(_set_vectors(content_lemma_sets(kept, TRUNCATE))),
            wording_vectors=trim_isolated(pos_trigram_vectors(kept, TRUNCATE)),
        )
        out[e["id"]] = e2
        print(f"  scored {e['id']} {e['label'][:48]:<48} {len(kept)}/{len(texts)}")

    pool = min(v["kept"] for v in out.values())
    for v in out.values():
        v["happens"] = rarefied_vendi(v["happens_vectors"], pool, draws=200)[0]
        v["wording"] = rarefied_vendi(v["wording_vectors"], pool, draws=200)[0]

    take = max(int(pool * 0.7), 2)
    rng = np.random.default_rng(0)
    for v in out.values():
        got = {"happens": [], "wording": []}
        for _ in range(interval):
            for key in ("happens", "wording"):
                vecs = v[f"{key}_vectors"]
                if len(vecs) < take:
                    continue
                idx = rng.choice(len(vecs), take, replace=False)
                got[key].append(rarefied_vendi(vecs[idx], take, draws=1,
                                               seed=int(rng.integers(1_000_000)))[0])
        v["draws"] = {k: np.array(x) for k, x in got.items()}
    return out, pool, take


def diff(v, base, key):
    a, b = v["draws"][key], base["draws"][key]
    if a.size == 0 or b.size == 0:
        return None
    d = a - b
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def verdict(d):
    if d is None:
        return "--", ""
    m, lo, hi = d
    if lo > 0:
        return f"{m:+.1f} [{lo:+.1f}, {hi:+.1f}]", "win"
    if hi < 0:
        return f"{m:+.1f} [{lo:+.1f}, {hi:+.1f}]", "loss"
    return f"{m:+.1f} [{lo:+.1f}, {hi:+.1f}]", "tie"


def broken_diff(v, base):
    rng = np.random.default_rng(1)
    a, b = v["broken_each"], base["broken_each"]
    d = np.array([rng.choice(a, len(a), True).mean() - rng.choice(b, len(b), True).mean()
                  for _ in range(4000)])
    m, lo, hi = float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
    tag = "win" if hi < 0 else ("loss" if lo > 0 else "tie")
    return f"{m:+.2f} [{lo:+.2f}, {hi:+.2f}]", tag


def fit(ws, widths):
    for i, w in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(i)].width = w


def header(ws, row, cells):
    for i, c in enumerate(cells, start=1):
        cell = ws.cell(row=row, column=i, value=c)
        cell.fill, cell.font = HEAD, HEADFONT
        cell.alignment = Alignment(vertical="top", wrap_text=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="method_review.xlsx")
    ap.add_argument("--interval", type=int, default=INTERVAL_DRAWS)
    args = ap.parse_args()

    print("scoring every condition ...")
    S, pool, take = score_all(ALL, args.interval)
    nucleus, topk, untouched = S["B2"], S["B3"], S["B1"]

    import json
    prompt = json.load(open("/tmp/prompt.json")) if Path("/tmp/prompt.json").is_file() else None

    wb = Workbook()

    # ---------------- read me ----------------
    ws = wb.active
    ws.title = "Read me first"
    fit(ws, [30, 120])
    rows = [
        ("What this file is", (
            "Every configuration of our method worth keeping, what it does, how to "
            "run it, what it scores, and the stories themselves beside the "
            "baselines' so they can be read against each other.")),
        ("Model", "Qwen3-1.7B. The push and the displacement act on layers 6 to 13 of its 28."),
        ("Sampling temperature", "1.0 for every method row. The two baselines that raise it say so in their name."),
        ("Task", "One instruction, asking for a single ~150 word story for a middle-school reader, with 12 requirements."),
        ("Stories per condition", "100 or 200, stated per row. Every story index uses the same seed in every run."),
        ("Prompt family", (
            "Every run listed shares a byte-identical prompt. This was checked "
            "rather than assumed: the project has two prompt families and mixing "
            "them would make the comparison meaningless.")),
        ("", ""),
        ("THE EXACT PROMPT", "below, as the model received it"),
        ("System message", prompt["system"] if prompt else "(not captured)"),
        ("User message", prompt["user"] if prompt else "(not captured)"),
        ("", ""),
        ("How the push works", (
            "Four directions -- present tense, sensory detail, a named character, "
            "and story format -- are extracted by contrast: for each of twelve "
            "matched pairs the model is run over a shared prefix followed by a "
            "positive and a negative continuation, and the direction is the average "
            "difference between the two sides' hidden states. The four are made "
            "mutually orthogonal, summed to a fixed TOTAL strength of 2 (so adding "
            "a fifth direction does not push harder, it divides the same budget), "
            "and added to the residual stream at layers 6-13, at the prompt "
            "positions and at every decode step.")),
        ("How the displacement works", (
            "Separately, each story gets ONE perturbation, drawn once and held for "
            "the whole story. It is drawn from the subspace the model's own stories "
            "differ along -- 32 stories are sampled once per run and their "
            "activations summarised -- and projected clear of the constraint "
            "directions so it cannot undo the push. It is added at the prompt "
            "positions only, sparing the last eight, which are the chat template's "
            "own tokens.")),
        ("Why one per story, not per token", (
            "A diversity score measures how different the stories are FROM EACH "
            "OTHER. A mechanism that varies within a story and is the same in kind "
            "across stories cannot move it: every story varies, they all vary the "
            "same way, and they end up no further apart. That one fact accounts for "
            "three separate null results in this project.")),
        ("", ""),
        ("SIZE UNITS -- IMPORTANT", (
            "Rows M02 to M06 state the displacement size as a fraction of the "
            "hidden state's own length. Rows M07 onward state it as a multiple of "
            "how far one of the model's own stories sits from their average, which "
            "is the meaningful unit. On this model a real story sits 6.9 from the "
            "average at layer 6 and the old unit's 0.15 works out at 15.0 -- more "
            "than twice as far out. THE TWO UNITS ARE NOT COMPARABLE. Old 0.15 is "
            "roughly 1.5 to 2.2 in the new unit.")),
        ("", ""),
        ("How the numbers were made", (
            "Drop the stories the coherence checks reject; cut each story to its "
            "first 40 words; trim the ones sitting far from the rest of their own "
            f"set; pool every condition at the size of the smallest ({pool} here). "
            "Variety intervals are 95% ranges from "
            f"{args.interval} subsamples of {take} stories drawn WITHOUT "
            "replacement -- drawing with replacement duplicates stories, and "
            "duplicates read as identical, which pushes every score far below its "
            "true value.")),
        ("Reading an interval", (
            "A difference whose interval spans zero is a tie, however the point "
            "estimate reads. Half a point of variety on a hundred stories is noise; "
            "this project reported such a margin as a result once and had to "
            "withdraw it.")),
        ("Formatting failures", (
            "A heading, a preamble ('Certainly! Here's a short story:') or lost "
            "capitals count as ONE BROKEN REQUIREMENT, not as a broken story, "
            "because the prose underneath is fine. They are removed before any "
            "diversity number is computed, since a title is distinctive content "
            "sitting in exactly the opening words the score is computed over.")),
        ("Markdown mid-story", "Not penalised anywhere. It is a training artefact, not a defect."),
        ("", ""),
        ("WHERE THINGS STAND", (
            "Three of the four things the goal asks for are won. Requirement "
            "compliance beats both raised-temperature baselines by a wide margin "
            "almost everywhere. Coherence reaches 100 of 100 on several rows. "
            "Variety of wording beats top-k on several rows. Variety of what "
            "happens is won only by the rows that aim at one of the untouched "
            "model's own stories, and those are the rows that lose coherence.")),
        ("The one unsolved trade", (
            "Distance from the average buys variety and spends coherence, and the "
            "two clouds the displacement can be measured against are good at "
            "different things. The untouched model's stories carry the variation "
            "in what happens; the stories written under the push are where the "
            "model still reliably writes from. Nothing found so far has both.")),
        ("A lead worth following", (
            "The coherence loss is not spread evenly. The displacement aims at one "
            "of 32 sampled stories, chosen by story number, and across six "
            "independent runs EIGHT of those 32 account for 62 of the 76 rejected "
            "stories. Twenty-one to twenty-two of the 32 never cause a rejection "
            "at all. So a quarter of the sampled stories are doing nearly all the "
            "damage, and the same ones each time. Two ways of spotting them in "
            "advance have already failed: how far they lean away from telling a "
            "story, and how far they stick out of the steered cloud (correlation "
            "+0.09, essentially none). Finding what does identify them is the most "
            "direct route left to keeping the variety and the coherence at once.")),
    ]
    for r, (a, b) in enumerate(rows, start=1):
        ws.cell(row=r, column=1, value=a).font = Font(bold=True)
        ws.cell(row=r, column=1).alignment = Alignment(vertical="top", wrap_text=True)
        c = ws.cell(row=r, column=2, value=b)
        c.alignment = Alignment(vertical="top", wrap_text=True)

    # ---------------- metrics ----------------
    write_metrics(wb, "Methods and metrics", ALL, S, S["B2"], S["B3"])
    two_hundred = [e for e in ALL if e["stories"] == 200]
    sub, subpool, subtake = repool(S, [e["id"] for e in two_hundred], args.interval)
    write_metrics(wb, "200-story comparison", two_hundred, sub, sub["B2"], sub["B3"],
                  note=(f"Only the conditions run at 200 stories, pooled among "
                        f"themselves at {subpool} rather than at the {pool} the "
                        f"full table is forced to. Same numbers, twice the sample, "
                        f"tighter intervals. Read this sheet for these five rows."))

    # ---------------- descriptions ----------------
    write_descriptions(wb, S)
    write_rejections(wb, S)
    write_stories(wb, S)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    wb.save(args.out)
    print(f"\nwrote {args.out}  ({len(ALL)} conditions, pooled at {pool}; "
          f"the 200-story sheet pools at {subpool})")


def write_metrics(wb, title, entries, S, nucleus, topk, note=None):
    ws = wb.create_sheet(title)
    start = 1
    if note:
        c = ws.cell(row=1, column=1, value=note)
        c.font = Font(bold=True, italic=True)
        c.alignment = Alignment(vertical="top", wrap_text=True)
        ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=12)
        ws.row_dimensions[1].height = 46
        start = 2
    cols = ["ID", "Method", "Family", "Stories", "Coherent", "Coherent %",
            "Requirements broken (of 12)", "vs nucleus", "vs top-k",
            "Variety: what happens", "vs nucleus", "vs top-k",
            "Variety: wording", "vs nucleus", "vs top-k",
            "Reading grade", "Words per sentence", "Words per story",
            "Sentences per story", "Preamble %", "Heading %",
            "Uncommon words %", "Commonest opener %",
            "Run folder", "Run id"]
    header(ws, start, cols)
    fit(ws, [6, 46, 22, 8, 10, 10, 13, 20, 20, 12, 20, 20, 12, 20, 20,
             9, 11, 11, 11, 9, 9, 10, 11, 16, 60])
    for r, e in enumerate(entries, start=start + 1):
        v = S[e["id"]]
        bn, bn_tag = broken_diff(v, nucleus)
        bk, bk_tag = broken_diff(v, topk)
        hn, hn_tag = verdict(diff(v, nucleus, "happens"))
        hk, hk_tag = verdict(diff(v, topk, "happens"))
        wn, wn_tag = verdict(diff(v, nucleus, "wording"))
        wk, wk_tag = verdict(diff(v, topk, "wording"))
        vals = [e["id"], e["label"], e["family"], v["n"],
                f"{v['kept']}/{v['n']}", v["kept"] / v["n"],
                round(v["broken"], 2), bn, bk,
                round(v["happens"], 1), hn, hk,
                round(v["wording"], 1), wn, wk,
                round(v["grade"], 2), round(v["words_per_sentence"], 1),
                round(v["words"], 0), round(v["sentences"], 1),
                v["preambled"], v["titled"], v["uncommon"], v["opener"],
                v["run"], v["rid"]]
        for i, val in enumerate(vals, start=1):
            c = ws.cell(row=r, column=i, value=val)
            c.alignment = Alignment(vertical="top", wrap_text=(i in (2, 25)))
        ws.cell(row=r, column=6).number_format = "0%"
        for i in (20, 21, 22, 23):
            ws.cell(row=r, column=i).number_format = "0%"
        for col, tag in ((8, bn_tag), (9, bk_tag), (11, hn_tag), (12, hk_tag),
                         (14, wn_tag), (15, wk_tag)):
            ws.cell(row=r, column=col).fill = {
                "win": GOOD, "loss": BAD, "tie": WARN}.get(tag, WARN)
        pct = v["kept"] / v["n"]
        ws.cell(row=r, column=6).fill = GOOD if pct >= 0.99 else (
            WARN if pct >= 0.95 else BAD)
    ws.freeze_panes = ws.cell(row=start + 1, column=3).coordinate


def write_descriptions(wb, S):
    ws = wb.create_sheet("How each method works")
    header(ws, 1, ["ID", "Method", "In one line", "How it works",
                   "What is different from the baselines",
                   "Command that produced it", "Run folder", "Run id"])
    fit(ws, [6, 44, 52, 90, 80, 96, 16, 60])
    for r, e in enumerate(ALL, start=2):
        v = S[e["id"]]
        for i, val in enumerate([e["id"], e["label"], e["summary"], e["works"],
                                 e["different"], e["command"], e["run"], v["rid"]],
                                start=1):
            c = ws.cell(row=r, column=i, value=val)
            c.alignment = Alignment(vertical="top", wrap_text=True)

def write_rejections(wb, S):
    ws = wb.create_sheet("Why stories were rejected")
    header(ws, 1, ["ID", "Method", "Coherent", "Rejected", "Reasons"])
    fit(ws, [6, 46, 12, 10, 90])
    for r, e in enumerate(ALL, start=2):
        v = S[e["id"]]
        reasons = ", ".join(f"{k} {n}" for k, n in v["rejected"].most_common())
        for i, val in enumerate([e["id"], e["label"], f"{v['kept']}/{v['n']}",
                                 v["n"] - v["kept"], reasons or "none"], start=1):
            c = ws.cell(row=r, column=i, value=val)
            c.alignment = Alignment(vertical="top", wrap_text=(i == 5))

def write_stories(wb, S):
    ws = wb.create_sheet("Stories side by side")
    cols = ["Story #"] + [f"{e['id']} {e['label']}" for e in ALL]
    header(ws, 1, cols)
    fit(ws, [8] + [70] * len(ALL))
    n_rows = max(v["n"] for v in S.values())
    for i in range(n_rows):
        ws.cell(row=i + 2, column=1, value=i).alignment = Alignment(vertical="top")
        for j, e in enumerate(ALL, start=2):
            v = S[e["id"]]
            txt = v["texts"][i] if i < len(v["texts"]) else ""
            c = ws.cell(row=i + 2, column=j, value=txt[:32000])
            c.alignment = Alignment(vertical="top", wrap_text=True)
    ws.freeze_panes = "B2"

    # ---------------- per-story verdicts ----------------
    ws = wb.create_sheet("Story verdicts")
    header(ws, 1, cols)
    fit(ws, [8] + [30] * len(ALL))
    for i in range(n_rows):
        ws.cell(row=i + 2, column=1, value=i)
        for j, e in enumerate(ALL, start=2):
            v = S[e["id"]]
            if i >= len(v["texts"]):
                continue
            rep = v["reports"][i]
            note = "coherent" if rep.ok else f"REJECTED: {rep.reason}"
            extra = []
            if trim_lead(v["texts"][i])[1] > 0:
                extra.append("preamble")
            if opens_with_a_title(trim_lead(v["texts"][i])[0]):
                extra.append("heading")
            c = ws.cell(row=i + 2, column=j,
                        value=note + ((" | " + ", ".join(extra)) if extra else ""))
            c.alignment = Alignment(vertical="top", wrap_text=True)
            if not rep.ok:
                c.fill = BAD
            elif extra:
                c.fill = WARN
    ws.freeze_panes = "B2"


if __name__ == "__main__":
    main()
