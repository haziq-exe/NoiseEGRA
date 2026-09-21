#!/usr/bin/env bash
# The two runs the measurements point at, ready to launch.
#
# Both were written when Kaggle stopped starting sessions on all three
# accounts, so that neither has to be derived again. Run them one at a time,
# one account each, and check `kaggle_harness.py sessions` afterwards.
#
# WHERE THE METHOD STANDS. At a total push of 2.5, a noise colour of 2 and a
# displacement of 1.5 story-distances, the method beats nucleus sampling on both
# variety measures and on requirements broken, at 194 of 200 coherent and
# temperature 1.0. Three things are measured about the 6 that fail:
#
#   * the untouched model at temperature 1.0 is itself 194 of 200, and every one
#     of its failures is a repetition loop;
#   * the push ALONE is 200 of 200 -- it cures those loops;
#   * the displacement at 1.5 returns coherence to exactly the base rate, with
#     all six failures loops again.
#
# So the push suppresses the base model's loops and the displacement re-admits
# them. The obvious inference -- push harder -- is WRONG, and the runs already
# on disk say so. At a matched displacement, coherence against total push:
#
#     gamma 1.50:  2.0 -> 145/200   2.5 -> 183-199   3.0 -> 173/200
#     gamma 1.75:  2.0 -> 162,177   2.5 -> 178-191   3.0 -> 164/200
#     gamma 2.00:  2.0 -> 165/200   2.5 -> 170-193   3.0 -> 176   3.5 -> 109/200
#
# 2.5 is already near the top and 3.5 collapses. PAPER_RESULTS said as much
# before this script was written -- "raising the total from 2 to 4 is far worse
# than leaving it at 2" -- and a budget-raising run was queued here anyway. It
# has been removed rather than left to be discovered by burning a run.
#
# And the budget split is measured on the UNTOUCHED model, while the
# displacement breaks requirements the untouched model does not: re-measured on
# the method's own 200 stories, naming a character falls from 66% to 42% and
# plain words rises to 100%. A story that loops opened in the first person 67%
# of the time against 14% of the kept ones, and named somebody 67% against 98%
# -- the unnamed first-person register is what fails to end. That is run two.
set -euo pipefail
cd "$(dirname "$0")/.."

COMMON=(--model Qwen3-1.7B --task generic --constraint-set middle --stories 200
        --layers 6 14 --temperature 1.0 --story-target 150 --truncate-words 40
        --pairs children --peek-stories 4 --abort-broken-arms
        --offset-basis story --basis-under-push --offset-basis-tokens 400
        --suite colourfront --tail-sweep 8 --offset-norm story
        --offset-gamma-spread 0.5 --offset-draw-shape manifold
        --shield-vectors in_story --noise-beta-sweep 2.0
        --steer-vectors present_tense mature_register dialogue varied_openers
                        plain_words sensory fresh_words named_character
                        no_repetition distinct_sentences fresh_openings
                        no_heading something_happens)

case "${1:-}" in
  push)
    echo "Removed. Raising the total push is measured to make coherence worse:" >&2
    echo "  gamma 1.5:  2.5 -> 183-199 coherent, 3.0 -> 173/200" >&2
    echo "  gamma 2.0:  2.5 -> 170-193 coherent, 3.5 -> 109/200" >&2
    echo "Use 'round2', which moves budget between directions at the same total." >&2
    exit 2
    ;;
  round2)
    # Run two: the budget re-measured on the method's own output. The shares
    # below are that measurement, not a guess -- see the table in
    # PAPER_RESULTS.md. Naming rises from 0.34 to 0.65 because the displacement
    # is what breaks it; plain words falls to nothing because the method already
    # satisfies it in every story.
    exec scripts/kaggle_run.sh r108-round2 --profile "${2:-coauth2}" --shards 2 -- \
      "${COMMON[@]}" --steer-budget 2.5 --gamma-sweep 1.5 1.75 \
      --steer-weights present_tense=1.0 varied_openers=0.73 named_character=0.65 \
                      sensory=0.61 dialogue=0.53 mature_register=0.44 \
                      something_happens=0.38 fresh_words=0.28 fresh_openings=0.17 \
                      distinct_sentences=0.07 no_repetition=0.03 no_heading=0.02 \
                      plain_words=0.0
    ;;
  closure)
    # The one intervention aimed at the coherence gap that is inside the method
    # rather than a decoding guard. The failures are stories that do not end --
    # 292 words against 206 for the ones kept, repetition starting at word 220
    # of 291 -- and "closure" is a direction meaning bring it to an end, on a
    # schedule quiet while a story is inside its budget and pressing harder the
    # longer it runs over. Measured on the children's task it took looping from
    # 48% of stories to 15% and improved compliance at the same time. It has
    # never been run on this task, and it is not in the steered set.
    #
    # The horizon is where a legal story ends: roughly 150 words, about 200
    # tokens. Two arms, with and without, so the brake is read against itself.
    exec scripts/kaggle_run.sh r109-closure --profile "${2:-coauth1}" --shards 2 -- \
      "${COMMON[@]}" --steer-allocate shortfall --steer-budget 2.5 \
      --gamma-sweep 1.5 --noise-beta-sweep 2.0 \
      --steer-vectors present_tense mature_register dialogue varied_openers \
                      plain_words sensory fresh_words named_character \
                      no_repetition distinct_sentences fresh_openings \
                      no_heading something_happens closure \
      --closure-ramp 200
    ;;
  *)
    echo "usage: $0 {round2|closure} [kaggle-profile]" >&2
    echo >&2
    echo "  closure a brake on stories that do not end - the coherence lever" >&2
    echo "  round2  the budget re-measured on the method's own output" >&2
    exit 2
    ;;
esac
