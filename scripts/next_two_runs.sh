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
# them. Every arm ever run here uses a push of 2.5. That is run one.
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
    # Run one: a stronger push at the same displacement. If the push is what
    # holds the loops off, this is where coherence comes back without giving up
    # the displacement that carries the variety.
    exec scripts/kaggle_run.sh r107-push --profile "${2:-coauth1}" --shards 2 -- \
      "${COMMON[@]}" --steer-allocate shortfall \
      --steer-budget 3.5 --gamma-sweep 1.5 1.75
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
  *)
    echo "usage: $0 {push|round2} [kaggle-profile]" >&2
    echo >&2
    echo "  push    a stronger total push at the same displacement" >&2
    echo "  round2  the budget re-measured on the method's own output" >&2
    exit 2
    ;;
esac
