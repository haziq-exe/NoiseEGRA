#!/usr/bin/env bash
# The two experiments waiting on a GPU, with the arguments that were validated
# against --dry-run so a rented card is never spent on a rejected flag.
#
#   scripts/lightning_next_runs.sh wholefive <name>
#   scripts/lightning_next_runs.sh wholeloop <name>
#
# 'wholefive' is the five-way comparison over the whole-story rules: untouched,
# nucleus at high temperature, ours, steering alone, noise alone.
# 'wholeloop' is the calibration-free steering, where each rule's strength rises
# only once the story is seen to be breaking it.
#
# One shard, unlike the Kaggle and Modal versions, because the free Lightning
# plan gives a single GPU. Two shards there would take turns on the one card and
# each pay the setup, which is slower than not splitting at all.
set -euo pipefail
cd "$(dirname "$0")/.."

WHOLEFIVE="--model Qwen3-1.7B --task generic --constraint-set whole --suite wholefive --stories 25 --layers 6 14 --temperature 1.0 --story-target 150 --truncate-words 40 --pairs children --peek-stories 3 --offset-basis story --basis-under-push --offset-basis-stories 16 --offset-basis-tokens 260 --tail-sweep 8 --offset-norm story --offset-gamma-spread 0.5 --offset-draw-shape manifold --shield-vectors in_story --steer-budget 2.5 --gamma-sweep 1.5 --noise-beta-sweep 2.0 --baseline-temperature 1.8"
WHOLELOOP="--model Qwen3-1.7B --task generic --constraint-set whole --suite wholeloop --stories 25 --layers 6 14 --temperature 1.0 --story-target 150 --truncate-words 40 --pairs children --peek-stories 3 --offset-basis story --basis-under-push --offset-basis-stories 16 --offset-basis-tokens 260 --tail-sweep 8 --offset-norm story --offset-gamma-spread 0.5 --offset-draw-shape manifold --shield-vectors in_story --steer-budget 2.5 --gamma-sweep 1.5 --noise-beta-sweep 2.0 --feedback-betas 1.0 2.0 --steer-vectors present_tense mature_register dialogue named_character both_genders simile distinct_sentences no_heading closure"

SUITE="${1:?usage: lightning_next_runs.sh <wholefive|wholeloop> <name>}"
NAME="${2:?usage: lightning_next_runs.sh <wholefive|wholeloop> <name>}"

case "$SUITE" in
  wholefive) ARGS="$WHOLEFIVE" ;;
  wholeloop) ARGS="$WHOLELOOP" ;;
  *) echo "unknown suite '$SUITE'"; exit 1 ;;
esac

python scripts/lightning_run.py --name "$NAME" --shards 1 --runner-args "$ARGS"
