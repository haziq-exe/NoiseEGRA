#!/usr/bin/env bash
# Run one suite across both of the kernel's GPUs.
#
#   scripts/run_sharded.sh <out-dir> <shards> -- <runner args...>
#
# A Kaggle kernel is given two T4s and a single model uses one of them, so half
# the hardware has been idle for every run so far. Qwen3-1.7B is about 3.4 GB in
# float16, so a second copy fits on the other card comfortably.
#
# Two processes rather than two threads: the generation hooks keep per-call state
# on the model object, so two threads sharing one model would interleave their
# decode-step counters. Separate processes share nothing.
#
# Each shard writes its own output directory and extracts its own steering
# vectors, which duplicates a couple of minutes of setup but costs no wall clock
# because it happens on both cards at once.
set -uo pipefail
OUT="${1:?usage: run_sharded.sh <out-dir> <shards> -- <args>}"; shift
N="${1:?number of shards}"; shift
[ "${1:-}" = "--" ] && shift

pids=()
for ((i=0; i<N; i++)); do
  CUDA_VISIBLE_DEVICES="$i" python scripts/run_english_experiment.py "$@" \
      --shard "$i/$N" --out "$OUT/shard$i" 2>&1 | sed "s/^/[gpu$i] /" &
  pids+=($!)
  sleep 20      # stagger the model loads so they do not contend for the disk
done

fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
echo "=== all $N shards finished (exit $fail) ==="
exit $fail
