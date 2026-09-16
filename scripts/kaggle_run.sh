#!/usr/bin/env bash
# Launch a Kaggle run, stream its output to a local file while it runs, and pull
# the results when it finishes.
#
#   scripts/kaggle_run.sh <name> [--profile NAME] -- <runner args...>
#
# Three things this handles that a bare `kaggle_harness.py run` does not.
#
# The access token expires every few hours and the client does not persist the
# refreshed one, so every call mints a fresh token instead of exporting one and
# hoping it outlives the run. A run once died mid-sweep on a 401 for exactly this.
#
# The log stream drops (Kaggle's load balancer closes idle connections, and the
# replay-from-the-start reconnect gets slower as the log grows), so the follower
# runs in a loop and reconnects. The output lands in a file that can be tailed at
# any time, so results are readable as they are produced rather than only at the
# end.
#
# --profile selects which account to run on, for a project with more than one
# person's Kaggle account: credentials live in ~/.kaggle/credentials-<name>.json
# and that file is used instead of the default.
set -uo pipefail
cd "$(dirname "$0")/.."

NAME="${1:?usage: kaggle_run.sh <name> [--profile NAME] [--shards N] -- <runner args...>}"; shift
PROFILE=""
SHARDS=2
while :; do
  case "${1:-}" in
    --profile) PROFILE="$2"; shift 2 ;;
    --shards)  SHARDS="$2";  shift 2 ;;
    --) shift; break ;;
    *) break ;;
  esac
done
ARGS="$*"

VENV="$HOME/.cache/noiseegra-harness/venv/bin/python"
# Which account to run on. The Kaggle client hard-codes its OAuth credentials to
# ~/.kaggle/credentials.json, so a second account cannot be selected by an
# environment variable; kaggle_token.py reads whichever credentials file it is
# told to and mints from that, caching the token because minting is itself
# rate-limited.
if ! "$VENV" scripts/kaggle_token.py --profile "${PROFILE:-default}" --whoami >/dev/null 2>&1; then
  echo "no credentials for profile '${PROFILE:-default}'"
  echo "  add an account with:  scripts/kaggle_add_account.sh <name>"
  exit 1
fi
ACCOUNT=$("$VENV" scripts/kaggle_token.py --profile "${PROFILE:-default}" --whoami)
echo "running as $ACCOUNT (profile ${PROFILE:-default})"
tok() { "$VENV" scripts/kaggle_token.py --profile "${PROFILE:-default}"; }
kh()  { KAGGLE_API_TOKEN="$(tok)" python scripts/kaggle_harness.py "$@"; }

LOG="/tmp/${NAME}.log"
LIVE="/tmp/${NAME}.live"
: > "$LIVE"

# Reject a bad flag here rather than after a session start and a model download.
python scripts/run_english_experiment.py $ARGS --dry-run || { echo "ARGS REJECTED"; exit 1; }

if [ "$SHARDS" -gt 1 ]; then
  RUNCMD="scripts/run_sharded.py {OUT} $SHARDS -- $ARGS"
else
  RUNCMD="scripts/run_english_experiment.py $ARGS"
fi
# If the harness refuses -- an uncommitted working tree, a bad flag, no quota --
# stop here. Carrying on polls a kernel that was never created, which looks
# exactly like a run in progress and wasted forty minutes once.
if ! kh run --name "$NAME" --max-minutes 420 --no-wait -- $RUNCMD 2>&1 | tee -a "$LOG"; then
  echo "launch refused; not watching" | tee -a "$LOG"; exit 1
fi
if ! grep -q "pushed\. Watch it with" "$LOG"; then
  echo "launch did not report a pushed kernel; not watching" | tee -a "$LOG"; exit 1
fi

( while true; do
    kh follow --name "$NAME" --timeout 600 >> "$LIVE" 2>/dev/null
    S=$(kh status --name "$NAME" 2>&1 | tail -1)
    case "$S" in *complete*|*error*|*cancel*) break ;; esac
    sleep 5
  done ) &
FOLLOW=$!

while true; do
  S=$(kh status --name "$NAME" 2>&1 | tail -1)
  echo "$(date +%H:%M:%S) $S" | tee -a "$LOG"
  case "$S" in *complete*|*error*|*cancel*) break ;; esac
  sleep 180
done
kill $FOLLOW 2>/dev/null
echo "=== pulling results ===" | tee -a "$LOG"
kh pull --name "$NAME" 2>&1 | tail -20 | tee -a "$LOG"
