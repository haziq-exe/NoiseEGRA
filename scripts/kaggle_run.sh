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

NAME="${1:?usage: kaggle_run.sh <name> [--profile NAME] -- <runner args...>}"; shift
PROFILE=""
if [ "${1:-}" = "--profile" ]; then PROFILE="$2"; shift 2; fi
[ "${1:-}" = "--" ] && shift
ARGS="$*"

VENV="$HOME/.cache/noiseegra-harness/venv/bin/python"
# The Kaggle client reads credentials.json out of KAGGLE_CONFIG_DIR, so a second
# account is a second config directory rather than a second machine.
CFG="$HOME/.kaggle"
[ -n "$PROFILE" ] && CFG="$HOME/.kaggle-$PROFILE"
if [ ! -f "$CFG/credentials.json" ]; then
  echo "no credentials at $CFG/credentials.json"
  if [ -n "$PROFILE" ]; then
    echo "  to add this account, with that person logged in on this machine:"
    echo "    mkdir -p $CFG"
    echo "    KAGGLE_CONFIG_DIR=$CFG $VENV -m kaggle auth login"
  fi
  exit 1
fi
export KAGGLE_CONFIG_DIR="$CFG"

# Minting a token is itself a rate-limited API call, so it is cached and refreshed
# on a timer rather than taken fresh for every command. Minting per call earns a
# 429 within a few minutes of polling.
TOKCACHE="/tmp/.kaggle-token-${PROFILE:-default}"
TOKAGE=2400          # 40 minutes; the token itself lasts hours
tok() {
  if [ ! -s "$TOKCACHE" ] || [ $(( $(date +%s) - $(stat -f %m "$TOKCACHE" 2>/dev/null || echo 0) )) -gt $TOKAGE ]; then
    local t
    t="$("$VENV" -m kaggle auth print-access-token 2>/dev/null | tr -d '\n')"
    [ -n "$t" ] && printf '%s' "$t" > "$TOKCACHE"
  fi
  cat "$TOKCACHE" 2>/dev/null
}
kh()  { KAGGLE_API_TOKEN="$(tok)" python scripts/kaggle_harness.py "$@"; }

LOG="/tmp/${NAME}.log"
LIVE="/tmp/${NAME}.live"
: > "$LIVE"

# Reject a bad flag here rather than after a session start and a model download.
python scripts/run_english_experiment.py $ARGS --dry-run || { echo "ARGS REJECTED"; exit 1; }

kh run --name "$NAME" --max-minutes 420 --no-wait -- \
   scripts/run_english_experiment.py $ARGS | tee -a "$LOG"

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
