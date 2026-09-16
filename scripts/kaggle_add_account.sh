#!/usr/bin/env bash
# Add a second Kaggle account to this project.
#
#   scripts/kaggle_add_account.sh coauth1
#
# The Kaggle client writes its OAuth credentials to ~/.kaggle/credentials.json
# and nowhere else -- KAGGLE_CONFIG_DIR only redirects the older kaggle.json --
# so `kaggle auth login` always overwrites whoever is already logged in, and
# without --force it refuses and reports the existing account. This moves the
# current credentials out of the way, runs the login for the new person, files
# the result under ~/.kaggle-<name>/, and puts the original back.
#
# The browser opens for the account being added. Nobody's password is handled
# here; it is the ordinary Kaggle OAuth flow.
set -euo pipefail
cd "$(dirname "$0")/.."

NAME="${1:?usage: kaggle_add_account.sh <profile-name>}"
VENV="$HOME/.cache/noiseegra-harness/venv/bin/python"
MAIN="$HOME/.kaggle/credentials.json"
DEST="$HOME/.kaggle-$NAME/credentials.json"
BACKUP="$HOME/.kaggle/credentials.json.saved-$$"

mkdir -p "$HOME/.kaggle" "$HOME/.kaggle-$NAME"
restore() {
  [ -f "$BACKUP" ] && mv -f "$BACKUP" "$MAIN"
}
trap restore EXIT

[ -f "$MAIN" ] && cp -p "$MAIN" "$BACKUP" && rm -f "$MAIN"

echo "A browser window will open. Sign in as the account you want to add as '$NAME'."
echo "If it shows an account you did not expect, sign out there first."
echo
"$VENV" -m kaggle auth login --force

if [ ! -f "$MAIN" ]; then
  echo "login did not write credentials; nothing was added"
  exit 1
fi
WHO=$(python -c "import json,sys;print(json.load(open('$MAIN')).get('username','?'))")
mv -f "$MAIN" "$DEST"
chmod 600 "$DEST"
echo
echo "added '$NAME' = $WHO  ->  $DEST"
echo "run on it with:  scripts/kaggle_run.sh <run-name> --profile $NAME -- <args>"
