#!/usr/bin/env python
"""Print an access token for one of the project's Kaggle accounts.

    python scripts/kaggle_token.py                 # the default account
    python scripts/kaggle_token.py --profile alice # ~/.kaggle-alice/credentials.json

The Kaggle client hard-codes its OAuth credentials to ~/.kaggle/credentials.json
-- KAGGLE_CONFIG_DIR only redirects the older kaggle.json -- so `kaggle auth
login` cannot be pointed at a second account, and a project with more than one
person's account has nowhere to put the second set. This reads a credentials file
from wherever it is told and mints a token from its refresh token, so several
accounts can live side by side and a run can name the one it wants.

Minting is itself a rate-limited call, so the token is cached under /tmp and
reused until it is close to expiring. Asking for one per API call earns a 429
within minutes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

CACHE_DIR = Path("/tmp")
REFRESH_BEFORE = timedelta(minutes=30)


def credentials_path(profile: str | None) -> Path:
    if not profile or profile == "default":
        return Path.home() / ".kaggle" / "credentials.json"
    return Path.home() / f".kaggle-{profile}" / "credentials.json"


def cached(profile: str) -> str | None:
    p = CACHE_DIR / f".kaggle-token-{profile}.json"
    if not p.is_file():
        return None
    try:
        blob = json.loads(p.read_text())
        exp = datetime.fromisoformat(blob["expires"])
    except Exception:
        return None
    if exp - datetime.now(timezone.utc) < REFRESH_BEFORE:
        return None
    return blob.get("token")


def store(profile: str, token: str, expires: datetime) -> None:
    p = CACHE_DIR / f".kaggle-token-{profile}.json"
    p.write_text(json.dumps({"token": token, "expires": expires.isoformat()}))
    try:
        p.chmod(0o600)
    except OSError:
        pass


def mint(path: Path) -> tuple[str, datetime]:
    from kagglesdk import KaggleClient
    from kagglesdk.kaggle_creds import KaggleCredentials

    client = KaggleClient()
    creds = KaggleCredentials.load(client, str(path))
    if creds is None:
        raise SystemExit(f"no usable credentials in {path}")
    client._credentials = creds
    creds._client = client
    creds.refresh_access_token()
    token = creds._access_token
    expires = creds._access_token_expiration or (
        datetime.now(timezone.utc) + timedelta(hours=1))
    if expires.tzinfo is None:
        expires = expires.replace(tzinfo=timezone.utc)
    # Write the refreshed token back, so the account's own file stays current and
    # the next process does not have to mint again.
    try:
        creds.save(str(path))
    except Exception:
        pass
    return token, expires


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="default")
    ap.add_argument("--whoami", action="store_true",
                    help="print the account name instead of the token")
    args = ap.parse_args()

    path = credentials_path(args.profile)
    if not path.is_file():
        raise SystemExit(
            f"no credentials at {path}\n"
            f"  to add this account: scripts/kaggle_add_account.sh {args.profile}")

    if args.whoami:
        print(json.loads(path.read_text()).get("username", "?"))
        return

    token = cached(args.profile)
    if token is None:
        token, expires = mint(path)
        store(args.profile, token, expires)
    print(token)


if __name__ == "__main__":
    main()
