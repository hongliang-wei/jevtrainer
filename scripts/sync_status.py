"""Is every checkout (this machine + the servers) clean and pushed?  Exit code 1 when something needs attention.

    python scripts/sync_status.py                       # servers from $JT_SERVERS
    python scripts/sync_status.py --ssh s1=41848 --ssh s2=43683
    JT_SERVERS="s1=41848,s2=43683" python scripts/sync_status.py

Per checkout it prints the branch, commits ahead / behind the remote branch, and uncommitted code files
(logs, runs/, scratch dirs and caches are ignored). Servers should only ever `git pull`; anything listed
as dirty there was edited on the server and has to be carried back, committed and pushed.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

HOST = os.environ.get("JT_SERVER_HOST", "connect.westd.seetacloud.com")
REMOTE_REPO = os.environ.get("JT_SERVER_REPO", "/root/jevtrainer")
IGNORE = ("runs", "_scratch", ".log", "prepare_", "__pycache__", ".pytest_cache")

SCRIPT = r"""
git fetch -q origin 2>/dev/null
b=$(git branch --show-current)
echo "branch=$b"
if git rev-parse -q --verify "origin/$b" >/dev/null; then
  set -- $(git rev-list --left-right --count "HEAD...origin/$b")
  echo "ahead=$1 behind=$2"
else
  echo "ahead=? behind=? (no origin/$b)"
fi
git status --porcelain
"""


def run(cmd: list[str], cwd: str | None = None, stdin: str | None = None) -> str:
    data = stdin.replace("\r\n", "\n").encode() if stdin is not None else None  # bytes: no CRLF on Windows
    p = subprocess.run(cmd, cwd=cwd, input=data, capture_output=True)
    return p.stdout.decode("utf-8", errors="replace")

def parse(label: str, out: str) -> bool:
    lines = [l for l in out.splitlines() if l.strip()]
    meta = dict(l.split("=", 1) for l in lines if l.startswith(("branch=", "ahead=")))
    dirty = [l for l in lines if not l.startswith(("branch=", "ahead=")) and not any(i in l for i in IGNORE)]
    counts = meta.get("ahead", "? ?").replace("behind=", "").split()
    ahead, behind = counts[0], counts[1] if len(counts) > 1 else "?"
    ok = not dirty and ahead == "0"
    print(f"{'OK  ' if ok else 'FIX '} {label:8s} branch={meta.get('branch', '?'):10s} ahead={ahead} behind={behind}")
    for l in dirty[:15]:
        print(f"       uncommitted: {l}")
    if len(dirty) > 15:
        print(f"       ... and {len(dirty) - 15} more")
    if ahead not in ("0", "?"):
        print(f"       {ahead} commit(s) not pushed from {label}")
    return ok

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ssh", action="append", default=[], help="label=port of a server")
    a = ap.parse_args()
    servers = a.ssh or [s for s in os.environ.get("JT_SERVERS", "").split(",") if s]
    root = str(Path(__file__).resolve().parents[1])

    ok = parse("local", local_checkout(root))
    for s in servers:
        label, port = s.split("=")
        out = run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-p", port, f"root@{HOST}", f"cd {REMOTE_REPO} && bash -s"], stdin=SCRIPT)
        if not out.strip():
            print(f"FIX  {label:8s} unreachable")
            ok = False
            continue
        ok = parse(label, out) and ok
    sys.exit(0 if ok else 1)


def local_checkout(root: str) -> str:
    run(["git", "fetch", "-q", "origin"], cwd=root)
    b = run(["git", "branch", "--show-current"], cwd=root).strip()
    counts = run(["git", "rev-list", "--left-right", "--count", f"HEAD...origin/{b}"], cwd=root).split()
    head = f"branch={b}\nahead={counts[0] if counts else '?'} behind={counts[1] if len(counts) > 1 else '?'}\n"
    return head + run(["git", "status", "--porcelain"], cwd=root)


if __name__ == "__main__":
    main()
