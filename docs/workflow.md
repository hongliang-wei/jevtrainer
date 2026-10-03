# Working on several machines

One development machine (the laptop), any number of run machines (GPU servers). Code is written in one place only.

## Rules

1. **Servers only pull.** Do not write code on a server. A quick fix there is carried back with `git diff`,
   committed on the development machine and pushed; then `git checkout -- .` on the server.
2. **Commit small, push at once.** An unpushed commit exists on one disk only. Experiments go on a branch.
3. **One branch per line of work, merged with the trunk often.** `main` = stable (text / Jev). `av-jev` = audio-video.
   Merge `main` into the branch every day or two; merge the branch back when it is stable.
4. **Never two writers on the same thing.** Before adding a dataset converter run
   `grep -rn 'DatasetSpec("<name>"' jevtrainer` - if it exists, extend it. Tests fail on duplicates.
5. **Generated output stays out of git**: logs, `runs/`, caches, scratch. Keep the script that produces them.
6. **Servers use a read-only deploy key**, so nothing is pushed from them. Anything done there that must be kept
   (a config, a patch) is copied back by `scp` / `git fetch ssh://...` and committed from the laptop.
7. **Checkpoints and data never go on the system disk** of a server (30 GB, it filled once and killed a run):
   `runs/` is a symlink to the data disk.

## Tools

```
git config core.hooksPath .githooks            # once per checkout: pre-push blocks duplicate registrations
export JT_SERVERS="s1=41848,s2=43683"          # label=ssh port
python scripts/sync_status.py                  # is every checkout clean and pushed?  exit 1 if not
python -m pytest -q tests/test_registry_unique.py
```

`sync_status.py` lists, per checkout, branch, commits ahead / behind, and uncommitted code files. Run it before
ending a session and after anything was done on a server. `.gitattributes` fixes line endings (LF) so files copied
from Windows are not reported as fully modified.

## Updating a server

```
cd /root/jevtrainer && git fetch && git status -sb      # must be clean
git pull --ff-only
```

A running training has already imported its code; pulling does not change it, but a restart or `resume` will use the
new code.
