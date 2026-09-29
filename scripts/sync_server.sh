#!/usr/bin/env bash
# Push local commits, then fast-forward the GPU server's clone and refresh the editable install.
#   JT_SERVER="-p 41848 root@connect.westd.seetacloud.com" scripts/sync_server.sh
set -euo pipefail
SERVER=${JT_SERVER:?set JT_SERVER, e.g. "-p 41848 root@host"}
REMOTE_DIR=${JT_REMOTE_DIR:-/root/jevtrainer}
git push
ssh $SERVER "set -e; source /root/autodl-tmp/env.sh 2>/dev/null || true; cd $REMOTE_DIR && git pull --ff-only && pip install -q -e . --no-deps && git log -1 --oneline"
