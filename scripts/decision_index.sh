#!/usr/bin/env bash
# Run one checkpoint through the Decision Index public suite (apolinario/decision-index).
#   bash scripts/decision_index.sh <checkpoint_dir> <name> [port] [extra decision_index args, e.g. --limit 100]
# Needs: the kit installed (pip install -e decision-index), the suite staged by `suite import`, `pip install -e ".[serve]"`.
# Results: $DI_RUNS/<name>/{results.jsonl,scores.json,index.json}. Re-running resumes; errored rows are retried.
set -euo pipefail
ckpt=${1:?checkpoint dir}
name=${2:?run name}
port=${3:-8009}
shift $(( $# < 3 ? $# : 3 ))
out=${DI_RUNS:-runs/decision_index}/$name
mkdir -p "$out"

jt serve "$ckpt" --port "$port" --host 127.0.0.1 > "$out/serve.log" 2>&1 &
srv=$!
trap 'kill $srv 2>/dev/null || true' EXIT
for _ in $(seq 1 180); do
  curl -sf "http://127.0.0.1:$port/v1/models" > /dev/null && break
  kill -0 $srv 2>/dev/null || { echo "server died, see $out/serve.log"; exit 1; }
  sleep 2
done

python -m decision_index pipeline --engine http \
  --option base_url="http://127.0.0.1:$port" --option model="$name" \
  --out "$out" --compact "$@"
