"""Print one table row per run directory: benchmark scores, holdout accuracy/ECE, temperature."""

import json
import sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else "runs")
for m in sorted(root.glob("**/metrics.json")):
    r = json.loads(m.read_text(encoding="utf-8"))
    bench = {k: round(100 * v["score"], 2) for k, v in (r.get("benchmarks") or {}).items() if v.get("n")}
    hold = r.get("holdout") or r.get("holdout_uncalibrated") or {}
    print(f"{m.parent.relative_to(root)}\tsteps={r.get('steps')}\tholdout_acc={hold.get('accuracy', 0):.3f}\tece={hold.get('ece', 0):.3f}"
          f"\tT={r.get('temperature', {}).get('default', '-')}\t{json.dumps(bench)}")
