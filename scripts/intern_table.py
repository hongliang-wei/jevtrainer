"""One markdown row per evaluated checkpoint of a run, in Intern-Decision's accuracy-v1 column order."""

import json
import sys
from pathlib import Path

COLS = ["jevbench_easy", "jevbench_original", "jevbench_hard", "typed_decision", "toolace", "agnews", "wildjailbreak"]
TARGET = [97.92, 80.56, 52.25, 77.35, 94.52, 88.61, 64.48]


def row(name, res, temp):
    vals = [100 * res[c]["score"] if c in res and res[c].get("n") else None for c in COLS]
    avg = sum(v for v in vals if v is not None) / len([v for v in vals if v is not None]) if any(v is not None for v in vals) else None
    hard = res.get("jevbench_hard", {})
    cells = " | ".join("" if v is None else f"{v:.2f}" for v in vals)
    return f"| {name} | {cells} | {'' if avg is None else f'{avg:.2f}'} | {hard.get('ece', float('nan')):.3f} | {temp} |"


run = Path(sys.argv[1])
lines = ["| checkpoint | Easy | Original | Hard | Typed | ToolACE | AG News | WJB | Avg | Hard ECE | T |", "|---|" + "---:|" * 11]
lines.append("| Intern-Decision-0.8B | " + " | ".join(f"{v:.2f}" for v in TARGET) + f" | {sum(TARGET) / 7:.2f} | 0.066 | 2.748 |")
cands = sorted(run.glob("step-*/eval/results.json"), key=lambda p: int(p.parent.parent.name.split("-")[1]))
cands += [p for p in [run / "eval" / "results.json"] if p.exists()]
for p in cands:
    ck = p.parent.parent
    cal = ck / "calibration.json"
    temp = json.loads(cal.read_text())["temperature"].get("default") if cal.exists() else 1.0
    lines.append(row(ck.name if ck != run else "final", json.loads(p.read_text()), f"{temp:.3f}"))
out = "\n".join(lines)
(run / "intern_table.md").write_text(out + "\n")
print(out)
