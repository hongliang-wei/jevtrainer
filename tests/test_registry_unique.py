"""Two converters must never register the same dataset name (two agents writing the same dataset is the usual cause).

Static scan, so the failing test names both files; the import check catches registrations made some other way.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

PKG = Path(__file__).resolve().parents[1] / "jevtrainer"
SPEC = re.compile(r'DatasetSpec\(\s*["\']([A-Za-z0-9_]+)["\']')


def test_dataset_names_registered_once():
    seen: dict[str, list[str]] = defaultdict(list)
    for path in PKG.rglob("*.py"):
        for name in SPEC.findall(path.read_text(encoding="utf-8-sig")):
            seen[name].append(path.relative_to(PKG).as_posix())
    dups = {n: files for n, files in seen.items() if len(files) > 1}
    assert not dups, "dataset registered in more than one file (keep one): " + "; ".join(
        f"{n}: {', '.join(files)}" for n, files in sorted(dups.items()))


def test_all_converters_import():
    import jevtrainer.data.converters  # noqa: F401  (raises KeyError on a duplicate registration)
    from jevtrainer.registry import DATASETS

    assert len(DATASETS.names()) > 100
