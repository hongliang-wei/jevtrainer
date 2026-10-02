"""Small additions to `avkit` for the audio converters (kept apart from avkit so that both can evolve independently).

The hub mirror rate-limits repo listings (HTTP 429) when many jobs run at once, so converters whose shard names are known
pass them in (`pq_files(repo, prefix, known)`); the listing is only used when no names are given.
"""

from __future__ import annotations

from jevtrainer.data import avkit


def pq_files(repo: str, prefix: str = "", known: list[str] | None = None) -> list[str]:
    """Parquet shards of `repo` whose name starts with `prefix`, taken from `known` without any listing request when given."""
    if known is not None:
        return sorted(f for f in known if f.startswith(prefix))
    return avkit.parquet_files(repo, prefix)
