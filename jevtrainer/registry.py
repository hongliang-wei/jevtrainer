"""Name -> object registries. Every extension point (datasets, benchmarks, readouts,
model families, losses, trainers) is one file plus one `@REGISTRY.register(...)`."""

from __future__ import annotations

import difflib
from typing import Any, Callable, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, kind: str):
        self.kind = kind
        self._items: dict[str, T] = {}

    def register(self, name: str | None = None) -> Callable[[T], T]:
        def deco(obj: T) -> T:
            self.add(name or getattr(obj, "name", None) or obj.__name__, obj)
            return obj

        return deco

    def add(self, name: str, obj: T) -> None:
        if name in self._items:
            raise KeyError(f"{self.kind} '{name}' is already registered")
        self._items[name] = obj

    def get(self, name: str) -> T:
        _load_plugins()
        if name not in self._items:
            close = difflib.get_close_matches(name, self._items, n=3, cutoff=0.5)
            hint = f" Did you mean: {', '.join(close)}?" if close else ""
            raise KeyError(f"unknown {self.kind} '{name}'.{hint} Run `jt {self.kind} list` to see all.")
        return self._items[name]

    def __contains__(self, name: str) -> bool:
        _load_plugins()
        return name in self._items

    def names(self) -> list[str]:
        _load_plugins()
        return sorted(self._items)

    def items(self) -> list[tuple[str, T]]:
        _load_plugins()
        return sorted(self._items.items())


DATASETS: Registry[Any] = Registry("data")
BENCHMARKS: Registry[Any] = Registry("bench")
READOUTS: Registry[Any] = Registry("readout")
FAMILIES: Registry[Any] = Registry("model")
LOSSES: Registry[Any] = Registry("loss")
TRAINERS: Registry[Any] = Registry("trainer")

_loaded = False


def _load_plugins() -> None:
    """Import the built-in modules once so their decorators run."""
    global _loaded
    if _loaded:
        return
    _loaded = True
    import importlib
    import pkgutil

    for pkg_name in (
        "jevtrainer.data.converters",
        "jevtrainer.eval.benchmarks",
        "jevtrainer.readouts",
        "jevtrainer.model.families",
        "jevtrainer.train",
    ):
        pkg = importlib.import_module(pkg_name)
        for info in pkgutil.iter_modules(pkg.__path__):
            if not info.name.startswith("_"):
                importlib.import_module(f"{pkg_name}.{info.name}")
