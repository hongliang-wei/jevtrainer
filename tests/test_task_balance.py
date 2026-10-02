import random

from jevtrainer.data.base import noul_record
from jevtrainer.data.mixture import balance_tasks, repeat_up


def _recs(task: str, n: int):
    return [noul_record(f"{task}{i}", {"q": f"{task} {i}"}, "ok?", True, task=task) for i in range(n)]


def test_each_task_gets_the_same_amount():
    rs = _recs("a", 500) + _recs("b", 120) + _recs("c", 40)
    out, tasks = balance_tasks(rs, "task", per_group=100, repeat_to=100, rng=random.Random(0))
    assert tasks == {"a": 100, "b": 100, "c": 100}  # c is repeated 2.5x, within the 3x limit
    assert len(out) == 300


def test_repeat_is_limited_to_three_copies():
    rs = _recs("a", 10)
    assert len(repeat_up(rs, 1000)) == 30
    out, tasks = balance_tasks(rs, "task", per_group=None, repeat_to=1000, rng=random.Random(0))
    assert tasks == {"a": 30}


def test_no_limits_keeps_everything():
    rs = _recs("a", 7) + _recs("b", 3)
    out, tasks = balance_tasks(rs, "task", None, None, random.Random(0))
    assert tasks == {"a": 7, "b": 3} and len(out) == 10
