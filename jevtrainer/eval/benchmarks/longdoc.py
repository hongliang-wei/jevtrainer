"""longdoc-dev: validation splits of the long-document training sources, for checkpoint selection and
temperature checks on a harder, longer distribution than the 2% holdout. Never a test split."""

from jevtrainer.eval.base import bench

SUITE = "longdoc-dev"
for name, n, metric in [
    ("musique", 300, "accuracy"), ("quality", None, "accuracy"), ("timeqa", 300, "accuracy"), ("hotpotqa", 200, "accuracy"),
    ("wikihop", 200, "accuracy"), ("finqa", 300, "accuracy"), ("tatqa", None, "accuracy"), ("docnli", 200, "accuracy"),
    ("wice", 200, "accuracy"), ("halueval_summ", 200, "accuracy"), ("ecthr", 200, "accuracy"), ("maud", 200, "accuracy"),
    ("qasper", None, "accuracy"), ("legalbench_consumer_contracts", None, "accuracy"), ("legalbench_rules", 200, "accuracy"),
    ("ppe_ifeval", None, "accuracy"), ("reward_bench2", None, "accuracy"), ("mt_bench_human", None, "accuracy"),
    ("swe_agent", 200, "accuracy"), ("sharc", 200, "macro_f1"),
]:
    bench(f"{name}_dev", name, "validation", metric=metric, suite=SUITE, area="longdoc", max_samples=n)
