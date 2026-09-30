"""Chinese suites. zh-dev: validation splits of the Chinese training sources (checkpoint selection, never a
test split). zh-bench: eval-only Chinese exams and safety knowledge (C-Eval val, CMMLU, AGIEval Chinese,
JEC-QA, FinanceIQ, Chinese-SafetyQA). vision-zh: MMBench-CN, CMMMU val. gui-zh: CAGUI."""

from jevtrainer.eval.base import bench

for name, n in [
    ("tnews", 300), ("thucnews", 280), ("iflytek", 300), ("fincuge_news", 300), ("massive_intent_zh", 300), ("zh_sentiment3", 300),
    ("c_stance", 300), ("toxicn", 300), ("cold", 300), ("tc260", 300), ("hc3_zh", 300), ("afqmc", 300), ("bq_corpus", 300),
    ("kuake_qqr", 300), ("zh_stsb", 300), ("t2_rerank", 300), ("cmnli", 300), ("c3", 300), ("chid", 300), ("cail2018", 300),
    ("legal_case_zh", 300), ("cmexam", 300), ("cvalues_rlhf", 300), ("ultrafeedback_zh", 300),
]:
    bench(f"{name}_dev", name, "validation", metric="accuracy", suite="zh-dev", area="zh", max_samples=n)

bench("ceval_val", "ceval_val", "validation", metric="accuracy", suite="zh-bench", area="zh")
bench("cmmlu", "cmmlu", "test", metric="accuracy", suite="zh-bench", area="zh")
bench("agieval_zh", "agieval_zh", "test", metric="accuracy", suite="zh-bench", area="zh")
bench("jecqa", "jecqa", "test", metric="accuracy", suite="zh-bench", area="zh")
bench("financeiq", "financeiq", "test", metric="accuracy", suite="zh-bench", area="zh")
bench("chinese_safetyqa", "chinese_safetyqa", "test", metric="accuracy", suite="zh-bench", area="zh")
bench("mmbench_cn", "mmbench_cn", "test", metric="accuracy", suite="vision-zh", area="zh")
bench("cmmmu", "cmmmu", "test", metric="accuracy", suite="vision-zh", area="zh")
bench("cagui", "cagui", "test", metric="accuracy", suite="gui-zh", area="zh")
