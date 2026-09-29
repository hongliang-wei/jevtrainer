"""Public benchmarks that download automatically. Suite "core" is a quick broad check."""

from jevtrainer.eval.base import bench

CORE = "core"
bench("mmlu", "mmlu", "test", suite=CORE, area="knowledge", max_samples=2000)
bench("mmlu_pro", "mmlu_pro", "test", suite=CORE, area="knowledge", max_samples=2000)
bench("arc_challenge_easy", "arc", "test", suite=CORE, area="knowledge")
bench("openbookqa", "openbookqa", "test", suite=CORE, area="knowledge")
bench("csqa", "csqa", "validation", suite=CORE, area="commonsense")
bench("hellaswag", "hellaswag", "test", suite=CORE, area="commonsense", max_samples=2000)
bench("winogrande", "winogrande", "test", suite=CORE, area="commonsense")
bench("gsm8k_mc", "gsm8k_mc", "test", suite=CORE, area="math")
bench("anli", "anli", "test", suite=CORE, area="nli", metric="macro_f1")
bench("mnli", "mnli", "test", suite=CORE, area="nli", max_samples=2000)
bench("boolq", "boolq", "validation", suite=CORE, area="reading")
bench("banking77", "banking77", "test", suite=CORE, area="intent", metric="macro_f1")
bench("clinc150", "clinc150", "test", suite=CORE, area="intent", metric="macro_f1", max_samples=2000)
bench("sst5", "sst5", "test", suite=CORE, area="sentiment")
bench("jailbreak_classification", "jailbreak_classification", "test", suite=CORE, area="safety")
bench("toxic_chat", "toxic_chat", "test", suite=CORE, area="safety", max_samples=2000)
bench("scienceqa_img", "scienceqa_img", "test", suite="vision", area="vision")
