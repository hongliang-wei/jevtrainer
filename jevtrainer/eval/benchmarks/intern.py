"""Intern-Decision accuracy-v1 (7 suites) and the known-distribution calibration pilot."""

from jevtrainer.eval.base import bench

SUITE = "intern-accuracy-v1"
bench("jevbench_easy", "intern/jevbench_easy", suite=SUITE, area="typed")
bench("jevbench_original", "intern/jevbench_original", suite=SUITE, area="typed")
bench("jevbench_hard", "intern/jevbench_hard", suite=SUITE, area="typed")
bench("typed_decision", "intern/typed_decisions_test", suite=SUITE, area="typed", description="decision-level accuracy over 2,000 decisions")
bench("toolace", "intern/toolace_test", suite=SUITE, area="tools")
bench("agnews", "intern/agnews_test", suite=SUITE, area="topic")
bench("wildjailbreak", "intern/wildjailbreak_test", suite=SUITE, area="safety")

bench("known_distribution_pilot", "intern/known_distribution_pilot", suite="intern-calibration", area="calibration",
      description="score = 1 - TVD against exact outcome distributions")
