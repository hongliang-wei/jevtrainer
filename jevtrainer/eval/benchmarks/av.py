"""Audio-visual QA on held-out splits (video + sound track). Suite "av-dev"."""

from jevtrainer.eval.base import bench

AV = "av-dev"
bench("avqa_val", "avqa", "val", suite=AV, area="av", max_samples=1000)
bench("music_avqa_val", "music_avqa", "val", suite=AV, area="av", max_samples=1000)
bench("music_avqa_test", "music_avqa", "test", suite=AV, area="av", max_samples=1000)
