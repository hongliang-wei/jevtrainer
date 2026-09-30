"""GUI / browser agents on held-out splits of the GUI training sets. Suite "gui-v1"."""

from jevtrainer.eval.base import bench

GUI = "gui-v1"
for _s in ("test_task", "test_website", "test_domain"):
    bench(f"mm_mind2web_{_s}", "mm_mind2web", _s, suite=GUI, area="agents", max_samples=1000)
for _p in ("web_single", "web_multi", "smartphone"):
    bench(f"guiact_{_p}_test", f"guiact_{_p}", "test", suite=GUI, area="agents", max_samples=1000)
bench("omniact_test", "omniact", "test", suite=GUI, area="agents", max_samples=1000)
bench("weblinx_valid", "weblinx", "valid", suite=GUI, area="agents", max_samples=1000)
