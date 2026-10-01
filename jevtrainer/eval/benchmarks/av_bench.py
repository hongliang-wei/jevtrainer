"""Audio / audio-visual / video evaluation suites (full splits, eval_only datasets).

    av-omni      audio-visual questions that need the sound  (WorldSense, Daily-Omni, OmniBench, AV-Odyssey, AV-SpeakerBench)
    audio-bench  audio understanding                          (MMAU test-mini, MMAR, MMSU, VoiceBench mmsu / openbookqa)
    video-bench  video understanding                          (MVBench, TempCompass, EgoSchema, LongVideoBench, Video-MME, Perception Test)

Not included (no usable open copy here): AVI-Bench (gated), AVHBench (questions only, no videos).
`video_mme_sub` (Video-MME with subtitles) is its own suite "video-bench-sub" so that video-bench stays one pass over each video set.
Use `ablate: mute | black | shuffle_audio` in the eval config to see how much each modality is really used.
"""

from jevtrainer.eval.base import bench

OMNI, AUDIO, VIDEO = "av-omni", "audio-bench", "video-bench"

bench("worldsense", "worldsense", "test", suite=OMNI, area="av")
bench("daily_omni", "daily_omni", "test", suite=OMNI, area="av")
bench("omnibench", "omnibench", "test", suite=OMNI, area="av")
bench("av_odyssey", "av_odyssey", "test", suite=OMNI, area="av")
bench("av_speakerbench", "av_speakerbench", "test", suite=OMNI, area="av")

bench("mmau_mini", "mmau_mini", "test", suite=AUDIO, area="audio")
bench("mmar", "mmar", "test", suite=AUDIO, area="audio")
bench("mmsu", "mmsu", "test", suite=AUDIO, area="audio")
bench("voicebench_mmsu", "voicebench_mmsu", "test", suite=AUDIO, area="audio")
bench("voicebench_openbookqa", "voicebench_openbookqa", "test", suite=AUDIO, area="audio")

bench("mvbench", "mvbench", "test", suite=VIDEO, area="video")
bench("tempcompass", "tempcompass", "test", suite=VIDEO, area="video")
bench("egoschema", "egoschema", "test", suite=VIDEO, area="video")
bench("longvideobench", "longvideobench", "val", suite=VIDEO, area="video")
bench("video_mme", "video_mme", "test", suite=VIDEO, area="video")
bench("perceptiontest_val", "perceptiontest_val", "val", suite=VIDEO, area="video")
bench("video_mme_sub", "video_mme_sub", "test", suite="video-bench-sub", area="video")
