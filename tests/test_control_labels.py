from jevtrainer.data.base import _cache_version
from jevtrainer.data.converters.control import cheapest_success, compact_label, correctness, model_tier


def test_cache_version_does_not_reuse_an_older_converter():
    assert _cache_version("train-cap100000-v2") == "2"
    assert _cache_version("train-cap20000-v1-fps1") == "1"
    assert _cache_version("train-cap100000-v2-fps1-full") == "2"
    assert _cache_version("train-cap100000-v10") == "10"


def test_model_tier_from_size_and_known_apis():
    assert model_tier("wxai-llama-3-2-1b-instruct") == "fast"
    assert model_tier("Qwen__Qwen1.5-7B-Chat") == "fast"
    assert model_tier("meta-llama__Llama-2-13b-chat-hf") == "mid"
    assert model_tier("wxai-mixtral-8x7b-instruct-v01") == "mid"
    assert model_tier("wxai-llama-3-405b-instruct") == "strong"
    assert model_tier("openai-gpt-4o-mini") == "fast"
    assert model_tier("openai-gpt-4o") == "strong"
    assert model_tier("aws-claude-3-5-sonnet-v1") == "strong"


def test_cheapest_tier_that_scored():
    assert cheapest_success({"a-1b": 1.0, "b-70b": 1.0}) == "fast"
    assert cheapest_success({"a-1b": 0.2, "b-13b": 0.9, "c-70b": 1.0}) == "mid"
    assert cheapest_success({"a-1b": 0.0, "b-13b": 0.4}) == "strong"


def test_sprout_score_blob():
    cell = {"judge_response": '{"correctness_score": 0.8}', "score": 1}
    assert correctness(cell) == 0.8


def test_compact_bands():
    assert compact_label(18, 20) == "keep"
    assert compact_label(1, 40) == "drop"
    assert compact_label(10, 20) == "truncate"
    assert compact_label(0, 0) is None
