import json
import pytest

import pipeline_utils as pu
from pipeline_utils import call_with_retry, retry_with_feedback, ParseError, StructureError


class RateLimitError(Exception):
    pass


def parse(text):
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ParseError(str(e))
    if "transcription" not in data:
        raise StructureError("missing field: transcription")
    return data


MESSAGES = [{"role": "user", "content": [{"type": "text", "text": "go"}]}]
GOOD = json.dumps({"transcription": "ok"})


def test_success_first_call(scripted_api):
    api = scripted_api([GOOD])
    a = call_with_retry(MESSAGES, 10, parse, call_api_fn=api, sleep_fn=lambda s: None)
    assert a.ok and a.attempts == 1 and a.parsed == {"transcription": "ok"}
    assert a.usage == {"input_tokens": 10, "output_tokens": 5}
    assert a.history == []


def test_transport_error_then_success_backs_off(scripted_api):
    api = scripted_api([RateLimitError("429"), GOOD])
    sleeps = []
    a = call_with_retry(MESSAGES, 10, parse, call_api_fn=api, sleep_fn=sleeps.append)
    assert a.ok and a.attempts == 2
    assert sleeps == [1]
    assert a.history == [{"kind": "transport", "error": "429"}]
    assert api.calls[1] == MESSAGES          # identical redo, no repair turn


def test_bad_json_gets_repair_turn(scripted_api):
    api = scripted_api(["{oops", GOOD])
    a = call_with_retry(MESSAGES, 10, parse, call_api_fn=api, sleep_fn=lambda s: None)
    assert a.ok and a.attempts == 2
    second = api.calls[1]
    assert second[0] == MESSAGES[0]
    assert second[1] == {"role": "assistant", "content": [{"type": "text", "text": "{oops"}]}
    assert second[2]["role"] == "user"
    repair = second[2]["content"][0]["text"]
    assert "could not be used" in repair and "Do not change" in repair
    assert a.history[0]["kind"] == "parse"


def test_missing_field_is_structure_kind(scripted_api):
    api = scripted_api([json.dumps({"title": "x"}), GOOD])
    a = call_with_retry(MESSAGES, 10, parse, call_api_fn=api, sleep_fn=lambda s: None)
    assert a.ok and a.history[0] == {"kind": "structure", "error": "missing field: transcription"}


def test_three_failures_gives_up(scripted_api):
    api = scripted_api(["{a", "{b", "{c", GOOD])
    a = call_with_retry(MESSAGES, 10, parse, call_api_fn=api, sleep_fn=lambda s: None)
    assert not a.ok and a.attempts == 3 and len(a.history) == 3
    assert a.raw_text == "{c"
    assert len(api.script) == 1                 # 4th scripted response never consumed


def test_non_retryable_fails_immediately(scripted_api):
    api = scripted_api([ValueError("bad key"), GOOD])
    a = call_with_retry(MESSAGES, 10, parse, call_api_fn=api, sleep_fn=lambda s: None)
    assert not a.ok and a.attempts == 1
    assert a.history == [{"kind": "fatal", "error": "bad key"}]


def test_usage_and_latency_accumulate(scripted_api, monkeypatch):
    import itertools
    api = scripted_api(["{a", GOOD])
    ticks = itertools.chain([100.0], itertools.repeat(100.25))
    monkeypatch.setattr(pu.time, "monotonic", lambda: next(ticks))
    a = call_with_retry(MESSAGES, 10, parse, call_api_fn=api, sleep_fn=lambda s: None)
    assert a.usage == {"input_tokens": 20, "output_tokens": 10}
    assert a.latency_ms == 250


def test_backoff_sequence(scripted_api):
    api = scripted_api([RateLimitError("1"), RateLimitError("2"), GOOD])
    sleeps = []
    a = call_with_retry(MESSAGES, 10, parse, call_api_fn=api, sleep_fn=sleeps.append)
    assert a.ok and sleeps == [1, 4]


def test_retry_with_feedback_seeds_conversation(scripted_api):
    api = scripted_api([GOOD])
    a = retry_with_feedback(MESSAGES, "{prior", "second line is wrong", 10, parse,
                            call_api_fn=api, sleep_fn=lambda s: None)
    assert a.ok
    sent = api.calls[0]
    assert sent[1]["content"][0]["text"] == "{prior"
    assert sent[2]["content"][0]["text"] == "second line is wrong"


def test_raw_text_is_none_when_final_attempt_produces_no_output(scripted_api):
    # Regression: stale raw_text must not leak across attempts when a parse
    # failure is followed by a transport error with no output.
    api = scripted_api(["{a", RateLimitError("429")])
    a = call_with_retry(MESSAGES, 10, parse, max_calls=2, call_api_fn=api, sleep_fn=lambda s: None)
    assert not a.ok and a.attempts == 2
    assert a.history[0]["kind"] == "parse"
    assert a.history[1]["kind"] == "transport"
    assert a.raw_text is None  # Final attempt produced no output
