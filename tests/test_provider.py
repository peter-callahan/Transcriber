import base64
import types
import pytest

import pipeline_utils as pu

IMG = base64.b64encode(b"jpegbytes").decode()
MESSAGES = [
    {"role": "user", "content": [{"type": "text", "text": "hi"}, {"type": "image", "base64": IMG}]},
    {"role": "assistant", "content": [{"type": "text", "text": "{bad"}]},
    {"role": "user", "content": [{"type": "text", "text": "fix it"}]},
]


def test_openai_translation():
    out = pu.to_provider_messages(MESSAGES, "openai")
    assert out[0]["content"][1] == {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{IMG}"}}
    assert out[1] == {"role": "assistant", "content": [{"type": "text", "text": "{bad"}]}


def test_anthropic_translation():
    out = pu.to_provider_messages(MESSAGES, "anthropic")
    assert out[0]["content"][1]["source"] == {"type": "base64", "media_type": "image/jpeg", "data": IMG}
    assert out[2]["role"] == "user"


def test_bedrock_translation_decodes_bytes():
    out = pu.to_provider_messages(MESSAGES, "bedrock")
    assert out[0]["content"][0] == {"text": "hi"}
    assert out[0]["content"][1]["image"]["source"]["bytes"] == b"jpegbytes"
    assert out[1]["content"] == [{"text": "{bad"}]


class _Usage:
    prompt_tokens = 7
    completion_tokens = 3
    input_tokens = 7
    output_tokens = 3


def test_call_api_openai_returns_text_and_usage(monkeypatch):
    class Client:
        class chat:
            class completions:
                @staticmethod
                def create(model, messages, max_tokens):
                    msg = types.SimpleNamespace(content="hello")
                    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)], usage=_Usage())
    monkeypatch.setenv("AI_PROVIDER", "openai")
    monkeypatch.setattr(pu, "get_client", lambda: Client())
    text, usage = pu.call_api(MESSAGES, 100)
    assert text == "hello"
    assert usage == {"input_tokens": 7, "output_tokens": 3}


def test_call_api_anthropic_returns_text_and_usage(monkeypatch):
    class Client:
        class messages:
            @staticmethod
            def create(model, messages, max_tokens):
                return types.SimpleNamespace(content=[types.SimpleNamespace(text="hey")], usage=_Usage())
    monkeypatch.setenv("AI_PROVIDER", "anthropic")
    monkeypatch.setattr(pu, "get_client", lambda: Client())
    assert pu.call_api(MESSAGES, 100) == ("hey", {"input_tokens": 7, "output_tokens": 3})


def test_call_api_bedrock_returns_text_and_usage(monkeypatch):
    class Client:
        @staticmethod
        def converse(modelId, messages, inferenceConfig):
            return {"output": {"message": {"content": [{"text": "yo"}]}},
                    "usage": {"inputTokens": 7, "outputTokens": 3}}
    monkeypatch.setenv("AI_PROVIDER", "bedrock")
    monkeypatch.setattr(pu, "get_client", lambda: Client())
    assert pu.call_api(MESSAGES, 100) == ("yo", {"input_tokens": 7, "output_tokens": 3})


class RateLimitError(Exception):
    pass


class AuthenticationError(Exception):
    status_code = 401


class StatusError(Exception):
    def __init__(self, code):
        self.status_code = code


class ClientError(Exception):
    def __init__(self, code):
        self.response = {"Error": {"Code": code}}


@pytest.mark.parametrize("exc,expected", [
    (RateLimitError("slow down"), True),
    (StatusError(503), True),
    (StatusError(429), True),
    (ClientError("ThrottlingException"), True),
    (AuthenticationError("bad key"), False),
    (StatusError(400), False),
    (ClientError("ValidationException"), False),
    (ValueError("nope"), False),
])
def test_is_transient(exc, expected):
    assert pu.is_transient(exc) is expected
