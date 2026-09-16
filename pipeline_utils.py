import base64
import os
import json
import logging
import time
from dotenv import load_dotenv
from dataclasses import dataclass, field, asdict
from typing import Optional

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("transcriber")

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.heic')
INPUT_FOLDER = os.path.expanduser(os.getenv('INPUT_FOLDER', 'input_images'))


def get_file_order(folder_path):
    """Files in user-defined order from order.json; sorted image files as fallback."""
    order_file = os.path.join(folder_path, 'order.json')
    if os.path.exists(order_file):
        try:
            with open(order_file, 'r') as f:
                return json.load(f).get('files', [])
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to read order.json: {e}, falling back to sorted order")
    return sorted(
        f for f in os.listdir(folder_path) if f.lower().endswith(IMAGE_EXTENSIONS)
    )


def resolve_image_path(folder_path, image_file):
    """(filename, path) for image_file, remapping to the .jpg that process_images
    produces for PNG/HEIC inputs. None if neither exists."""
    path = os.path.join(folder_path, image_file)
    if os.path.isfile(path):
        return image_file, path
    jpg_name = os.path.splitext(image_file)[0] + '.jpg'
    jpg_path = os.path.join(folder_path, jpg_name)
    if os.path.isfile(jpg_path):
        return jpg_name, jpg_path
    return None


@dataclass
class StageResult:
    ok: bool
    error: Optional[str] = None


@dataclass
class CallMeta:
    run_id: str
    group_name: str
    filename: Optional[str]
    kind: str                       # "page" | "summary"
    provider: str
    model: str
    prompt_hash: str
    attempts: int
    latency_ms: int
    input_tokens: int
    output_tokens: int
    word_count: int
    uncertain_count: int
    timestamp: str


@dataclass
class PageResult:
    filename: str
    status: str                     # "done" | "warning" | "failed"
    attempts: int = 0
    data: Optional[dict] = None
    uncertain: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    error: Optional[str] = None
    history: list = field(default_factory=list)
    meta: Optional[CallMeta] = None


@dataclass
class GroupResult:
    group_name: str
    status: str                     # "done" | "warning" | "failed"
    file_order: list
    image_paths: list
    pages: list                     # list[PageResult]
    summary: Optional[dict] = None
    summary_attempts: int = 0
    summary_meta: Optional[CallMeta] = None
    summary_error: Optional[str] = None
    warnings: list = field(default_factory=list)
    errors: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


def rollup_status(page_statuses, summary_error, group_warnings):
    if not page_statuses or "failed" in page_statuses or summary_error:
        return "failed"
    if "warning" in page_statuses or group_warnings:
        return "warning"
    return "done"


# --- Provider layer ---------------------------------------------------------

_client = None
_client_provider = None


def get_provider():
    return os.getenv('AI_PROVIDER', 'openai').lower()


def get_model():
    provider = get_provider()
    if provider == 'anthropic':
        return os.getenv('ANTHROPIC_MODEL', 'claude-sonnet-4-6')
    if provider == 'bedrock':
        return os.getenv('BEDROCK_MODEL', '')
    return os.getenv('OPENAI_MODEL', 'gpt-4o')


def get_client():
    """SDK client for the configured provider, created on first use."""
    global _client, _client_provider
    provider = get_provider()
    if _client is not None and _client_provider == provider:
        return _client
    if provider == 'anthropic':
        import anthropic
        _client = anthropic.Anthropic()
    elif provider == 'bedrock':
        import boto3
        from botocore.config import Config as BotocoreConfig
        _client = boto3.client(
            'bedrock-runtime',
            region_name=os.getenv('AWS_REGION', 'us-east-1'),
            config=BotocoreConfig(read_timeout=300, connect_timeout=10),
        )
    else:
        import openai
        _client = openai.OpenAI()
    _client_provider = provider
    logger.info(f"Using AI provider: {provider}, model: {get_model()}")
    return _client


def to_provider_messages(messages, provider):
    out = []
    for message in messages:
        blocks = []
        for block in message["content"]:
            if block["type"] == "text":
                if provider == 'bedrock':
                    blocks.append({"text": block["text"]})
                else:
                    blocks.append({"type": "text", "text": block["text"]})
            elif block["type"] == "image":
                if provider == 'anthropic':
                    blocks.append({"type": "image", "source": {
                        "type": "base64", "media_type": "image/jpeg", "data": block["base64"]}})
                elif provider == 'bedrock':
                    blocks.append({"image": {"format": "jpeg", "source": {
                        "bytes": base64.b64decode(block["base64"])}}})
                else:
                    blocks.append({"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{block['base64']}"}})
        out.append({"role": message["role"], "content": blocks})
    return out


def call_api(messages, max_tokens):
    """Send a provider-neutral message list. Returns (text, usage)."""
    provider = get_provider()
    client = get_client()
    model = get_model()
    provider_messages = to_provider_messages(messages, provider)

    if provider == 'anthropic':
        r = client.messages.create(model=model, messages=provider_messages, max_tokens=max_tokens)
        return r.content[0].text, {"input_tokens": r.usage.input_tokens,
                                   "output_tokens": r.usage.output_tokens}
    if provider == 'bedrock':
        r = client.converse(modelId=model, messages=provider_messages,
                            inferenceConfig={"maxTokens": max_tokens})
        usage = r.get('usage', {})
        return r['output']['message']['content'][0]['text'], {
            "input_tokens": usage.get('inputTokens', 0),
            "output_tokens": usage.get('outputTokens', 0)}
    r = client.chat.completions.create(model=model, messages=provider_messages, max_tokens=max_tokens)
    return r.choices[0].message.content, {
        "input_tokens": getattr(r.usage, 'prompt_tokens', 0),
        "output_tokens": getattr(r.usage, 'completion_tokens', 0)}


TRANSIENT_EXCEPTION_NAMES = {
    "RateLimitError", "APITimeoutError", "APIConnectionError", "InternalServerError",
    "ReadTimeoutError", "ConnectTimeoutError", "EndpointConnectionError", "ConnectionClosedError",
}
TRANSIENT_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}
TRANSIENT_AWS_CODES = {
    "ThrottlingException", "ServiceUnavailableException", "ModelTimeoutException",
    "InternalServerException", "ModelNotReadyException",
}


def is_transient(exc):
    """True for rate limits, timeouts, connection drops and 5xx from any provider SDK."""
    if type(exc).__name__ in TRANSIENT_EXCEPTION_NAMES:
        return True
    if getattr(exc, "status_code", None) in TRANSIENT_STATUS_CODES:
        return True
    response = getattr(exc, "response", None)
    if isinstance(response, dict) and response.get("Error", {}).get("Code") in TRANSIENT_AWS_CODES:
        return True
    return False


# --- Retry -----------------------------------------------------------------

class ParseError(Exception):
    """Model output is not valid JSON."""


class StructureError(Exception):
    """Model output parsed but is missing or mistyping a required field."""


@dataclass
class Attempt:
    ok: bool
    parsed: Optional[dict]
    attempts: int
    history: list
    error: Optional[str]
    usage: dict
    latency_ms: int
    raw_text: Optional[str]


REPAIR_TEMPLATE = (
    "The previous response could not be used: {error}. "
    "Return the same transcription in the required JSON structure exactly as specified. "
    "Do not change, correct, reorder, or omit any transcribed text."
)
BACKOFF_SECONDS = (1, 4, 10)


def _text_message(role, text):
    return {"role": role, "content": [{"type": "text", "text": text}]}


def call_with_retry(messages, max_tokens, parse, max_calls=3, call_api_fn=None, sleep_fn=time.sleep):
    """Call the model until parse() accepts the output or max_calls is spent.

    Transport errors are re-sent unchanged after backoff. Parse/structure errors
    continue the conversation with the bad output and a repair instruction.
    Any other exception ends the attempt immediately.
    """
    call_api_fn = call_api_fn or call_api
    conversation = list(messages)
    history = []
    usage = {"input_tokens": 0, "output_tokens": 0}
    started = time.monotonic()
    attempts = 0
    raw = None

    while attempts < max_calls:
        attempts += 1
        raw = None
        try:
            raw, call_usage = call_api_fn(conversation, max_tokens)
        except Exception as exc:
            if is_transient(exc):
                history.append({"kind": "transport", "error": str(exc)})
                logger.warning(f"Transient API error on attempt {attempts}: {exc}")
                if attempts < max_calls:
                    sleep_fn(BACKOFF_SECONDS[min(attempts - 1, len(BACKOFF_SECONDS) - 1)])
                continue
            history.append({"kind": "fatal", "error": str(exc)})
            logger.error(f"Non-retryable API error: {exc}")
            break

        usage["input_tokens"] += call_usage.get("input_tokens", 0)
        usage["output_tokens"] += call_usage.get("output_tokens", 0)

        kind = error = None
        try:
            parsed = parse(raw)
        except ParseError as exc:
            kind, error = "parse", str(exc)
        except StructureError as exc:
            kind, error = "structure", str(exc)

        if error is None:
            return Attempt(True, parsed, attempts, history, None, usage,
                           int((time.monotonic() - started) * 1000), raw)

        history.append({"kind": kind, "error": error})
        logger.warning(f"Unusable model output on attempt {attempts} ({kind}): {error}")
        conversation = conversation + [
            _text_message("assistant", raw),
            _text_message("user", REPAIR_TEMPLATE.format(error=error)),
        ]

    last_error = history[-1]["error"] if history else "no response"
    return Attempt(False, None, attempts, history, last_error, usage,
                   int((time.monotonic() - started) * 1000), raw)


def retry_with_feedback(messages, prior_output, feedback, max_tokens, parse, **kwargs):
    """Re-run a call with the prior output and a caller-supplied correction in context."""
    seeded = list(messages) + [
        _text_message("assistant", prior_output),
        _text_message("user", feedback),
    ]
    return call_with_retry(seeded, max_tokens, parse, **kwargs)
