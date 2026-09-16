import base64
import os
import json
import logging
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
