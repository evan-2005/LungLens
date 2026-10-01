"""
Model backends for the LLM and VLM report generators.

Two interchangeable backends with one method, generate():
  OllamaBackend  local models over Ollama's HTTP API (default; nothing leaves
                 the machine)
  ClaudeBackend  Claude through the official Anthropic SDK (hosted; opt-in)

Both run deterministically where the provider allows it (temperature 0 and a
fixed seed on Ollama; Claude models from Opus 4.7 on do not accept sampling
parameters). Every failure surfaces as BackendError so the runner can record it
per film instead of silently dropping the film.
"""
import base64
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_CLAUDE_MODEL = "claude-opus-5"
CLAUDE_FALLBACK_BETA = "server-side-fallback-2026-07-01"
DEFAULT_TIMEOUT_S = 300


class BackendError(RuntimeError):
    """Any failure to get a usable reply from a model."""


@dataclass(frozen=True)
class LLMResult:
    text: str
    model: str
    latency_s: float
    meta: dict = field(default_factory=dict)


def _b64(data):
    return base64.standard_b64encode(data).decode("ascii")


def _http_post_json(url, payload, timeout):
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


class OllamaBackend:
    name = "ollama"

    def __init__(self, model, host=DEFAULT_OLLAMA_HOST, seed=42, timeout=DEFAULT_TIMEOUT_S,
                 post=_http_post_json):
        self.model = model
        self.host = host.rstrip("/")
        self.seed = seed
        self.timeout = timeout
        self._post = post

    def generate(self, system, user, images=None, schema=None, max_tokens=1024):
        user_msg = {"role": "user", "content": user}
        if images:
            user_msg["images"] = [_b64(img) for img in images]
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, user_msg],
            "stream": False,
            "options": {"temperature": 0, "seed": self.seed, "num_predict": max_tokens},
        }
        if schema is not None:
            payload["format"] = schema
        start = time.perf_counter()
        try:
            data = self._post(f"{self.host}/api/chat", payload, self.timeout)
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as e:
            raise BackendError(f"Ollama request failed ({self.host}, {self.model}): {e}") from e
        latency = time.perf_counter() - start
        try:
            text = data["message"]["content"]
        except (KeyError, TypeError) as e:
            raise BackendError(f"Unexpected Ollama response: {str(data)[:200]}") from e
        return LLMResult(text=text, model=data.get("model", self.model), latency_s=latency,
                         meta={"eval_count": data.get("eval_count")})


class ClaudeBackend:
    """
    Claude via the Anthropic SDK. Credentials come from the environment
    (ANTHROPIC_API_KEY or an `ant auth login` profile); nothing is hardcoded.

    Server-side refusal fallbacks are on by default: if the model declines, the
    API re-runs the request on Anthropic's recommended fallback model in the
    same call. The model that actually answered is recorded in the result.
    """
    name = "claude"

    def __init__(self, model=DEFAULT_CLAUDE_MODEL, client=None, fallbacks=True, effort=None):
        self.model = model
        self.fallbacks = fallbacks
        self.effort = effort
        if client is None:
            try:
                import anthropic
            except ImportError as e:
                raise BackendError("Install the SDK first: pip install anthropic") from e
            client = anthropic.Anthropic()
        self._client = client

    def _request(self, system, user, images, schema, max_tokens):
        content = [{"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                                "data": _b64(img)}} for img in images or []]
        content.append({"type": "text", "text": user})
        kwargs = {"model": self.model, "max_tokens": max_tokens, "system": system,
                  "messages": [{"role": "user", "content": content}]}
        output_config = {}
        if schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": schema}
        if self.effort:
            output_config["effort"] = self.effort
        if output_config:
            kwargs["output_config"] = output_config
        if self.fallbacks:
            kwargs["betas"] = [CLAUDE_FALLBACK_BETA]
            kwargs["fallbacks"] = "default"
        return kwargs

    def generate(self, system, user, images=None, schema=None, max_tokens=16000):
        kwargs = self._request(system, user, images, schema, max_tokens)
        start = time.perf_counter()
        try:
            response = self._client.beta.messages.create(**kwargs)
        except Exception as e:  # SDK errors are already retried by the client
            raise BackendError(f"Claude request failed ({self.model}): "
                               f"{type(e).__name__}: {e}") from e
        latency = time.perf_counter() - start
        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            raise BackendError(f"Claude refused (category: {getattr(details, 'category', None)})")
        text = "".join(b.text for b in response.content if getattr(b, "type", "") == "text")
        return LLMResult(text=text, model=getattr(response, "model", self.model), latency_s=latency,
                         meta={"stop_reason": response.stop_reason})
