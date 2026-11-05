"""Minimal BYOK Relay client (https://byokrelay.com).

BYOK Relay stores a user's own provider key encrypted on the relay and hands
back a relay token, so ``wild`` never stores provider credentials itself.
Every model is reached through the relay's OpenAI-format endpoint, which
routes ``claude-*``, ``gpt-*``, and ``gemini-*`` models (or an explicit
``provider/model``) to the right provider and normalizes the response.

This module is network-capable and must only be imported lazily, after a
caller has explicitly opted into LLM output.
"""
from __future__ import annotations

import json
import os
import re
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Tuple

from diffgraph import __version__

DEFAULT_RELAY_URL = "https://relay.byokrelay.com"
APP_ID = "wild-cli"
RELAY_URL_ENV = "BYOK_RELAY_URL"
RELAY_TOKEN_ENV = "BYOK_RELAY_TOKEN"

# (method, url, headers, body, timeout) -> (status, headers, body)
Transport = Callable[
    [str, str, Mapping[str, str], Optional[bytes], float],
    Tuple[int, Mapping[str, str], bytes],
]

_PROVIDER_PATTERNS = (
    (re.compile(r"^claude-", re.IGNORECASE), "anthropic"),
    (re.compile(r"^(gpt-|chatgpt-|o[1-9](-|$))", re.IGNORECASE), "openai"),
    (re.compile(r"^gemini-", re.IGNORECASE), "google"),
)
SUPPORTED_PROVIDERS = ("anthropic", "openai", "google")

# `--llm <provider>` uses that provider's latest coding/reasoning flagship.
# Update these as providers release new flagships; an explicit model id
# always takes precedence.
PROVIDER_ALIASES = {
    "anthropic": "anthropic", "claude": "anthropic",
    "openai": "openai", "gpt": "openai",
    "google": "google", "gemini": "google",
}
# ``--llm <provider>:<tier>`` picks a speed/cost tier. ``best`` (the bare
# provider default) is the latest coding/reasoning flagship; ``fast`` is the
# cheapest, quickest current model. Update as providers release new models.
MODEL_TIERS = {
    "anthropic": {"fast": "claude-haiku-5-5", "balanced": "claude-sonnet-5-5", "best": "claude-opus-5-5"},
    "openai": {"fast": "gpt-6-luna", "balanced": "gpt-6.1-sol", "best": "gpt-6-astra"},
    "google": {"fast": "gemini-3.8-flash", "best": "gemini-4-argon"},
}
DEFAULT_MODELS = {provider: tiers["best"] for provider, tiers in MODEL_TIERS.items()}
# Environment variables that may hold a provider key for first-time setup.
PROVIDER_KEY_ENVS = {
    "anthropic": ("ANTHROPIC_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    "google": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
}
_MISSING_KEY = re.compile(r'No API key stored for provider "([^"]+)"')


class RelayError(RuntimeError):
    """A relay request failed; ``status`` is the HTTP status when known."""

    def __init__(self, message: str, status: Optional[int] = None):
        super().__init__(message)
        self.status = status


class RelayTimeout(RelayError):
    """The relay or provider did not answer in time."""


class RelayAccountError(RelayError):
    """The relay token or provider account was refused (401/402/403).

    These apply to every request (expired token, invalid key, exhausted
    billing), so callers should stop rather than retry per file.
    """


class MissingProviderKey(RelayError):
    """The relay token has no stored key for ``provider``."""

    def __init__(self, provider: str):
        super().__init__(f"no {provider} key is stored on the relay", status=400)
        self.provider = provider


def provider_for_model(model: str) -> str:
    """Return the relay provider for a model name, or raise ValueError."""
    if "/" in model:
        provider = model.split("/", 1)[0].lower()
        if provider in SUPPORTED_PROVIDERS:
            return provider
    for pattern, provider in _PROVIDER_PATTERNS:
        if pattern.search(model):
            return provider
    raise ValueError(
        f"cannot tell which provider serves {model!r}; use claude, gpt, or "
        "gemini for the provider's default model, a claude-*, gpt-*, or gemini-* "
        "model id, or prefix it as anthropic/, openai/, or google/"
    )


def resolve_model(value: str) -> Tuple[str, str]:
    """Return ``(provider, model)`` for ``provider``, ``provider:tier``, or a model id."""
    value = value.strip()
    name, separator, tier = value.lower().partition(":")
    provider = PROVIDER_ALIASES.get(name)
    # Only a bare provider selects the flagship: an explicit but empty tier
    # (e.g. an unset ``$TIER`` in ``claude:$TIER``) is an error, not Opus.
    if provider is not None and not separator:
        return provider, DEFAULT_MODELS[provider]
    if provider is not None:
        tiers = MODEL_TIERS[provider]
        if tier not in tiers:
            raise ValueError(
                "unknown tier {!r} for {}; use one of: {}".format(
                    tier, name, ", ".join(sorted(tiers))
                )
            )
        return provider, tiers[tier]
    return provider_for_model(value), value


def trusted_env(name: str) -> Optional[str]:
    """A variable from the process environment, never from a loaded .env.

    ``wild`` auto-loads ``.env`` from the current directory, which may be an
    untrusted cloned repository. Letting it choose the relay endpoint, the
    relay token, or a provider key would let a repository redirect the user's
    credentials and code, so only the pre-existing environment counts.
    """
    from diffgraph.env_loader import DOTENV_INJECTED

    if name in DOTENV_INJECTED:
        return None
    value = (os.environ.get(name) or "").strip()
    return value or None


@dataclass(frozen=True)
class KeyCandidate:
    """A provider key found in the environment, with where it came from."""

    variable: str
    key: str
    source: str

    @property
    def label(self) -> str:
        return f"{self.variable} from {self.source} (ends …{self.key[-4:]})"


def provider_key_candidates(provider: str) -> List[KeyCandidate]:
    """Return every distinct key for ``provider`` in the trusted environment.

    Keys from a ``.env`` file are ignored (see :func:`trusted_env`); several
    variables (e.g. ``GEMINI_API_KEY`` and ``GOOGLE_API_KEY``) may hold
    different keys, and all are returned so the caller never silently picks one.
    """
    candidates: List[KeyCandidate] = []
    seen = set()
    for variable in PROVIDER_KEY_ENVS.get(provider, ()):
        key = trusted_env(variable)
        if key and key not in seen:
            seen.add(key)
            candidates.append(KeyCandidate(variable, key, "the shell environment"))
    return candidates


def relay_model(model: str) -> str:
    """Return ``provider/model`` so the relay never has to guess the route.

    The relay strips the prefix before forwarding, and an explicit prefix
    also covers names its own patterns miss (e.g. a bare ``o3``).
    """
    if "/" in model and model.split("/", 1)[0].lower() in SUPPORTED_PROVIDERS:
        return model
    return f"{provider_for_model(model)}/{model}"


def _urllib_transport(
    method: str,
    url: str,
    headers: Mapping[str, str],
    body: Optional[bytes],
    timeout: float,
) -> Tuple[int, Mapping[str, str], bytes]:
    request = urllib.request.Request(url, data=body, headers=dict(headers), method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, dict(response.headers), response.read()
    except urllib.error.HTTPError as error:
        return error.code, dict(error.headers or {}), error.read()
    except (socket.timeout, TimeoutError) as error:
        raise RelayTimeout(f"relay request timed out: {error}") from error
    except urllib.error.URLError as error:
        if isinstance(error.reason, (socket.timeout, TimeoutError)):
            raise RelayTimeout(f"relay request timed out: {error.reason}") from error
        raise RelayError(f"could not reach the relay: {error.reason}") from error


class RelayClient:
    """Talk to one BYOK Relay instance with one relay token."""

    def __init__(
        self,
        base_url: str = DEFAULT_RELAY_URL,
        token: Optional[str] = None,
        *,
        timeout: float = 120.0,
        max_retries: int = 3,
        transport: Optional[Transport] = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout
        self.max_retries = max_retries
        self._transport = transport or _urllib_transport
        self._sleep = sleep

    def _request(self, method: str, path: str, payload: Optional[dict] = None) -> dict:
        headers = {
            "Accept": "application/json",
            "User-Agent": f"wild/{__version__}",
        }
        body = None
        if payload is not None:
            headers["Content-Type"] = "application/json"
            body = json.dumps(payload).encode("utf-8")
        if self.token:
            headers["x-relay-token"] = self.token
        url = self.base_url + path
        for attempt in range(self.max_retries + 1):
            status, response_headers, raw = self._transport(
                method, url, headers, body, self.timeout
            )
            if status == 429 and attempt < self.max_retries:
                self._sleep(_retry_after(response_headers, attempt))
                continue
            break
        try:
            data = json.loads(raw.decode("utf-8")) if raw else {}
        except (UnicodeDecodeError, json.JSONDecodeError):
            data = {"error": raw[:200].decode("utf-8", "replace")}
        if 200 <= status < 300:
            return data if isinstance(data, dict) else {}
        message = _error_message(data) or f"HTTP {status}"
        missing = _MISSING_KEY.search(message)
        if missing:
            raise MissingProviderKey(missing.group(1))
        if status in (408, 504):
            raise RelayTimeout(f"relay request timed out (HTTP {status}): {message}", status)
        if status in (401, 402, 403):
            raise RelayAccountError(f"relay returned HTTP {status}: {message.strip()}", status)
        raise RelayError(f"relay returned HTTP {status}: {message}", status)

    def register(self, app_id: str = APP_ID) -> str:
        """Create a relay user and return (and adopt) its relay token."""
        data = self._request("POST", "/users", {"app_id": app_id})
        token = data.get("token")
        if not isinstance(token, str) or not token:
            raise RelayError("relay registration returned no token")
        self.token = token
        return token

    def store_key(self, provider: str, key: str) -> None:
        """Send a provider key to the relay once; it is never returned."""
        self._request("POST", f"/keys/{provider}", {"key": key})

    def chat(self, model: str, messages: List[Dict[str, str]], max_tokens: int = 2048) -> str:
        """Return the assistant text for one non-streaming chat completion."""
        data = self._request(
            "POST",
            "/relay/v1/chat/completions",
            {
                "model": relay_model(model),
                "messages": messages,
                "max_tokens": max_tokens,
                # No temperature: newer models (e.g. claude-opus-5-5) reject it.
                "stream": False,
            },
        )
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise RelayError("relay response had no completion text") from error
        if isinstance(content, list):  # content parts
            content = "".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        if not isinstance(content, str):
            raise RelayError("relay response had no completion text")
        return content


def _retry_after(headers: Mapping[str, str], attempt: int) -> float:
    lowered = {key.lower(): value for key, value in headers.items()}
    try:
        return min(60.0, max(0.0, float(lowered.get("retry-after", ""))))
    except ValueError:
        return min(60.0, 2.0 ** (attempt + 1))


def _error_message(data: object) -> str:
    if isinstance(data, dict):
        error = data.get("error")
        if isinstance(error, dict):
            return str(error.get("message") or "")
        if error:
            return str(error)
    return ""


# ---------------------------------------------------------------------------
# Relay token storage (never provider keys)
# ---------------------------------------------------------------------------

def config_path() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(Path.home(), ".config")
    return Path(base) / "wild" / "byok-relay.json"


def _user_config() -> dict:
    try:
        data = json.loads(config_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def relay_url() -> str:
    """The relay endpoint: trusted environment, then user config, else the default.

    A self-hosted relay is accepted only from these trusted channels, never
    from a repository's ``.env``.
    """
    configured = _user_config().get("relay_url")
    url = trusted_env(RELAY_URL_ENV) or (configured if isinstance(configured, str) else None)
    return (url or DEFAULT_RELAY_URL).rstrip("/")


def load_token(base_url: str) -> Optional[str]:
    """Return the relay token from the trusted environment or the per-URL store."""
    token = trusted_env(RELAY_TOKEN_ENV)
    if token:
        return token
    data = _user_config()
    entry = data.get("relays", {}).get(base_url.rstrip("/")) if isinstance(data, dict) else None
    token = entry.get("token") if isinstance(entry, dict) else None
    return token if isinstance(token, str) and token else None


def save_token(base_url: str, token: str) -> Path:
    """Store a relay token per relay URL in a user-only (0600) file."""
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            data = {}
    except (OSError, ValueError):
        data = {}
    data.setdefault("relays", {})[base_url.rstrip("/")] = {
        "token": token,
        "app_id": APP_ID,
    }
    temporary = path.with_name(path.name + ".tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)
    return path
