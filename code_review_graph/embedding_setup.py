"""Configure embeddings against an OpenAI-compatible endpoint, unattended.

Exists so a fresh machine needs one command rather than a remembered sequence.
Detects which embedding models a local Ollama actually has, picks one, works out
a sensible vector width by probing the endpoint, writes ``embeddings.toml``, and
verifies the result — reporting each decision so nothing is silently assumed.

Network access uses stdlib ``urllib`` to match the OpenAI-compatible provider:
configuring embeddings must not itself require an optional dependency.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from .constants import crg_home

#: Default endpoint. Ollama's OpenAI-compatible surface on the standard port.
DEFAULT_BASE_URL = "http://127.0.0.1:11434/v1"

#: Ollama ignores the value but the OpenAI schema requires the header, and the
#: provider refuses to start without something non-empty.
LOCAL_PLACEHOLDER_API_KEY = "ollama"

#: Preference order when the caller does not name a model. Larger Qwen variants
#: first for retrieval quality, then the small models that suit a weak GPU or
#: CPU-only machine, then older general-purpose encoders.
MODEL_PREFERENCE = (
    "qwen3-embedding:8b",
    "qwen3-embedding:4b",
    "qwen3-embedding",
    "qwen3-embedding:0.6b",
    "embeddinggemma",
    "mxbai-embed-large",
    "nomic-embed-text-v2-moe",
    "nomic-embed-text",
    "all-minilm",
)

#: Target vector width. Search scans every stored vector, so cost scales with
#: this: 512 keeps a large graph responsive and stays well inside the quality of
#: a Matryoshka-capable model. Models that cannot truncate keep their native
#: width instead.
DEFAULT_DIMENSION = 512

#: Suggested when the endpoint has no embedding model at all. Small enough for a
#: weak GPU or CPU-only machine.
FALLBACK_PULL_SUGGESTION = "embeddinggemma"

_TIMEOUT_SECONDS = 60


class SetupError(RuntimeError):
    """Configuration could not be completed. Message is user-facing."""


def _api_root(base_url: str) -> str:
    """Ollama's native API root for an OpenAI-compatible base URL."""
    trimmed = base_url.rstrip("/")
    if trimmed.endswith("/v1"):
        return trimmed[: -len("/v1")]
    return trimmed


def _post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as response:  # nosec B310
        return json.loads(response.read().decode("utf-8"))


def _get_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as response:  # nosec B310
        return json.loads(response.read().decode("utf-8"))


def discover_embedding_models(
    base_url: str,
    *,
    get_json: Callable[[str], dict[str, Any]] | None = None,
    post_json: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
) -> list[str]:
    """Return installed models that report the ``embedding`` capability.

    Asking is necessary: ``/api/tags`` does not distinguish embedding models
    from chat models, and naming conventions do not either — ``qwen3-embedding``
    also advertises ``tools``, while a coder model advertises ``completion``.
    """
    get_json = get_json or _get_json
    post_json = post_json or _post_json
    root = _api_root(base_url)

    try:
        tags = get_json(f"{root}/api/tags")
    except (urllib.error.URLError, OSError) as exc:
        raise SetupError(
            f"No endpoint answered at {root}. Is Ollama running? ({exc})"
        ) from exc

    embedding_models = []
    for entry in tags.get("models", []):
        name = entry.get("model") or entry.get("name")
        if not name:
            continue
        try:
            shown = post_json(f"{root}/api/show", {"model": name})
        except Exception:
            # A model we cannot interrogate is simply not offered.
            continue
        if "embedding" in (shown.get("capabilities") or []):
            embedding_models.append(name)
    return embedding_models


def select_model(installed: Sequence[str], requested: str | None = None) -> str:
    """Choose which embedding model to configure.

    An explicit request wins even when the endpoint does not list it, so a model
    pulled moments later, or served by something other than Ollama, is usable.
    """
    if requested:
        return requested
    if not installed:
        raise SetupError(
            "The endpoint has no embedding model installed. Pull one first, "
            f"e.g. `ollama pull {FALLBACK_PULL_SUGGESTION}`."
        )
    for preferred in MODEL_PREFERENCE:
        for candidate in installed:
            if candidate == preferred or candidate.startswith(f"{preferred}:"):
                return candidate
    return installed[0]


def probe_dimensions(
    base_url: str,
    model: str,
    api_key: str,
    requested_dimension: int | None,
    *,
    post_json: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
) -> tuple[int, int, bool]:
    """Return ``(native, chosen, truncation_honoured)`` for the model.

    The endpoint is asked twice rather than trusted: once without a dimension to
    learn the native width, and once with the target to confirm the model really
    truncates. Recording a width the endpoint ignores would mean every stored
    vector silently had a different shape from the configuration.
    """
    post_json = post_json or _post_json
    url = f"{base_url.rstrip('/')}/embeddings"
    probe_text = "def resolve(case_no): return None"

    def _embed(dimension: int | None) -> int:
        body: dict[str, Any] = {"model": model, "input": probe_text}
        if dimension is not None:
            body["dimensions"] = dimension
        try:
            response = post_json(url, body)
        except (urllib.error.URLError, OSError) as exc:
            raise SetupError(f"Endpoint did not answer for model {model}: {exc}") from exc
        data = response.get("data") or []
        if not data or "embedding" not in data[0]:
            raise SetupError(f"Endpoint returned no embedding for model {model}.")
        return len(data[0]["embedding"])

    native = _embed(None)
    target = requested_dimension if requested_dimension else min(DEFAULT_DIMENSION, native)
    if target >= native:
        return native, native, True
    reduced = _embed(target)
    return native, (reduced if reduced == target else native), reduced == target


def render_config(
    *,
    provider: str,
    model: str,
    base_url: str,
    api_key: str,
    dimension: int | None,
) -> str:
    """Render the TOML written to the state directory."""
    lines = [
        "# Written by `code-review-graph setup-embeddings`.",
        "# Environment variables override every value here.",
        f'provider = "{provider}"',
        f'model = "{model}"',
        f'base_url = "{base_url}"',
        f'api_key = "{api_key}"',
    ]
    if dimension is not None:
        lines.append(f"dimension = {dimension}")
    return "\n".join(lines) + "\n"


def config_destination() -> Path:
    """Where the settings file is written."""
    return crg_home() / "embeddings.toml"


def setup_embeddings(
    *,
    base_url: str = DEFAULT_BASE_URL,
    model: str | None = None,
    dimension: int | None = None,
    api_key: str | None = None,
    provider: str = "openai",
    force: bool = False,
    write: bool = True,
) -> dict[str, Any]:
    """Detect, choose, verify and record embedding settings.

    Returns a summary describing every decision, so a caller (or a log) can see
    what was chosen and why rather than only that it succeeded.
    """
    # Checked before any network work: refusing to overwrite after several
    # seconds of probing wastes the caller's time and tells them nothing they
    # could not have been told immediately.
    destination = config_destination()
    if write and destination.exists() and not force:
        raise SetupError(
            f"{destination} already exists. Re-run with --force to replace it."
        )

    resolved_key = api_key or LOCAL_PLACEHOLDER_API_KEY
    installed = discover_embedding_models(base_url) if not model else []
    chosen_model = select_model(installed, model)
    native, chosen_dimension, truncated = probe_dimensions(
        base_url, chosen_model, resolved_key, dimension
    )

    written = False
    if write:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            render_config(
                provider=provider,
                model=chosen_model,
                base_url=base_url,
                api_key=resolved_key,
                dimension=chosen_dimension if chosen_dimension != native else None,
            ),
            encoding="utf-8",
        )
        written = True

    return {
        "base_url": base_url,
        "provider": provider,
        "model": chosen_model,
        "installed_embedding_models": installed,
        "native_dimension": native,
        "dimension": chosen_dimension,
        "truncation_honoured": truncated,
        "config_path": str(destination),
        "written": written,
    }
