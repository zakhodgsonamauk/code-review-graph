"""Vector embedding support for semantic code search.

Supports multiple providers:
1. Local (sentence-transformers) - Private, fast, offline.
2. Google Gemini - High-quality, cloud-based. Requires explicit opt-in.
3. MiniMax (embo-01) - High-quality 1536-dim cloud embeddings. Requires MINIMAX_API_KEY.
4. OpenAI-compatible - Any endpoint speaking OpenAI /v1/embeddings (real OpenAI,
   Azure OpenAI, self-hosted gateways like new-api / LiteLLM / vLLM / LocalAI / Ollama).
"""

from __future__ import annotations

import errno
import functools
import hashlib
import logging
import os
import re
import sqlite3
import struct
import sys
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar
from urllib.parse import urlparse

from . import __version__ as _crg_version
from .graph import GraphNode, GraphStore, node_to_dict

try:
    import numpy as _np
except ImportError:
    # NumPy only ships in the optional extras. Semantic search stays correct
    # without it by falling back to the pure-Python cosine loop.
    _np = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# Rows pulled per fetchmany() while scanning the embeddings table.
_SEARCH_CHUNK_SIZE = 500

# struct-packed vectors are float32; used to validate blob length against the
# query dimension before reinterpreting bytes as a matrix.
_FLOAT32_BYTES = 4

# Sent on every cloud-provider HTTP request. Some providers (e.g. Fireworks)
# sit behind Cloudflare and reject the urllib default ``Python-urllib/X.Y``
# UA with HTTP 403 / error 1010 ("browser signature banned"). A real UA gets
# us through and gives upstream a way to identify CRG-driven traffic.
_USER_AGENT = (
    f"code-review-graph/{_crg_version} "
    "(+https://github.com/tirth8205/code-review-graph)"
)

# Retry budgets. Ordinary transient failures (429, 5xx, dropped sockets) clear
# quickly; host resource exhaustion needs long enough for the OS to reclaim
# what ran out, which is far slower than an exponential-from-1s schedule.
_RETRY_ATTEMPTS = 3
_RETRY_BACKOFF_BASE_SECONDS = 2
_RESOURCE_EXHAUSTION_ATTEMPTS = 5
_RESOURCE_EXHAUSTION_BACKOFF_SECONDS = 30
_MAX_RETRY_ATTEMPTS = max(_RETRY_ATTEMPTS, _RESOURCE_EXHAUSTION_ATTEMPTS)

# Some endpoints report a *server-side* transient shortage using a client-error
# status, which makes status code alone the wrong signal. Ollama proxies to an
# internal runner and returns HTTP 400 whose body is a Go dial error when the
# host runs out of ephemeral sockets:
#
#   OpenAI API HTTP 400: Post "http://127.0.0.1:54310/tokenize": dial tcp ...
#   bind: An operation on a socket could not be performed because the system
#   lacked sufficient buffer space or because a queue was full.
#
# Retrying 4xx as a class would turn genuine bad requests (invalid model, batch
# too large) into slow failures, so match exhaustion signatures explicitly.
_RESOURCE_EXHAUSTION_RE = re.compile(
    r"queue was full"
    r"|buffer space"
    r"|\bbind:"
    r"|dial tcp"
    r"|address already in use"
    r"|cannot assign requested address"
    r"|resource temporarily unavailable"
    r"|too many open files"
    r"|connection reset by peer",
    re.IGNORECASE,
)


_T = TypeVar("_T")

# The graph database often lives inside a synced folder (Dropbox, OneDrive,
# iCloud). Those clients open files to upload them, which on Windows surfaces
# as a short-lived EPERM/EACCES or a SQLite lock. A long embedding pass writes
# repeatedly over many minutes, so it is very likely to collide with a sync at
# some point -- and without a retry a single collision aborts the whole run.
_DB_LOCK_RETRY_ATTEMPTS = 5
_DB_LOCK_BACKOFF_SECONDS = 0.5
_DB_LOCK_MESSAGE_RE = re.compile(
    r"database is locked"
    r"|database table is locked"
    r"|unable to open database file"
    r"|attempt to write a readonly database"
    r"|disk i/o error",
    re.IGNORECASE,
)


def _is_transient_db_error(exc: BaseException) -> bool:
    """True for file-lock style failures a sync client causes transiently.

    Deliberately narrow: a genuinely corrupt database or a missing parent
    directory must fail immediately rather than after five sleeps.
    """
    if isinstance(exc, OSError):
        return isinstance(exc, PermissionError) or exc.errno in (
            errno.EACCES,
            errno.EPERM,
        )
    if isinstance(exc, sqlite3.Error):
        return bool(_DB_LOCK_MESSAGE_RE.search(str(exc)))
    return False


def _with_db_retry(operation: Callable[[], _T], *, what: str) -> _T:
    """Run a database operation, retrying transient file-lock failures."""
    for attempt in range(_DB_LOCK_RETRY_ATTEMPTS):
        try:
            return operation()
        except Exception as exc:
            if (
                not _is_transient_db_error(exc)
                or attempt == _DB_LOCK_RETRY_ATTEMPTS - 1
            ):
                raise
            wait = _DB_LOCK_BACKOFF_SECONDS * (2 ** attempt)
            logger.warning(
                "%s blocked by a file lock (attempt %d/%d), retrying in %.1fs: %s",
                what, attempt + 1, _DB_LOCK_RETRY_ATTEMPTS, wait, exc,
            )
            time.sleep(wait)
    raise AssertionError("unreachable")  # pragma: no cover


#: Node kinds embedded by default.
#:
#: Tests are excluded. On a real Django/React graph they were 7,883 of 19,595
#: vectors -- 40% of the index and 40% of every scan -- and crowded source out
#: of results: a query about promoting uploads to permanent storage returned
#: mostly integration tests. Coverage questions are answered better by the
#: TESTED_BY edges the graph already stores (39,717 of them), which are exact
#: rather than approximate. File nodes have never been embedded; their text is
#: a path.
DEFAULT_EMBEDDED_KINDS = frozenset({"Class", "Function", "Type"})
_DEFAULT_EMBEDDED_KINDS = DEFAULT_EMBEDDED_KINDS  # backwards-compatible alias
_EMBED_KINDS_ENV = "CRG_EMBED_KINDS"


def embedded_node_kinds() -> frozenset[str]:
    """Node kinds to embed.

    Override with ``CRG_EMBED_KINDS`` as a comma-separated list, e.g.
    ``Class,Function,Type,Test`` to restore test embedding. An empty or
    all-whitespace value falls back to the default rather than embedding
    nothing, which would silently disable semantic search.
    """
    raw = os.environ.get(_EMBED_KINDS_ENV, "").strip()
    if not raw:
        return _DEFAULT_EMBEDDED_KINDS
    kinds = frozenset(part.strip() for part in raw.split(",") if part.strip())
    return kinds or _DEFAULT_EMBEDDED_KINDS


#: Optional settings file under the per-user state directory. Exists because
#: environment variables are not reliably present in every process that needs
#: to embed: git hooks inherit the committing shell, and an MCP server started
#: before the variables were exported keeps the stale environment for its whole
#: life. Both then silently skip embedding refresh while the graph moves on.
_EMBEDDING_CONFIG_FILENAME = "embeddings.toml"

#: TOML key -> the environment variable that overrides it.
_EMBEDDING_CONFIG_KEYS = (
    "provider",
    "model",
    "base_url",
    "api_key",
    "dimension",
    "batch_size",
)


def embedding_config_path() -> Path:
    """Path to the optional embedding settings file."""
    from .constants import crg_home

    return crg_home() / _EMBEDDING_CONFIG_FILENAME


def load_embedding_config() -> dict[str, str]:
    """Load embedding settings from the per-user state directory.

    Returns an empty mapping when the file is absent or unreadable -- the file
    is an optional convenience, never a requirement, so a malformed one must
    not break callers that are configured entirely through the environment. It
    is logged at warning level rather than swallowed silently.

    Values are stringified so callers can treat them exactly like environment
    variables. This file may hold an API key, so it is read and never written
    by this module: creating it, and choosing its permissions, stays with the
    user.
    """
    path = embedding_config_path()
    try:
        raw = path.read_bytes()
    except OSError:
        return {}

    try:
        if sys.version_info >= (3, 11):
            import tomllib
        else:  # pragma: no cover - exercised on older interpreters only
            import tomli as tomllib  # type: ignore[no-redef]
    except ImportError:  # pragma: no cover - tomli is a declared dependency
        logger.warning("Cannot read %s: no TOML parser available", path)
        return {}

    try:
        parsed = tomllib.loads(raw.decode("utf-8", errors="replace"))
    except tomllib.TOMLDecodeError as exc:
        logger.warning("Ignoring malformed %s: %s", path, exc)
        return {}

    section = parsed.get("embeddings", parsed)
    if not isinstance(section, dict):
        return {}
    return {
        key: str(section[key]).strip()
        for key in _EMBEDDING_CONFIG_KEYS
        if section.get(key) is not None and str(section[key]).strip()
    }


def _setting(env_var: str, config: dict[str, str], config_key: str) -> str | None:
    """Return a setting from the environment, else the config file."""
    value = os.environ.get(env_var, "").strip()
    if value:
        return value
    return config.get(config_key) or None


class TransientEmbeddingError(RuntimeError):
    """A provider error whose cause is expected to clear on its own.

    Carries the provider's original message so it stays as actionable as a
    permanent failure, while letting the retry loop wait rather than abort.
    """


def _embedding_error(message: str) -> RuntimeError:
    """Return a transient or permanent error for a provider message."""
    if _RESOURCE_EXHAUSTION_RE.search(message):
        return TransientEmbeddingError(message)
    return RuntimeError(message)

# ---------------------------------------------------------------------------
# Provider Interface and Implementations
# ---------------------------------------------------------------------------


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        pass

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a search query (may use a different task type than indexing)."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass


LOCAL_DEFAULT_MODEL = "all-MiniLM-L6-v2"


# Process-wide cache and initialization lock for sentence-transformer models.
# The dependency import itself touches process-global Torch state, so one lock
# must cover availability checks, imports, and model construction across every
# model name. A per-model lock would still allow two first imports to race.
# Populated by ``prewarm_local_embeddings()`` at server startup (see ``main.main``)
# and by ``LocalEmbeddingProvider._get_model`` on first lazy load. Sharing the
# loaded model across ``LocalEmbeddingProvider`` instances avoids re-importing
# ``sentence_transformers`` + ``torch`` from worker threads, which deadlocks
# ``semantic_search_nodes_tool`` on Windows stdio MCP (#385 fixed the peer
# tools via ``asyncio.to_thread``; this cache fixes the remaining case where
# torch DLL / OpenMP init runs inside an executor thread).
_MODEL_CACHE: dict[str, Any] = {}
_MODEL_INIT_LOCK = threading.RLock()


def prewarm_local_embeddings(model_name: str | None = None) -> None:
    """Eagerly load the local sentence-transformer model on the calling thread.

    Call this from the **main thread** before entering an asyncio event loop
    (e.g. before ``mcp.run()``) on Windows to prevent a deadlock where lazy-
    loading ``sentence_transformers`` + ``torch`` inside a FastMCP executor
    worker thread blocks indefinitely on DLL init / OpenMP thread-pool
    registration.

    No-op when ``sentence-transformers`` is not installed (cloud-provider
    setups remain unaffected) or when the configured model is already cached.

    Args:
        model_name: Optional override; falls back to the ``CRG_EMBEDDING_MODEL``
            environment variable and then to ``LOCAL_DEFAULT_MODEL``.
    """
    resolved = model_name or os.environ.get(
        "CRG_EMBEDDING_MODEL", LOCAL_DEFAULT_MODEL
    )
    try:
        LocalEmbeddingProvider(resolved)._get_model()
    except ImportError:
        return  # cloud-only setup: nothing to pre-warm
    except Exception as exc:  # pragma: no cover — best-effort startup hook
        logger.warning("prewarm_local_embeddings(%s) skipped: %s", resolved, exc)


class LocalEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name or os.environ.get(
            "CRG_EMBEDDING_MODEL", LOCAL_DEFAULT_MODEL
        )
        self._model = None  # Lazy-loaded

    def _get_model(self):
        if self._model is not None:
            return self._model

        # Fast path for a model fully published by another provider instance.
        cached = _MODEL_CACHE.get(self._model_name)
        if cached is not None:
            self._model = cached
            return self._model

        with _MODEL_INIT_LOCK:
            # A competing caller may have initialized this provider or cache
            # entry while we waited. Recheck both under the process-wide lock.
            if self._model is not None:
                return self._model
            cached = _MODEL_CACHE.get(self._model_name)
            if cached is not None:
                self._model = cached
                return self._model

            try:
                from sentence_transformers import SentenceTransformer
                # Check environment variable, default to False to prevent RCE
                _rce_val = os.environ.get("CRG_ALLOW_REMOTE_CODE", "0")
                allow_remote_code = _rce_val.lower() in ("1", "true", "yes")

                model = SentenceTransformer(
                    self._model_name,
                    trust_remote_code=allow_remote_code,
                )
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Run: pip install code-review-graph[embeddings]"
                )

            # Publish only a fully constructed model. Failed attempts leave
            # both the provider and shared cache empty so a waiter can retry.
            _MODEL_CACHE[self._model_name] = model
            self._model = model
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        model = self._get_model()
        vectors = model.encode(texts, show_progress_bar=False)
        return [v.tolist() for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]

    @property
    def dimension(self) -> int:
        model = self._get_model()
        if hasattr(model, "get_embedding_dimension"):
            return model.get_embedding_dimension()
        return model.get_sentence_embedding_dimension()

    @property
    def name(self) -> str:
        return f"local:{self._model_name}"


class GoogleEmbeddingProvider(EmbeddingProvider):
    def __init__(self, api_key: str, model: str = "gemini-embedding-001") -> None:
        try:
            from google import genai
            self._client = genai.Client(api_key=api_key)
            self.model = model
            self._dimension: int | None = None
        except ImportError:
            raise ImportError(
                "google-generativeai not installed. "
                "Run: pip install code-review-graph[google-embeddings]"
            )

    def embed(self, texts: list[str]) -> list[list[float]]:
        batch_size = 100
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self._call_with_retry(
                lambda b=batch: self._client.models.embed_content(
                    model=self.model,
                    contents=b,
                    config={"task_type": "RETRIEVAL_DOCUMENT"},
                )
            )
            results.extend([e.values for e in response.embeddings])
        if self._dimension is None and results:
            self._dimension = len(results[0])
        return results

    @staticmethod
    def _call_with_retry(fn, max_retries: int = 3):
        """Call fn with exponential backoff on transient API errors."""
        retryable_statuses = ("429", "500", "503")
        for attempt in range(max_retries):
            try:
                return fn()
            except Exception as e:
                # Retry on rate-limit (429) or server errors (5xx)
                err_str = str(e)
                is_retryable = any(status in err_str for status in retryable_statuses)
                if not is_retryable:
                    logger.debug(
                        "Non-retryable Gemini API error: %s",
                        type(e).__name__,
                    )
                    raise
                if attempt == max_retries - 1:
                    logger.error(
                        "Gemini API request failed after %d requests.",
                        max_retries,
                    )
                    raise

                wait = 2 ** attempt

                logger.warning(
                    "Gemini API retry %d/%d in %ds (%s): %s",
                    attempt + 1,
                    max_retries,
                    wait,
                    type(e).__name__,
                    e,
                )

                time.sleep(wait)

    def embed_query(self, text: str) -> list[float]:
        response = self._call_with_retry(
            lambda: self._client.models.embed_content(
                model=self.model,
                contents=[text],
                config={"task_type": "RETRIEVAL_QUERY"},
            )
        )
        vec = response.embeddings[0].values
        if self._dimension is None:
            self._dimension = len(vec)
        return vec

    @property
    def dimension(self) -> int:
        if self._dimension is not None:
            return self._dimension
        # Default for gemini-embedding-001; updated dynamically after first call
        return 768

    @property
    def name(self) -> str:
        return f"google:{self.model}"


class MiniMaxEmbeddingProvider(EmbeddingProvider):
    """MiniMax embo-01 embedding provider (1536 dimensions).

    Uses the MiniMax Embeddings API (https://api.minimax.io/v1/embeddings)
    with the embo-01 model. Requires the MINIMAX_API_KEY environment variable.
    """

    _ENDPOINT = "https://api.minimax.io/v1/embeddings"
    _MODEL = "embo-01"
    _DIMENSION = 1536

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    def _call_api(self, texts: list[str], task_type: str) -> list[list[float]]:
        import json as _json
        import urllib.request

        payload = _json.dumps({
            "model": self._MODEL,
            "texts": texts,
            "type": task_type,
        }).encode("utf-8")

        req = urllib.request.Request(
            self._ENDPOINT,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
                "User-Agent": _USER_AGENT,
                "Accept": "application/json",
            },
        )

        max_retries = 3
        for attempt in range(max_retries):
            try:
                import ssl
                _ssl_ctx = ssl.create_default_context()
                with urllib.request.urlopen(req, timeout=60, context=_ssl_ctx) as resp:  # nosec B310
                    body = _json.loads(resp.read().decode("utf-8"))

                base_resp = body.get("base_resp", {})
                if base_resp.get("status_code", 0) != 0:
                    raise RuntimeError(
                        f"MiniMax API error: {base_resp.get('status_msg', 'unknown')}"
                    )

                return body["vectors"]
            except Exception as e:
                err_str = str(e)
                is_retryable = "429" in err_str or "500" in err_str or "503" in err_str
                if not is_retryable or attempt == max_retries - 1:
                    raise
                wait = 2 ** attempt
                logger.warning(
                    "MiniMax API error (attempt %d/%d), retrying in %ds: %s",
                    attempt + 1, max_retries, wait, e,
                )
                time.sleep(wait)

        return []  # unreachable, but keeps mypy happy

    def embed(self, texts: list[str]) -> list[list[float]]:
        batch_size = 100
        results: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            results.extend(self._call_api(batch, "db"))
        return results

    def embed_query(self, text: str) -> list[float]:
        return self._call_api([text], "query")[0]

    @property
    def dimension(self) -> int:
        return self._DIMENSION

    @property
    def name(self) -> str:
        return f"minimax:{self._MODEL}"


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI-compatible embedding provider.

    Works with any endpoint that speaks the OpenAI ``/v1/embeddings`` schema:
    - Real OpenAI API (``https://api.openai.com/v1``)
    - Azure OpenAI
    - Self-hosted gateways: new-api, LiteLLM, vLLM, LocalAI, Ollama (openai mode)

    Provider identity in ``name`` includes both the model and the endpoint
    host (``openai:{model}@{host}``), so switching base URL while keeping the
    same model ID re-partitions the embeddings table and forces a clean
    re-embed. This is the only defense against silently mixing vector spaces
    from different backends (e.g. real OpenAI vs. an OpenAI-compatible
    gateway that ships different weights under the same model name).

    When no dimension is explicitly requested, it is detected from the first
    response and retained as local metadata. Switching the ``model`` in the
    environment also changes ``provider.name`` and triggers re-embed via the
    same isolation key.
    """

    _DEFAULT_BATCH_SIZE = 100

    # Default ports by scheme; stripped from the host_key so the user can't
    # accidentally force a re-embed by toggling an explicit default port.
    _DEFAULT_PORTS = {"http": 80, "https": 443}

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        dimension: int | None = None,
        timeout: int = 120,
        batch_size: int | None = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._requested_dimension = dimension
        self._dimension = dimension
        self._timeout = timeout
        self._batch_size = batch_size or self._DEFAULT_BATCH_SIZE
        self._host_key = self._make_host_key(self._base_url)

    @classmethod
    def _make_host_key(cls, base_url: str) -> str:
        """Normalize the identity key used in ``provider.name``.

        Codex review pushed this well past naive ``netloc`` because that
        alone has three leaks:

        1. ``netloc`` preserves ``userinfo`` (``user:pass@host``) — we'd
           persist credentials into the DB's ``embeddings.provider`` column.
           Use ``hostname`` instead.
        2. Default ports (``:80`` for http, ``:443`` for https) are
           semantically identical to omitting the port; keeping them would
           cause spurious re-embeds when the user just spelled the URL
           differently.
        3. Path is part of the backend identity for path-routed gateways:
           ``https://gw/openai/v1`` and ``https://gw/vendor-b/v1`` front
           different models and must not share cached vectors.
        """
        parsed = urlparse(base_url)
        hostname = (parsed.hostname or "").lower()
        scheme = (parsed.scheme or "").lower()
        port = parsed.port
        if port and port != cls._DEFAULT_PORTS.get(scheme):
            # Bracket IPv6 literals when appending a port.
            host_part = f"[{hostname}]:{port}" if ":" in hostname else f"{hostname}:{port}"
        else:
            host_part = hostname
        # Preserve path routing. Trim any trailing slash and any
        # ``/embeddings`` suffix that callers may have included — we append
        # that ourselves when building the request URL.
        path = (parsed.path or "").rstrip("/")
        if path.endswith("/embeddings"):
            path = path[: -len("/embeddings")].rstrip("/")
        # Include scheme: http and https to the same host+path front
        # different endpoints in practice (plaintext vs TLS, dev vs prod
        # gateway), and sharing cached vectors across them is the same
        # silent-mixing failure mode as switching base URL entirely.
        return f"{scheme}://{host_part}{path}" if path else f"{scheme}://{host_part}"

    def _call_api(self, texts: list[str]) -> list[list[float]]:
        import http.client
        import json as _json
        import socket
        import ssl
        import urllib.error
        import urllib.request

        body: dict[str, Any] = {"model": self._model, "input": texts}
        # Forward only a dimension explicitly requested by the user. The
        # model name may be an Azure deployment or gateway alias, so it cannot
        # tell us whether the endpoint accepts dimension reduction. A dimension
        # learned from a response is local metadata and must never leak into a
        # later request.
        if self._requested_dimension is not None:
            body["dimensions"] = self._requested_dimension

        payload = _json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            f"{self._base_url}/embeddings",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
                "User-Agent": _USER_AGENT,
                "Accept": "application/json",
            },
        )

        for attempt in range(_MAX_RETRY_ATTEMPTS):
            try:
                _ssl_ctx = ssl.create_default_context()
                try:
                    with urllib.request.urlopen(  # nosec B310
                        req, timeout=self._timeout, context=_ssl_ctx,
                    ) as resp:
                        raw = resp.read().decode("utf-8")
                except urllib.error.HTTPError as http_err:
                    # 429 / 5xx: re-raise and let the outer retry loop handle it.
                    # (We must not convert to RuntimeError here or retry below
                    # can't tell it was a transient HTTP failure.)
                    if http_err.code == 429 or 500 <= http_err.code < 600:
                        raise
                    # Other 4xx: surface the API error body instead of a bare
                    # "400 Bad Request" — gateways like new-api return JSON
                    # with the real reason (batch size limits, invalid model,
                    # etc.) which is far more actionable.
                    try:
                        err_body = http_err.read().decode("utf-8", errors="replace")
                    except Exception:
                        err_body = ""
                    err_msg = err_body or str(http_err)
                    try:
                        parsed = _json.loads(err_body)
                        if isinstance(parsed, dict) and "error" in parsed:
                            err_obj = parsed["error"]
                            err_msg = (
                                err_obj.get("message", err_msg)
                                if isinstance(err_obj, dict) else str(err_obj)
                            )
                    except Exception:  # nosec B110
                        # Non-JSON error body is fine: we already seeded
                        # err_msg with the raw body above, so fall through.
                        pass
                    raise _embedding_error(
                        f"OpenAI API HTTP {http_err.code}: {err_msg}"
                    ) from http_err

                response = _json.loads(raw)

                if "error" in response:
                    err = response["error"]
                    msg = err.get("message", "unknown") if isinstance(err, dict) else str(err)
                    # A 200 body can carry the same exhaustion text as a 4xx.
                    raise _embedding_error(f"OpenAI API error: {msg}")

                data = response.get("data", [])
                if not data:
                    raise RuntimeError("OpenAI API returned empty data")
                # OpenAI spec: data[i].index maps to input[i], but some
                # compatible gateways re-order results or drop entries on
                # partial failure, and others omit `index` entirely. Three
                # disjoint cases:
                #   1. All items have a valid int ``index``: must form a
                #      permutation of 0..N-1, then sort and use.
                #   2. NO item carries an ``index`` field: trust server
                #      order, only verify count matches.
                #   3. Anything in between (partial indices, str indices,
                #      missing on some): refuse. Zipping server order in
                #      that case would happily misalign the indexed items.
                any_has_index = any("index" in item for item in data)
                all_int_index = all(
                    isinstance(item.get("index"), int) for item in data
                )
                if all_int_index:
                    expected = set(range(len(texts)))
                    indices = [int(item["index"]) for item in data]
                    if len(set(indices)) != len(indices) or set(indices) != expected:
                        raise RuntimeError(
                            "OpenAI API returned malformed indices "
                            f"(got {indices}, expected permutation of "
                            f"0..{len(texts) - 1}) — refusing to misalign vectors."
                        )
                    data = sorted(data, key=lambda item: int(item["index"]))
                elif not any_has_index:
                    if len(data) != len(texts):
                        raise RuntimeError(
                            f"OpenAI API returned {len(data)} embeddings for "
                            f"{len(texts)} inputs with no index field — "
                            "refusing to misalign vectors."
                        )
                else:
                    # Mixed: some items have index, others don't (or carry
                    # non-int index). Server order would silently misplace
                    # the indexed items, so we refuse.
                    raise RuntimeError(
                        "OpenAI API returned mixed indexed/unindexed data — "
                        "refusing to misalign vectors."
                    )

                vectors = [item["embedding"] for item in data]
                if vectors and self._dimension is None:
                    self._dimension = len(vectors[0])
                return vectors

            except Exception as e:
                # Retryable = HTTP 429/5xx, network/timeout/TLS issues.
                # Non-retryable = HTTP 4xx (other), malformed responses,
                # misaligned data length — those are caller-side bugs that
                # will keep failing on retry.
                is_retryable = False
                # Host resource exhaustion reported behind a client-error
                # status. Needs a longer pause and more attempts than an
                # ordinary hiccup, because the OS reclaims slowly.
                exhausted = isinstance(e, TransientEmbeddingError)
                if exhausted:
                    is_retryable = True
                elif isinstance(e, urllib.error.HTTPError):
                    is_retryable = e.code == 429 or 500 <= e.code < 600
                elif isinstance(e, (
                    urllib.error.URLError,
                    socket.timeout,
                    TimeoutError,
                    ConnectionError,
                    ssl.SSLError,
                    # Reverse proxies and edge gateways surface transient
                    # disconnects as these stdlib classes. Real incidents
                    # have been observed on Cloudflare-fronted endpoints
                    # and on LiteLLM when upstream providers hiccup.
                    http.client.IncompleteRead,
                    http.client.BadStatusLine,
                    http.client.RemoteDisconnected,
                )):
                    is_retryable = True
                budget = (
                    _RESOURCE_EXHAUSTION_ATTEMPTS if exhausted else _RETRY_ATTEMPTS
                )
                if not is_retryable or attempt >= budget - 1:
                    raise
                wait = (
                    _RESOURCE_EXHAUSTION_BACKOFF_SECONDS * (attempt + 1)
                    if exhausted
                    else _RETRY_BACKOFF_BASE_SECONDS ** attempt
                )
                logger.warning(
                    "OpenAI embeddings API error (attempt %d/%d), retrying in %ds: %s",
                    attempt + 1, budget, wait, e,
                )
                time.sleep(wait)

        return []  # unreachable

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        results: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            results.extend(self._call_api(texts[i:i + self._batch_size]))
        return results

    def embed_query(self, text: str) -> list[float]:
        return self._call_api([text])[0]

    @property
    def dimension(self) -> int:
        if self._dimension is not None:
            return self._dimension
        # Default for text-embedding-3-small; updated after first call.
        return 1536

    @property
    def name(self) -> str:
        # Endpoint-aware identity: model alone is NOT enough — two backends
        # can serve the same model ID with different weights or dimensions,
        # and re-using cached embeddings across them silently corrupts
        # semantic ranking. Including the host partitions the embeddings
        # table so switching CRG_OPENAI_BASE_URL triggers a safe re-embed.
        return f"openai:{self._model}@{self._host_key}"


CLOUD_PROVIDERS = {"google", "minimax", "openai"}


def _is_localhost_url(url: str) -> bool:
    """Return True if url points to a localhost host (never treat as cloud egress).

    Uses urlparse.hostname so we compare the actual host, not a substring
    match that could be fooled by e.g. ``https://my-openai.127.0.0.1.nip.io``.
    """
    try:
        host = (urlparse(url).hostname or "").lower()
    except Exception:
        return False
    # nosec B104: we're *matching* a URL hostname, not binding a listener.
    return host in {"127.0.0.1", "localhost", "0.0.0.0", "::1"}  # nosec B104


def _warn_cloud_egress(provider_name: str) -> None:
    """Print a stderr warning before a cloud embedding provider is used.

    The warning is suppressed when ``CRG_ACCEPT_CLOUD_EMBEDDINGS=1`` is
    set in the environment, so scripted / CI workloads can acknowledge
    once and move on. Use stderr (never stdin/input) to stay compatible
    with the MCP stdio transport — anything we write to stdout would
    corrupt the JSON-RPC stream. See: #174
    """
    if os.environ.get("CRG_ACCEPT_CLOUD_EMBEDDINGS", "").strip() == "1":
        return
    print(
        f"\n⚠️  code-review-graph: about to embed code via the '{provider_name}' "
        "cloud provider.\n"
        "    Your source code (function names, docstrings, file paths) will be "
        "sent to an external API.\n"
        "    This is necessary for semantic search with the cloud provider you "
        "selected.\n"
        "    To skip this warning in future runs, set "
        "CRG_ACCEPT_CLOUD_EMBEDDINGS=1 in your environment.\n"
        "    To stay fully offline, use the default 'local' provider instead "
        "(no API key needed).\n",
        file=sys.stderr,
    )


_VALID_PROVIDERS = {"local", "openai", "google", "minimax"}


def get_provider(
    provider: str | None = None,
    model: str | None = None,
) -> EmbeddingProvider | None:
    """Get an embedding provider by name.

    Args:
        provider: Provider name. One of "local", "google", "minimax", "openai",
                  or None. When omitted, configured OpenAI-compatible
                  credentials select OpenAI; otherwise the local provider is
                  used. Names are case-insensitive and surrounding whitespace
                  is ignored; unknown names raise ValueError instead of
                  silently falling back to the local provider.
                  Google requires GOOGLE_API_KEY env var and explicit opt-in.
                  MiniMax requires MINIMAX_API_KEY env var and explicit opt-in.
                  OpenAI requires CRG_OPENAI_API_KEY + CRG_OPENAI_BASE_URL +
                  CRG_OPENAI_MODEL env vars (or the ``model`` arg). The egress
                  warning is skipped when the base URL points to localhost.
                  Cloud providers emit a one-time stderr warning before use
                  unless ``CRG_ACCEPT_CLOUD_EMBEDDINGS=1`` is set. See: #174
        model: Model name/path to use. For local provider this is any
               sentence-transformers compatible model. Falls back to
               CRG_EMBEDDING_MODEL env var, then to all-MiniLM-L6-v2.
               For Google provider this is a Gemini model ID.
               For OpenAI provider this overrides CRG_OPENAI_MODEL.

    Raises:
        ValueError: If the provider name is not one of the known providers,
                    or if required environment variables are missing.
    """
    name = provider.strip().lower() if provider else ""
    if name and name not in _VALID_PROVIDERS:
        raise ValueError(
            f"Unknown embedding provider '{name}'. "
            "Valid: local, openai, google, minimax"
        )

    config = load_embedding_config()

    # When no explicit provider is given but OpenAI-compatible settings are
    # configured, default to the openai provider so MCP tool calls that omit
    # the optional `provider` parameter still use the configured backend
    # (#551). The config file counts as configuration too, so a git hook or an
    # MCP server started from a stale environment resolves the same backend.
    if provider is None:
        configured = config.get("provider", "").strip().lower()
        if configured and configured in _VALID_PROVIDERS:
            name = configured
        elif (
            _setting("CRG_OPENAI_API_KEY", config, "api_key")
            and _setting("CRG_OPENAI_BASE_URL", config, "base_url")
        ):
            name = "openai"

    if name == "openai":
        api_key = _setting("CRG_OPENAI_API_KEY", config, "api_key")
        base_url = _setting("CRG_OPENAI_BASE_URL", config, "base_url")
        resolved_model = model or _setting("CRG_OPENAI_MODEL", config, "model")
        if not api_key or not base_url or not resolved_model:
            missing = [
                name for name, val in [
                    ("CRG_OPENAI_API_KEY", api_key),
                    ("CRG_OPENAI_BASE_URL", base_url),
                    ("CRG_OPENAI_MODEL", resolved_model),
                ] if not val
            ]
            raise ValueError(
                "Missing required setting(s) for the OpenAI embedding "
                f"provider: {', '.join(missing)}. Set them in the environment "
                f"or in {embedding_config_path()}."
            )
        dim_env = _setting("CRG_OPENAI_DIMENSION", config, "dimension")
        dimension = int(dim_env) if dim_env else None
        batch_env = _setting("CRG_OPENAI_BATCH_SIZE", config, "batch_size")
        batch_size = int(batch_env) if batch_env else None
        if not _is_localhost_url(base_url):
            _warn_cloud_egress("openai")
        return OpenAIEmbeddingProvider(
            api_key=api_key,
            base_url=base_url,
            model=resolved_model,
            dimension=dimension,
            batch_size=batch_size,
        )

    if name == "minimax":
        api_key = os.environ.get("MINIMAX_API_KEY")
        if not api_key:
            raise ValueError(
                "MINIMAX_API_KEY environment variable is required for "
                "the MiniMax embedding provider."
            )
        _warn_cloud_egress("minimax")
        return MiniMaxEmbeddingProvider(api_key=api_key)

    if name == "google":
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY environment variable is required for "
                "the Google embedding provider."
            )
        _warn_cloud_egress("google")
        try:
            return GoogleEmbeddingProvider(
                api_key=api_key,
                **({"model": model} if model else {}),
            )
        except ImportError:
            return None

    # Default: local
    if not _check_available():
        return None
    try:
        return LocalEmbeddingProvider(model_name=model)
    except ImportError:
        return None


def _check_available() -> bool:
    """Check whether local embedding support is available."""
    with _MODEL_INIT_LOCK:
        try:
            import sentence_transformers  # noqa: F401
            return True
        except ImportError:
            return False


# ---------------------------------------------------------------------------
# SQLite vector storage
# ---------------------------------------------------------------------------

_EMBEDDINGS_SCHEMA = """
CREATE TABLE IF NOT EXISTS embeddings (
    qualified_name TEXT PRIMARY KEY,
    vector BLOB NOT NULL,
    text_hash TEXT NOT NULL,
    provider TEXT NOT NULL DEFAULT 'unknown'
);
"""


def _encode_vector(vec: list[float]) -> bytes:
    """Encode a float vector as a compact binary blob."""
    return struct.pack(f"{len(vec)}f", *vec)


def _decode_vector(blob: bytes) -> list[float]:
    """Decode a binary blob back to a float vector."""
    n = len(blob) // 4  # 4 bytes per float32
    return list(struct.unpack(f"{n}f", blob))


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


_IDENTIFIER_SPLIT_RE = re.compile(r"([a-z])([A-Z])|[_./\-]+")
# Kept equal to the parser's _MAX_DOCSTRING_CHARS. Applied defensively because
# an existing database may hold values written under an older, larger bound.
_MAX_EMBEDDED_DOCSTRING_CHARS = 1200


def _split_identifier(name: str) -> str:
    """Split snake_case / camelCase / PascalCase / dotted into space-separated words.

    Examples:
        get_route_handler -> "get route handler"
        APIRoute          -> "API Route"
        dispatch_request  -> "dispatch request"
        full_dispatch_request -> "full dispatch request"
    """
    if not name:
        return ""
    # Insert space between lowercase->uppercase transitions, then collapse
    # snake_case / dotted / hyphenated separators.
    spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    spaced = re.sub(r"[_./\-]+", " ", spaced)
    return " ".join(spaced.split())


def _node_to_text(node: GraphNode) -> str:
    """Convert a node to a searchable text representation.

    Designed so natural-language queries land on the right node, not just on
    the enclosing class. We include the dotted ``Parent.name`` form, the
    identifier split into words, an explicit ``"in <Parent>"`` phrase, the
    enclosing module directory, and the language. Tested by the
    ``multi_hop_retrieval`` benchmark — see ``docs/REPRODUCING.md``.
    """
    parts: list[str] = []

    # 1. Dotted form first — strongest lexical signal for "method in class"
    if node.parent_name and node.kind != "File":
        parts.append(f"{node.parent_name}.{node.name}")

    # 2. Bare name (always present)
    parts.append(node.name)

    # 3. Split-words form of the name (only if it differs from the bare name)
    name_split = _split_identifier(node.name)
    if name_split and name_split.lower() != node.name.lower():
        parts.append(name_split)

    # 4. Kind ("function", "class", "test", ...)
    if node.kind != "File":
        parts.append(node.kind.lower())

    # 5. Parent context with the split form too
    if node.parent_name:
        parts.append(f"in {node.parent_name}")
        parent_split = _split_identifier(node.parent_name)
        if parent_split and parent_split.lower() != node.parent_name.lower():
            parts.append(parent_split)

    # 6. Signature bits
    if node.params:
        parts.append(node.params)
    if node.return_type:
        parts.append(f"returns {node.return_type}")

    # 7. Documentation summary.  Existing databases may contain arbitrary
    # values in ``extra`` so accept strings only, normalize whitespace, and
    # re-apply the parser's bound before the text enters a provider request.
    raw_docstring = node.extra.get("docstring") if node.extra else None
    if isinstance(raw_docstring, str):
        docstring = " ".join(raw_docstring.split())[:_MAX_EMBEDDED_DOCSTRING_CHARS]
        if docstring:
            parts.append(docstring)

    # 8. Module / directory context from the file path — gives queries a
    # term like "routing" or "client" to anchor against.
    if node.file_path:
        parent_dir = Path(node.file_path).parent.name
        if parent_dir and parent_dir not in (".", "src", "lib"):
            parts.append(parent_dir)

    # 9. Language
    if node.language:
        parts.append(node.language)

    return " ".join(parts)


class EmbeddingStore:
    """Manages vector embeddings for graph nodes in SQLite."""

    def __init__(
        self,
        db_path: str | Path,
        provider: str | None = None,
        model: str | None = None,
    ) -> None:
        self.provider = get_provider(provider, model=model)
        self.available = self.provider is not None
        self.db_path = Path(db_path)
        self._conn = _with_db_retry(
            lambda: sqlite3.connect(
                str(self.db_path), timeout=30, check_same_thread=False,
                isolation_level=None,
            ),
            what=f"opening {self.db_path.name}",
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_EMBEDDINGS_SCHEMA)

        # Migration for existing DBs missing the provider column
        try:
            self._conn.execute("SELECT provider FROM embeddings LIMIT 1")
        except sqlite3.OperationalError:
            self._conn.execute(
                "ALTER TABLE embeddings ADD COLUMN provider "
                "TEXT NOT NULL DEFAULT 'unknown'"
            )

        self._conn.commit()

    def __enter__(self) -> "EmbeddingStore":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore[no-untyped-def]
        self.close()

    def close(self) -> None:
        self._conn.close()

    def embed_nodes(self, nodes: list[GraphNode], batch_size: int = 64) -> int:
        """Compute and store embeddings for a list of nodes."""
        if not self.provider:
            return 0

        # Filter to nodes that need embedding
        to_embed: list[tuple[GraphNode, str, str]] = []
        provider_name = self.provider.name

        kinds = embedded_node_kinds()
        for node in nodes:
            if node.kind not in kinds:
                continue
            text = _node_to_text(node)
            text_hash = hashlib.sha256(text.encode()).hexdigest()

            existing = self._conn.execute(
                "SELECT text_hash, provider FROM embeddings WHERE qualified_name = ?",
                (node.qualified_name,),
            ).fetchone()

            # Re-embed if text changed OR provider changed
            if (existing and existing["text_hash"] == text_hash
                    and existing["provider"] == provider_name):
                continue
            to_embed.append((node, text, text_hash))

        if not to_embed:
            return 0

        # Encode and persist in batches, committing each one before requesting
        # the next. Embedding a large graph is a long remote conversation that
        # can fail part-way through -- a provider timeout, a rate limit, or (on
        # Windows) ephemeral-port exhaustion from a local endpoint opening a
        # socket per input. Persisting only at the end discarded every vector
        # computed before the failure; a 19k-node graph lost minutes of GPU
        # work and made no progress across retries. Committing per batch means
        # a failure costs one batch, and re-running resumes from where it
        # stopped via the text_hash gate above.
        embedded = 0
        for start in range(0, len(to_embed), batch_size):
            batch = to_embed[start:start + batch_size]
            vectors = self.provider.embed([text for _, text, _ in batch])
            # Retried as a unit: a sync client holding the file must not cost
            # the GPU work already done for this batch.
            _with_db_retry(
                functools.partial(
                    self._persist_batch, batch, vectors, provider_name
                ),
                what="writing embeddings",
            )
            embedded += len(batch)

        return embedded

    def _persist_batch(
        self,
        batch: list[tuple[GraphNode, str, str]],
        vectors: list[list[float]],
        provider_name: str,
    ) -> None:
        """Write one batch of vectors and commit it."""
        for (node, _text, text_hash), vec in zip(batch, vectors):
            self._conn.execute(
                "INSERT OR REPLACE INTO embeddings "
                "(qualified_name, vector, text_hash, provider) "
                "VALUES (?, ?, ?, ?)",
                (node.qualified_name, _encode_vector(vec), text_hash, provider_name),
            )
        self._conn.commit()

    def search(self, query: str, limit: int = 20) -> list[tuple[str, float]]:
        """Search for nodes by semantic similarity.

        Scans every vector stored under the active provider identity. When
        NumPy is importable the scan is one reinterpret-cast plus one matrix
        product instead of a per-row decode-and-dot: measured at 1.69s -> 110ms
        (15x) on 19,394 nodes at 512 dimensions. The remaining cost is
        materialising the blobs, not the arithmetic, so it grows with the
        stored vector count rather than the dimension.

        Both paths rank identically. A stored vector whose dimension differs
        from the query scores 0.0 rather than raising, a zero-magnitude vector
        scores 0.0, and equal scores keep insertion order.
        """
        if not self.provider:
            return []

        provider_name = self.provider.name
        query_vec = self.provider.embed_query(query)

        # Only rows written under the current identity are comparable.
        cursor = self._conn.execute(
            "SELECT qualified_name, vector FROM embeddings WHERE provider = ?",
            (provider_name,),
        )
        if _np is None:
            return self._search_python(cursor, query_vec, limit)
        return self._search_numpy(cursor, query_vec, limit)

    def _search_python(
        self,
        cursor: sqlite3.Cursor,
        query_vec: list[float],
        limit: int,
    ) -> list[tuple[str, float]]:
        """Pure-Python cosine scan used when NumPy is unavailable."""
        scored: list[tuple[str, float]] = []
        while True:
            rows = cursor.fetchmany(_SEARCH_CHUNK_SIZE)
            if not rows:
                break
            for row in rows:
                vec = _decode_vector(row["vector"])
                sim = _cosine_similarity(query_vec, vec)
                scored.append((row["qualified_name"], sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]

    def _search_numpy(
        self,
        cursor: sqlite3.Cursor,
        query_vec: list[float],
        limit: int,
    ) -> list[tuple[str, float]]:
        """Vectorised cosine scan: one reinterpret-cast plus one matrix product."""
        expected_bytes = len(query_vec) * _FLOAT32_BYTES
        names: list[str] = []
        blobs: list[bytes] = []
        # Positions in `names` that have a dimension-compatible vector. Rows
        # that do not are kept in the ranking at 0.0 to match the Python path.
        comparable_positions: list[int] = []
        while True:
            rows = cursor.fetchmany(_SEARCH_CHUNK_SIZE)
            if not rows:
                break
            for row in rows:
                blob = row["vector"]
                if len(blob) == expected_bytes:
                    comparable_positions.append(len(names))
                    blobs.append(bytes(blob))
                names.append(row["qualified_name"])

        if not names:
            return []

        similarities = _np.zeros(len(names), dtype=_np.float32)
        # float32 throughout: the stored blobs already are, and upcasting the
        # whole matrix to float64 doubles the memory traffic that dominates
        # this scan for no ranking benefit (agreement with the Python path
        # stays within ~1e-7).
        query = _np.asarray(query_vec, dtype=_np.float32)
        query_norm = float(_np.linalg.norm(query))
        if blobs and query_norm > 0.0:
            matrix = _np.frombuffer(b"".join(blobs), dtype=_np.float32).reshape(
                len(blobs), len(query_vec)
            )
            norms = _np.linalg.norm(matrix, axis=1)
            nonzero = norms > 0.0
            if nonzero.all():
                # Common case. Masking would copy the entire matrix.
                comparable = (matrix @ query) / (norms * query_norm)
            else:
                comparable = _np.zeros(len(blobs), dtype=_np.float32)
                comparable[nonzero] = (matrix[nonzero] @ query) / (
                    norms[nonzero] * query_norm
                )
            similarities[_np.asarray(comparable_positions, dtype=_np.intp)] = comparable

        # Stable sort so equal scores keep the order rows arrived in.
        order = _np.argsort(-similarities, kind="stable")[:limit]
        return [(names[index], float(similarities[index])) for index in order]

    def remove_node(self, qualified_name: str) -> None:
        self._conn.execute(
            "DELETE FROM embeddings WHERE qualified_name = ?", (qualified_name,)
        )
        self._conn.commit()

    def purge_orphans(self) -> int:
        """Delete vectors that no longer belong in the index.

        Two cases: the graph node is gone, or the node's kind is no longer
        embedded. The second matters because narrowing ``CRG_EMBED_KINDS``
        otherwise has no visible effect -- ``embed_nodes`` simply stops
        refreshing those vectors while search keeps returning the stale ones,
        so excluding a kind would appear not to work.

        Embeddings and graph nodes normally share a SQLite file.  Standalone
        embedding databases remain supported, so a missing ``nodes`` table is
        an intentional no-op rather than an error.
        """
        has_nodes = self._conn.execute(
            "SELECT 1 FROM sqlite_master "
            "WHERE type = 'table' AND name = 'nodes'",
        ).fetchone()
        if has_nodes is None:
            return 0
        kinds = sorted(embedded_node_kinds())
        placeholders = ", ".join("?" for _ in kinds)
        cursor = self._conn.execute(
            "DELETE FROM embeddings "
            "WHERE NOT EXISTS ("
            "SELECT 1 FROM nodes "
            "WHERE nodes.qualified_name = embeddings.qualified_name"
            ") OR EXISTS ("
            "SELECT 1 FROM nodes "
            "WHERE nodes.qualified_name = embeddings.qualified_name "
            f"AND nodes.kind NOT IN ({placeholders})"
            ")",
            kinds,
        )
        self._conn.commit()
        return max(cursor.rowcount, 0)

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]


def embed_all_nodes(graph_store: GraphStore, embedding_store: EmbeddingStore) -> int:
    """Purge deleted nodes, then embed all current non-file nodes."""
    embedding_store.purge_orphans()
    if not embedding_store.available:
        return 0

    all_files = graph_store.get_all_files()
    all_nodes: list[GraphNode] = []
    for f in all_files:
        all_nodes.extend(graph_store.get_nodes_by_file(f))

    return embedding_store.embed_nodes(all_nodes)


def refresh_embeddings(
    graph_store: GraphStore,
    *,
    provider: str,
    model: str,
) -> dict[str, int] | None:
    """Refresh a previously embedded graph under one exact provider identity.

    This function is deliberately not called by default build paths.  Callers
    must supply both provider and model explicitly.  A graph with no existing
    vectors returns before provider resolution, so routine builds cannot load
    a local model, contact a cloud service, or incur API cost.

    Existing vectors must all use the identity resolved from the requested
    provider/model (including the endpoint for OpenAI-compatible providers).
    Refresh never silently migrates an index to another model or endpoint.
    """
    provider = provider.strip().lower()
    model = model.strip()
    if not provider or not model:
        raise ValueError(
            "Embedding refresh requires an explicit provider and model.",
        )

    has_table = graph_store._conn.execute(
        "SELECT 1 FROM sqlite_master "
        "WHERE type = 'table' AND name = 'embeddings'",
    ).fetchone()
    if has_table is None:
        return None
    has_rows = graph_store._conn.execute(
        "SELECT 1 FROM embeddings LIMIT 1",
    ).fetchone()
    if has_rows is None:
        return None
    try:
        rows = graph_store._conn.execute(
            "SELECT DISTINCT provider FROM embeddings ORDER BY provider",
        ).fetchall()
    except sqlite3.OperationalError as exc:
        if "no such column" in str(exc).lower() and "provider" in str(exc).lower():
            raise ValueError(
                "Embedding refresh refused: existing rows have no provider identity; "
                "run an explicit embed to migrate and rebuild the index.",
            ) from exc
        raise
    identities = {str(row["provider"]) for row in rows}

    embedding_store = EmbeddingStore(
        graph_store.db_path,
        provider=provider,
        model=model,
    )
    try:
        if not embedding_store.available or embedding_store.provider is None:
            raise RuntimeError(
                f"Embedding provider '{provider}' is unavailable in this environment.",
            )
        resolved_identity = embedding_store.provider.name
        if provider == "minimax":
            resolved_model = resolved_identity.partition(":")[2]
            if model != resolved_model:
                raise ValueError(
                    f"MiniMax refresh model must be '{resolved_model}', got '{model}'.",
                )
        if identities != {resolved_identity}:
            existing = ", ".join(sorted(identities))
            raise ValueError(
                "Embedding refresh refused: existing embeddings use "
                f"{existing}; requested provider resolves to {resolved_identity}.",
            )

        purged = embedding_store.purge_orphans()
        all_nodes: list[GraphNode] = []
        for file_path in graph_store.get_all_files():
            all_nodes.extend(graph_store.get_nodes_by_file(file_path))
        embedded = embedding_store.embed_nodes(all_nodes)
        return {"embedded": embedded, "purged": purged}
    finally:
        embedding_store.close()


def semantic_search(
    query: str,
    graph_store: GraphStore,
    embedding_store: EmbeddingStore,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Search nodes using vector similarity, falling back to keyword search."""
    if embedding_store.available and embedding_store.count() > 0:
        results = embedding_store.search(query, limit=limit)
        output = []
        for qn, score in results:
            node = graph_store.get_node(qn)
            if node:
                d = node_to_dict(node)
                d["similarity_score"] = round(score, 4)
                output.append(d)
        return output

    # Fallback to keyword search
    nodes = graph_store.search_nodes(query, limit=limit)
    return [node_to_dict(n) for n in nodes]
