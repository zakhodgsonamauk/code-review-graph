"""Shared test fixtures.

Keeps code-review-graph's own per-user state out of the developer's real
home directory. Scoped deliberately: the editor-integration installers in
``skills.py`` write to other user-level locations (``~/.codex``,
``~/.cursor``, ``~/.config/opencode``) that are outside CRG state and are
not covered here — those tests patch ``Path.home()`` themselves.
"""

from __future__ import annotations

import pytest

# Every environment variable that can select or configure an embedding
# provider. A developer with a working local embedding setup exports these
# permanently, so they leak into the suite unless cleared per test.
_EMBEDDING_ENV_VARS = (
    "CRG_OPENAI_API_KEY",
    "CRG_OPENAI_BASE_URL",
    "CRG_OPENAI_MODEL",
    "CRG_OPENAI_DIMENSION",
    "CRG_OPENAI_BATCH_SIZE",
    "CRG_EMBEDDING_PROVIDER",
    "CRG_EMBEDDING_MODEL",
    "CRG_AUTO_EMBED_REFRESH",
    "CRG_ACCEPT_CLOUD_EMBEDDINGS",
    "GOOGLE_API_KEY",
    "MINIMAX_API_KEY",
)


@pytest.fixture(autouse=True)
def isolated_embedding_env(monkeypatch):
    """Clear embedding provider configuration inherited from the machine.

    ``get_provider()`` selects the openai provider whenever
    ``CRG_OPENAI_API_KEY`` and ``CRG_OPENAI_BASE_URL`` are set, and builds
    infer a refresh identity from the same variables. Both are correct in
    production and both make the suite machine-dependent: a developer running
    a local Ollama endpoint would see "refresh is off by default" assertions
    fail, while CI passes.

    Autouse and unconditional, matching ``isolated_crg_home``. Tests that
    want a provider configured set the variables they need with
    ``monkeypatch.setenv``, which applies after this fixture has run.
    """
    for name in _EMBEDDING_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def isolated_crg_home(tmp_path_factory, monkeypatch):
    """Redirect the per-user state directory into a temporary directory.

    ``~/.code-review-graph`` holds ``registry.json``, ``watch.toml``,
    ``daemon.pid``, ``daemon-state.json`` and ``logs/``. Two paths reached
    the real one:

    * ``Registry()`` defaults there, and ``incremental.get_data_dir()``
      constructs one internally — so any test touching data-dir resolution
      both read and wrote the registry of whoever ran the suite. That put
      pytest tmp paths into a developer's home directory, and made those
      tests depend on machine state: a developer with a registered repo
      could get different results from one without.
    * ``daemon`` built its config/PID/state paths from ``Path.home()``.

    Autouse and unconditional: an opt-in fixture would silently stop
    protecting a test the day someone forgets to request it.
    """
    home = tmp_path_factory.mktemp("crg-home")
    monkeypatch.setenv("CRG_HOME", str(home))
    return home
