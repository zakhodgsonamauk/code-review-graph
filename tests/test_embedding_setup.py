"""Unattended embedding configuration."""

from __future__ import annotations

import pytest

from code_review_graph.embedding_setup import (
    DEFAULT_DIMENSION,
    FALLBACK_PULL_SUGGESTION,
    SetupError,
    _api_root,
    discover_embedding_models,
    probe_dimensions,
    render_config,
    select_model,
    setup_embeddings,
)


class _FakeEndpoint:
    """Minimal stand-in for Ollama's native and OpenAI-compatible surfaces."""

    def __init__(self, models: dict[str, list[str]], native: int, truncates: bool = True):
        self._models = models
        self._native = native
        self._truncates = truncates
        self.embed_calls: list[dict] = []

    def get_json(self, url: str) -> dict:
        assert url.endswith("/api/tags"), url
        return {"models": [{"model": name} for name in self._models]}

    def post_json(self, url: str, payload: dict) -> dict:
        if url.endswith("/api/show"):
            model = payload["model"]
            if model not in self._models:
                raise RuntimeError("unknown model")
            return {"capabilities": self._models[model]}
        assert url.endswith("/embeddings"), url
        self.embed_calls.append(payload)
        width = payload.get("dimensions")
        if width is None or not self._truncates:
            width = self._native
        return {"data": [{"index": 0, "embedding": [0.0] * width}]}


class TestApiRoot:
    @pytest.mark.parametrize(
        ("base_url", "expected"),
        [
            ("http://127.0.0.1:11434/v1", "http://127.0.0.1:11434"),
            ("http://127.0.0.1:11434/v1/", "http://127.0.0.1:11434"),
            ("http://host:8000", "http://host:8000"),
        ],
    )
    def test_strips_openai_suffix(self, base_url, expected):
        assert _api_root(base_url) == expected


class TestDiscovery:
    def test_only_embedding_capable_models_are_offered(self):
        endpoint = _FakeEndpoint(
            {
                "qwen3-embedding:4b": ["tools", "embedding"],
                "nomic-embed-text:latest": ["embedding"],
                "qwen3-coder:30b": ["completion", "tools"],
            },
            native=2560,
        )

        found = discover_embedding_models(
            "http://127.0.0.1:11434/v1",
            get_json=endpoint.get_json,
            post_json=endpoint.post_json,
        )

        assert found == ["qwen3-embedding:4b", "nomic-embed-text:latest"]

    def test_unreachable_endpoint_explains_itself(self):
        def explode(_url):
            raise OSError("connection refused")

        with pytest.raises(SetupError, match="Is Ollama running"):
            discover_embedding_models("http://127.0.0.1:11434/v1", get_json=explode)


class TestModelSelection:
    def test_prefers_larger_qwen_over_small_encoders(self):
        assert select_model(
            ["all-minilm:latest", "qwen3-embedding:4b", "nomic-embed-text"]
        ) == "qwen3-embedding:4b"

    def test_falls_back_to_a_small_model_when_that_is_all_there_is(self):
        assert select_model(["embeddinggemma:latest"]) == "embeddinggemma:latest"

    def test_explicit_request_wins_even_if_not_listed(self):
        """The model may have been pulled seconds ago, or served by something
        other than Ollama."""
        assert select_model(["all-minilm"], "text-embedding-3-small") == (
            "text-embedding-3-small"
        )

    def test_unknown_names_still_yield_a_choice(self):
        assert select_model(["some-private-encoder:v2"]) == "some-private-encoder:v2"

    def test_no_models_names_what_to_pull(self):
        with pytest.raises(SetupError, match=FALLBACK_PULL_SUGGESTION):
            select_model([])


class TestDimensionProbe:
    def test_truncation_is_verified_not_assumed(self):
        endpoint = _FakeEndpoint({"m": ["embedding"]}, native=2560)

        native, chosen, honoured = probe_dimensions(
            "http://x/v1", "m", "k", None, post_json=endpoint.post_json
        )

        assert (native, chosen, honoured) == (2560, DEFAULT_DIMENSION, True)
        # Asked twice: once bare to learn native width, once to confirm.
        assert [call.get("dimensions") for call in endpoint.embed_calls] == [
            None, DEFAULT_DIMENSION
        ]

    def test_endpoint_ignoring_dimensions_records_native_width(self):
        """Recording a width the endpoint ignores would misdescribe every vector."""
        endpoint = _FakeEndpoint({"m": ["embedding"]}, native=768, truncates=False)

        native, chosen, honoured = probe_dimensions(
            "http://x/v1", "m", "k", 256, post_json=endpoint.post_json
        )

        assert (native, chosen, honoured) == (768, 768, False)

    def test_model_smaller_than_the_target_is_left_alone(self):
        endpoint = _FakeEndpoint({"m": ["embedding"]}, native=384)

        native, chosen, honoured = probe_dimensions(
            "http://x/v1", "m", "k", None, post_json=endpoint.post_json
        )

        assert (native, chosen, honoured) == (384, 384, True)
        assert [call.get("dimensions") for call in endpoint.embed_calls] == [None]


class TestRenderConfig:
    def test_dimension_is_omitted_when_native(self):
        rendered = render_config(
            provider="openai", model="m", base_url="http://x/v1",
            api_key="k", dimension=None,
        )
        assert "dimension" not in rendered
        assert 'model = "m"' in rendered

    def test_environment_precedence_is_stated_in_the_file(self):
        rendered = render_config(
            provider="openai", model="m", base_url="http://x/v1",
            api_key="k", dimension=512,
        )
        assert "override" in rendered
        assert "dimension = 512" in rendered


class TestSetupEmbeddings:
    def test_refuses_to_clobber_without_force(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CRG_HOME", str(tmp_path))
        (tmp_path / "embeddings.toml").write_text("provider = 'openai'\n")

        with pytest.raises(SetupError, match="--force"):
            setup_embeddings(model="m", dimension=8)

    def test_print_only_writes_nothing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CRG_HOME", str(tmp_path))
        endpoint = _FakeEndpoint({"m": ["embedding"]}, native=16)
        monkeypatch.setattr(
            "code_review_graph.embedding_setup._post_json", endpoint.post_json
        )

        summary = setup_embeddings(model="m", write=False)

        assert summary["written"] is False
        assert not (tmp_path / "embeddings.toml").exists()
