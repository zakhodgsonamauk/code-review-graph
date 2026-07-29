"""Explicit, provider-scoped embedding refresh and orphan cleanup."""

import asyncio
import sys
from unittest.mock import MagicMock, patch

import pytest

from code_review_graph.embeddings import EmbeddingStore, embed_all_nodes
from code_review_graph.graph import GraphStore
from code_review_graph.parser import NodeInfo
from code_review_graph.postprocessing import run_post_processing
from code_review_graph.tools.build import _run_postprocess


class _StubProvider:
    dimension = 2

    def __init__(self, name: str = "local:test-model") -> None:
        self.name = name
        self.embedded: list[str] = []

    def embed(self, texts):
        self.embedded.extend(texts)
        return [[float(len(text)), 1.0] for text in texts]

    def embed_query(self, text):
        return [1.0, 0.0]


def _graph_with_function(tmp_path):
    db = tmp_path / "graph.db"
    store = GraphStore(db)
    file_path = str(tmp_path / "module.py")
    store.upsert_node(
        NodeInfo(
            kind="File",
            name=file_path,
            file_path=file_path,
            line_start=1,
            line_end=20,
            language="python",
        )
    )
    store.upsert_node(
        NodeInfo(
            kind="Function",
            name="keep",
            file_path=file_path,
            line_start=1,
            line_end=2,
            language="python",
        )
    )
    store.commit()
    return store, file_path


class TestOrphanCleanup:
    def test_purge_removes_only_vectors_without_graph_nodes(self, tmp_path):
        graph, _ = _graph_with_function(tmp_path)
        provider = _StubProvider()
        with patch("code_review_graph.embeddings.get_provider", return_value=provider):
            embeddings = EmbeddingStore(graph.db_path, provider="local", model="test-model")
        embeddings.embed_nodes(graph.get_all_nodes(exclude_files=False))
        embeddings._conn.execute(
            "INSERT INTO embeddings (qualified_name, vector, text_hash, provider) "
            "VALUES (?, ?, ?, ?)",
            ("deleted.py::ghost", b"\x00" * 8, "old", provider.name),
        )
        embeddings._conn.commit()

        try:
            assert embeddings.purge_orphans() == 1
            remaining = embeddings._conn.execute(
                "SELECT qualified_name FROM embeddings ORDER BY qualified_name",
            ).fetchall()
            assert [row["qualified_name"] for row in remaining] == [
                f"{tmp_path / 'module.py'}::keep",
            ]
        finally:
            embeddings.close()
            graph.close()

    def test_purge_is_safe_without_a_nodes_table(self, tmp_path):
        with patch("code_review_graph.embeddings.get_provider", return_value=None):
            embeddings = EmbeddingStore(tmp_path / "standalone.db")
        try:
            assert embeddings.purge_orphans() == 0
        finally:
            embeddings.close()

    def test_manual_embed_purges_even_when_provider_is_unavailable(self, tmp_path):
        graph, _ = _graph_with_function(tmp_path)
        with patch("code_review_graph.embeddings.get_provider", return_value=None):
            embeddings = EmbeddingStore(graph.db_path)
        embeddings._conn.execute(
            "INSERT INTO embeddings (qualified_name, vector, text_hash, provider) "
            "VALUES ('deleted.py::ghost', ?, 'old', 'unknown')",
            (b"\x00" * 8,),
        )
        embeddings._conn.commit()

        try:
            assert embed_all_nodes(graph, embeddings) == 0
            assert embeddings.count() == 0
        finally:
            embeddings.close()
            graph.close()


class TestExplicitRefresh:
    def test_never_embedded_graph_skips_without_resolving_provider(self, tmp_path):
        from code_review_graph.embeddings import refresh_embeddings

        graph, _ = _graph_with_function(tmp_path)
        try:
            with patch("code_review_graph.embeddings.get_provider") as get_provider:
                assert (
                    refresh_embeddings(
                        graph,
                        provider="openai",
                        model="costly-model",
                    )
                    is None
                )
            get_provider.assert_not_called()
        finally:
            graph.close()

    def test_exact_provider_refreshes_changed_nodes_and_purges_orphans(self, tmp_path):
        from code_review_graph.embeddings import refresh_embeddings

        graph, file_path = _graph_with_function(tmp_path)
        provider = _StubProvider()
        with patch("code_review_graph.embeddings.get_provider", return_value=provider):
            embeddings = EmbeddingStore(graph.db_path, provider="local", model="test-model")
            embeddings.embed_nodes(graph.get_all_nodes(exclude_files=False))
            embeddings._conn.execute(
                "INSERT INTO embeddings (qualified_name, vector, text_hash, provider) "
                "VALUES ('deleted.py::ghost', ?, 'old', ?)",
                (b"\x00" * 8, provider.name),
            )
            embeddings._conn.commit()
            embeddings.close()

            graph.upsert_node(
                NodeInfo(
                    kind="Function",
                    name="added",
                    file_path=file_path,
                    line_start=4,
                    line_end=5,
                    language="python",
                )
            )
            graph.commit()
            result = refresh_embeddings(
                graph,
                provider="local",
                model="test-model",
            )

        try:
            assert result == {"embedded": 1, "purged": 1}
        finally:
            graph.close()

    def test_provider_identity_mismatch_refuses_migration(self, tmp_path):
        from code_review_graph.embeddings import refresh_embeddings

        graph, _ = _graph_with_function(tmp_path)
        original = _StubProvider("local:original-model")
        with patch("code_review_graph.embeddings.get_provider", return_value=original):
            embeddings = EmbeddingStore(graph.db_path)
            embeddings.embed_nodes(graph.get_all_nodes(exclude_files=False))
            embeddings.close()

        requested = _StubProvider("local:new-model")
        try:
            with patch(
                "code_review_graph.embeddings.get_provider",
                return_value=requested,
            ):
                with pytest.raises(ValueError, match="existing embeddings use"):
                    refresh_embeddings(
                        graph,
                        provider="local",
                        model="new-model",
                    )
            assert requested.embedded == []
        finally:
            graph.close()

    def test_legacy_rows_without_provider_identity_are_refused_precisely(self, tmp_path):
        from code_review_graph.embeddings import refresh_embeddings

        graph, _ = _graph_with_function(tmp_path)
        graph._conn.executescript(
            "CREATE TABLE embeddings ("
            "qualified_name TEXT PRIMARY KEY, vector BLOB NOT NULL, "
            "text_hash TEXT NOT NULL"
            ");"
        )
        graph._conn.execute(
            "INSERT INTO embeddings (qualified_name, vector, text_hash) "
            "VALUES (?, ?, ?)",
            (f"{tmp_path / 'module.py'}::keep", b"\x00" * 8, "old"),
        )
        graph.commit()

        try:
            with patch("code_review_graph.embeddings.get_provider") as get_provider:
                with pytest.raises(ValueError, match="provider identity"):
                    refresh_embeddings(
                        graph,
                        provider="local",
                        model="test-model",
                    )
            get_provider.assert_not_called()
        finally:
            graph.close()


class TestRefreshWiring:
    def test_shared_postprocessing_is_default_off(self, tmp_path):
        graph, _ = _graph_with_function(tmp_path)
        try:
            with patch(
                "code_review_graph.embeddings.refresh_embeddings",
            ) as refresh:
                run_post_processing(graph)
            refresh.assert_not_called()
        finally:
            graph.close()

    def test_shared_postprocessing_refresh_is_explicit_and_fail_soft(self, tmp_path):
        graph, _ = _graph_with_function(tmp_path)
        try:
            with patch(
                "code_review_graph.embeddings.refresh_embeddings",
                return_value={"embedded": 3, "purged": 2},
            ) as refresh:
                result = run_post_processing(
                    graph,
                    embedding_provider="local",
                    embedding_model="test-model",
                )
            refresh.assert_called_once_with(
                graph,
                provider="local",
                model="test-model",
            )
            assert result["embeddings_refreshed"] == 3
            assert result["embeddings_purged"] == 2

            with patch(
                "code_review_graph.embeddings.refresh_embeddings",
                side_effect=RuntimeError("provider unavailable offline"),
            ):
                failed = run_post_processing(
                    graph,
                    embedding_provider="local",
                    embedding_model="test-model",
                )
            assert any("provider unavailable offline" in warning for warning in failed["warnings"])
        finally:
            graph.close()

    def test_build_postprocess_is_default_off_and_explicit_at_every_level(self, tmp_path):
        graph, _ = _graph_with_function(tmp_path)
        try:
            with patch(
                "code_review_graph.embeddings.refresh_embeddings",
                return_value={"embedded": 1, "purged": 1},
            ) as refresh:
                default_result: dict = {}
                _run_postprocess(graph, default_result, "none")
                refresh.assert_not_called()

                explicit_result: dict = {}
                _run_postprocess(
                    graph,
                    explicit_result,
                    "none",
                    embedding_provider="local",
                    embedding_model="test-model",
                )
            refresh.assert_called_once_with(
                graph,
                provider="local",
                model="test-model",
            )
            assert explicit_result["embeddings_refreshed"] == 1
            assert explicit_result["embeddings_purged"] == 1
        finally:
            graph.close()

    def test_partial_provider_scope_warns_without_attempting_refresh(self, tmp_path):
        graph, _ = _graph_with_function(tmp_path)
        try:
            with patch(
                "code_review_graph.embeddings.refresh_embeddings",
            ) as refresh:
                result = run_post_processing(
                    graph,
                    embedding_provider="local",
                )
            refresh.assert_not_called()
            assert any("provider and model" in warning.lower() for warning in result["warnings"])
        finally:
            graph.close()

    def test_missing_cloud_credentials_are_a_warning_not_a_build_failure(
        self,
        tmp_path,
        monkeypatch,
    ):
        graph, _ = _graph_with_function(tmp_path)
        with patch("code_review_graph.embeddings.get_provider", return_value=None):
            embeddings = EmbeddingStore(graph.db_path)
        embeddings._conn.execute(
            "INSERT INTO embeddings (qualified_name, vector, text_hash, provider) "
            "VALUES (?, ?, ?, ?)",
            (
                f"{tmp_path / 'module.py'}::keep",
                b"\x00" * 8,
                "old",
                "openai:test-model@https://api.example.test/v1",
            ),
        )
        embeddings._conn.commit()
        embeddings.close()
        for variable in (
            "CRG_OPENAI_API_KEY",
            "CRG_OPENAI_BASE_URL",
            "CRG_OPENAI_MODEL",
        ):
            monkeypatch.delenv(variable, raising=False)

        try:
            result = run_post_processing(
                graph,
                embedding_provider="openai",
                embedding_model="test-model",
            )
            assert result["signatures_computed"] == 2
            assert any(
                "Missing required setting" in warning
                for warning in result["warnings"]
            )
        finally:
            graph.close()

    def test_mcp_build_and_postprocess_forward_exact_scope(self):
        from code_review_graph import main as crg_main

        build_tool = getattr(
            crg_main.build_or_update_graph_tool,
            "fn",
            crg_main.build_or_update_graph_tool,
        )
        postprocess_tool = getattr(
            crg_main.run_postprocess_tool,
            "fn",
            crg_main.run_postprocess_tool,
        )
        with (
            patch.object(
                crg_main,
                "with_provenance",
                side_effect=lambda result, _root: result,
            ),
            patch.object(
                crg_main,
                "build_or_update_graph",
                return_value={"status": "ok"},
            ) as build,
            patch.object(
                crg_main,
                "run_postprocess",
                return_value={"status": "ok"},
            ) as postprocess,
        ):
            asyncio.run(
                build_tool(
                    repo_root="/repo",
                    embedding_provider="local",
                    embedding_model="test-model",
                )
            )
            asyncio.run(
                postprocess_tool(
                    repo_root="/repo",
                    embedding_provider="local",
                    embedding_model="test-model",
                )
            )

        assert build.call_args.kwargs["embedding_provider"] == "local"
        assert build.call_args.kwargs["embedding_model"] == "test-model"
        assert postprocess.call_args.kwargs["embedding_provider"] == "local"
        assert postprocess.call_args.kwargs["embedding_model"] == "test-model"

    def test_cli_build_forwards_exact_scope(self):
        from code_review_graph import cli

        argv = [
            "code-review-graph",
            "build",
            "--repo",
            "repo-root",
            "--embedding-provider",
            "local",
            "--embedding-model",
            "test-model",
        ]
        result = {"files_parsed": 1, "total_nodes": 2, "total_edges": 1}
        with (
            patch.object(sys, "argv", argv),
            patch(
                "code_review_graph.graph.GraphStore",
            ) as graph_store,
            patch(
                "code_review_graph.incremental.get_db_path",
                return_value=MagicMock(),
            ),
            patch(
                "code_review_graph.tools.build.build_or_update_graph",
                return_value=result,
            ) as build,
        ):
            graph_store.return_value = MagicMock()
            cli.main()

        build.assert_called_once_with(
            full_rebuild=True,
            repo_root="repo-root",
            postprocess="full",
            embedding_provider="local",
            embedding_model="test-model",
        )


class TestInferredRefreshIdentity:
    """Builds keep an existing index fresh without per-invocation flags.

    Git hooks and other automation call ``code-review-graph update`` with no
    arguments, so a flags-only refresh never runs and semantic search silently
    drifts behind the graph. Inference closes that gap; these tests pin the
    conditions under which it does and does not fire.
    """

    def test_openai_credentials_infer_provider_and_model(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CRG_OPENAI_API_KEY", "ollama")
        monkeypatch.setenv("CRG_OPENAI_BASE_URL", "http://127.0.0.1:11434/v1")
        monkeypatch.setenv("CRG_OPENAI_MODEL", "qwen3-embedding:4b")
        graph, _ = _graph_with_function(tmp_path)
        try:
            with patch(
                "code_review_graph.embeddings.refresh_embeddings",
                return_value={"embedded": 4, "purged": 0},
            ) as refresh:
                result: dict = {}
                _run_postprocess(graph, result, "none")
            refresh.assert_called_once_with(
                graph,
                provider="openai",
                model="qwen3-embedding:4b",
            )
            assert result["embeddings_refreshed"] == 4
        finally:
            graph.close()

    def test_explicit_provider_env_overrides_credential_inference(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setenv("CRG_OPENAI_API_KEY", "ollama")
        monkeypatch.setenv("CRG_OPENAI_BASE_URL", "http://127.0.0.1:11434/v1")
        monkeypatch.setenv("CRG_OPENAI_MODEL", "qwen3-embedding:4b")
        monkeypatch.setenv("CRG_EMBEDDING_PROVIDER", "local")
        monkeypatch.setenv("CRG_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        graph, _ = _graph_with_function(tmp_path)
        try:
            with patch(
                "code_review_graph.embeddings.refresh_embeddings",
                return_value={"embedded": 1, "purged": 0},
            ) as refresh:
                _run_postprocess(graph, {}, "none")
            refresh.assert_called_once_with(
                graph,
                provider="local",
                model="all-MiniLM-L6-v2",
            )
        finally:
            graph.close()

    def test_explicit_arguments_win_over_environment(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CRG_OPENAI_API_KEY", "ollama")
        monkeypatch.setenv("CRG_OPENAI_BASE_URL", "http://127.0.0.1:11434/v1")
        monkeypatch.setenv("CRG_OPENAI_MODEL", "qwen3-embedding:4b")
        graph, _ = _graph_with_function(tmp_path)
        try:
            with patch(
                "code_review_graph.embeddings.refresh_embeddings",
                return_value={"embedded": 1, "purged": 0},
            ) as refresh:
                _run_postprocess(
                    graph,
                    {},
                    "none",
                    embedding_provider="local",
                    embedding_model="test-model",
                )
            refresh.assert_called_once_with(
                graph,
                provider="local",
                model="test-model",
            )
        finally:
            graph.close()

    @pytest.mark.parametrize("opt_out", ["0", "false", "no", "off", "OFF"])
    def test_opt_out_env_disables_inference(self, monkeypatch, tmp_path, opt_out):
        monkeypatch.setenv("CRG_OPENAI_API_KEY", "ollama")
        monkeypatch.setenv("CRG_OPENAI_BASE_URL", "http://127.0.0.1:11434/v1")
        monkeypatch.setenv("CRG_OPENAI_MODEL", "qwen3-embedding:4b")
        monkeypatch.setenv("CRG_AUTO_EMBED_REFRESH", opt_out)
        graph, _ = _graph_with_function(tmp_path)
        try:
            with patch(
                "code_review_graph.embeddings.refresh_embeddings",
            ) as refresh:
                _run_postprocess(graph, {}, "none")
            refresh.assert_not_called()
        finally:
            graph.close()

    @pytest.mark.parametrize(
        "env",
        [
            {"CRG_OPENAI_API_KEY": "ollama"},
            {"CRG_OPENAI_BASE_URL": "http://127.0.0.1:11434/v1"},
            {
                "CRG_OPENAI_API_KEY": "ollama",
                "CRG_OPENAI_BASE_URL": "http://127.0.0.1:11434/v1",
            },
            {"CRG_EMBEDDING_PROVIDER": "local"},
            {"CRG_EMBEDDING_PROVIDER": "google", "CRG_EMBEDDING_MODEL": "gemini"},
        ],
        ids=[
            "key-only",
            "base-url-only",
            "credentials-without-model",
            "provider-without-model",
            "provider-without-model-env-mapping",
        ],
    )
    def test_partial_configuration_is_a_silent_no_op(self, monkeypatch, tmp_path, env):
        """A half-configured machine must not warn on every build."""
        for name, value in env.items():
            monkeypatch.setenv(name, value)
        graph, _ = _graph_with_function(tmp_path)
        try:
            warnings = []
            with patch(
                "code_review_graph.embeddings.refresh_embeddings",
            ) as refresh:
                warnings = _run_postprocess(graph, {}, "none")
            refresh.assert_not_called()
            assert warnings == []
        finally:
            graph.close()

    def test_never_embedded_graph_is_untouched_by_inference(self, monkeypatch, tmp_path):
        """The real safety net: refresh_embeddings bails before provider resolution.

        Inference hands over an identity, but a graph holding no vectors must
        still not load a model or reach an endpoint. Asserted against the real
        ``refresh_embeddings`` rather than a mock, with no reachable endpoint
        configured, so a regression here surfaces as an error rather than a
        silent network call.
        """
        monkeypatch.setenv("CRG_OPENAI_API_KEY", "unused")
        monkeypatch.setenv("CRG_OPENAI_BASE_URL", "http://127.0.0.1:9/v1")
        monkeypatch.setenv("CRG_OPENAI_MODEL", "should-never-be-loaded")
        graph, _ = _graph_with_function(tmp_path)
        try:
            result: dict = {}
            warnings = _run_postprocess(graph, result, "none")
            assert warnings == []
            assert "embeddings_refreshed" not in result
        finally:
            graph.close()


class _FailingAfterNBatchesProvider:
    """Embeds normally for N batches, then raises like a dropped connection."""

    name = "local:test-model"
    dimension = 2

    def __init__(self, batches_before_failure: int) -> None:
        self._remaining = batches_before_failure
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        if self._remaining <= 0:
            raise RuntimeError("OpenAI API HTTP 400: dial tcp: queue was full")
        self._remaining -= 1
        return [[float(len(text)), 1.0] for text in texts]

    def embed_query(self, text):
        return [1.0, 0.0]


def _graph_with_functions(tmp_path, count):
    db = tmp_path / "graph.db"
    store = GraphStore(db)
    file_path = str(tmp_path / "module.py")
    store.upsert_node(
        NodeInfo(
            kind="File",
            name=file_path,
            file_path=file_path,
            line_start=1,
            line_end=count * 2,
            language="python",
        )
    )
    for index in range(count):
        store.upsert_node(
            NodeInfo(
                kind="Function",
                name=f"func_{index}",
                file_path=file_path,
                line_start=index * 2 + 1,
                line_end=index * 2 + 2,
                language="python",
            )
        )
    store.commit()
    return store, file_path


class TestIncrementalEmbeddingPersistence:
    """Vectors are committed per batch, so a mid-run failure keeps progress.

    Persisting only after every vector returned meant one dropped connection
    discarded the whole pass. On a 19k-node graph that lost minutes of GPU work
    and left the index empty, so retries restarted from nothing.
    """

    def test_batches_are_committed_before_the_next_request(self, tmp_path):
        graph, _ = _graph_with_functions(tmp_path, 10)
        provider = _StubProvider()
        try:
            with patch(
                "code_review_graph.embeddings.get_provider", return_value=provider
            ):
                store = EmbeddingStore(graph.db_path)
            try:
                embedded = store.embed_nodes(graph.get_all_nodes(), batch_size=4)
                assert embedded == 10
                assert store.count() == 10
            finally:
                store.close()
        finally:
            graph.close()

    def test_failure_part_way_through_keeps_completed_batches(self, tmp_path):
        graph, _ = _graph_with_functions(tmp_path, 10)
        provider = _FailingAfterNBatchesProvider(batches_before_failure=2)
        try:
            with patch(
                "code_review_graph.embeddings.get_provider", return_value=provider
            ):
                store = EmbeddingStore(graph.db_path)
            try:
                with pytest.raises(RuntimeError, match="queue was full"):
                    store.embed_nodes(graph.get_all_nodes(), batch_size=4)

                # Two batches of four were committed before the third failed.
                assert store.count() == 8
            finally:
                store.close()
        finally:
            graph.close()

    def test_rerun_resumes_only_the_missing_nodes(self, tmp_path):
        graph, _ = _graph_with_functions(tmp_path, 10)
        failing = _FailingAfterNBatchesProvider(batches_before_failure=2)
        try:
            with patch(
                "code_review_graph.embeddings.get_provider", return_value=failing
            ):
                store = EmbeddingStore(graph.db_path)
            try:
                with pytest.raises(RuntimeError):
                    store.embed_nodes(graph.get_all_nodes(), batch_size=4)
            finally:
                store.close()

            healthy = _StubProvider()
            with patch(
                "code_review_graph.embeddings.get_provider", return_value=healthy
            ):
                store = EmbeddingStore(graph.db_path)
            try:
                embedded = store.embed_nodes(graph.get_all_nodes(), batch_size=4)
                # Only the two nodes left unembedded are re-requested.
                assert embedded == 2
                assert store.count() == 10
                assert len(healthy.embedded) == 2
            finally:
                store.close()
        finally:
            graph.close()
