"""Documentation summaries used by semantic embeddings."""

from pathlib import Path
from unittest.mock import patch

from code_review_graph.embeddings import (
    _MAX_EMBEDDED_DOCSTRING_CHARS,
    _node_to_text,
)
from code_review_graph.graph import GraphNode, GraphStore
from code_review_graph.incremental import full_build, incremental_update
from code_review_graph.parser import _MAX_DOCSTRING_CHARS, CodeParser


def _parsed_node(path: str, source: bytes, name: str):
    nodes, _ = CodeParser().parse_bytes(Path(path), source)
    return next(node for node in nodes if node.name == name)


class TestDocumentationSummaryExtraction:
    def test_python_uses_runtime_string_value_and_keeps_later_paragraphs(self):
        node = _parsed_node(
            "module.py",
            (
                b"def parse_rates():\n"
                b'    (r"Parse\\n" " uploaded \\u20ac rate sheets.\\n\\nDetails.")\n'
                b"    return []\n"
            ),
            "parse_rates",
        )

        assert node.extra["docstring"] == (
            r"Parse\n uploaded " + "\N{EURO SIGN} rate sheets. Details."
        )

    def test_python_rejects_bytes_and_fstrings(self):
        for literal in (b'b"not docs"', b'f"not {1} docs"'):
            node = _parsed_node(
                "module.py",
                b"def f():\n    " + literal + b"\n    return 1\n",
                "f",
            )
            assert "docstring" not in node.extra

    def test_python_class_and_method_docstrings_are_independent(self):
        nodes, _ = CodeParser().parse_bytes(
            Path("module.py"),
            (
                b"class Parser:\n"
                b'    """Parses uploaded files."""\n'
                b"\n"
                b"    def run(self):\n"
                b'        """Run one parse."""\n'
                b"        return None\n"
            ),
        )
        by_name = {node.name: node for node in nodes}

        assert by_name["Parser"].extra["docstring"] == "Parses uploaded files."
        assert by_name["run"].extra["docstring"] == "Run one parse."

    def test_jsdoc_on_exported_function_and_blank_line_boundary(self):
        documented = _parsed_node(
            "module.ts",
            b"/** Parse uploaded sheets. */\nexport function parse() {}\n",
            "parse",
        )
        detached = _parsed_node(
            "module.ts",
            b"/** Module banner. */\n\nexport function parse() {}\n",
            "parse",
        )

        assert documented.extra["docstring"] == "Parse uploaded sheets."
        assert "docstring" not in detached.extra

    def test_plain_javascript_comment_is_not_documentation(self):
        node = _parsed_node(
            "module.js",
            b"// implementation note\nfunction parse() {}\n",
            "parse",
        )

        assert "docstring" not in node.extra

    def test_go_plain_comment_block_is_documentation(self):
        node = _parsed_node(
            "module.go",
            (
                b"package parser\n\n"
                b"// Parse reads an uploaded sheet\n"
                b"// and returns normalized rows.\n"
                b"func Parse() {}\n"
            ),
            "Parse",
        )

        assert node.extra["docstring"] == (
            "Parse reads an uploaded sheet and returns normalized rows."
        )

    def test_go_compiler_directive_is_not_embedding_text(self):
        node = _parsed_node(
            "module.go",
            (
                b"package parser\n\n"
                b"// Parse reads an uploaded sheet.\n"
                b"//go:noinline\n"
                b"func Parse() {}\n"
            ),
            "Parse",
        )

        assert node.extra["docstring"] == "Parse reads an uploaded sheet."

    def test_rust_outer_docs_cross_attributes_but_inner_docs_do_not_attach(self):
        documented = _parsed_node(
            "module.rs",
            b"/// Parse a sheet.\n#[inline]\nfn parse() {}\n",
            "parse",
        )
        inner = _parsed_node(
            "module.rs",
            b"//! Module documentation.\nfn parse() {}\n",
            "parse",
        )

        assert documented.extra["docstring"] == "Parse a sheet."
        assert "docstring" not in inner.extra

    def test_javadoc_keeps_only_the_summary_paragraph(self):
        node = _parsed_node(
            "Parser.java",
            (
                b"class Parser {\n"
                b"  /**\n"
                b"   * Parse a rate sheet.\n"
                b"   *\n"
                b"   * @param path uploaded file\n"
                b"   */\n"
                b"  void parse(String path) {}\n"
                b"}\n"
            ),
            "parse",
        )

        assert node.extra["docstring"] == "Parse a rate sheet."

    def test_javadoc_html_paragraph_boundary_excludes_details(self):
        node = _parsed_node(
            "Parser.java",
            (
                b"class Parser {\n"
                b"  /** Parse a {@code RateSheet}.\n"
                b"   * <p>Implementation details must not be embedded.\n"
                b"   */\n"
                b"  void parse() {}\n"
                b"}\n"
            ),
            "parse",
        )

        assert node.extra["docstring"] == "Parse a RateSheet."

    def test_csharp_xml_summary_crosses_attribute(self):
        node = _parsed_node(
            "Parser.cs",
            (
                b"class Parser {\n"
                b"  /// <summary>\n"
                b"  /// Parse a rate sheet.\n"
                b"  /// </summary>\n"
                b"  [Obsolete]\n"
                b"  public void Parse() {}\n"
                b"}\n"
            ),
            "Parse",
        )

        assert node.extra["docstring"] == "Parse a rate sheet."

    def test_doxygen_comment_attaches_to_cpp_template_function(self):
        node = _parsed_node(
            "parser.cpp",
            (
                b"/** Parse a typed sheet. */\n"
                b"template <typename T>\n"
                b"T parse(T value) { return value; }\n"
            ),
            "parse",
        )

        assert node.extra["docstring"] == "Parse a typed sheet."

    def test_doxygen_brief_marker_is_not_embedded_as_prose(self):
        node = _parsed_node(
            "parser.cpp",
            b"/** @brief Parse a typed sheet. */\nint parse() { return 0; }\n",
            "parse",
        )

        assert node.extra["docstring"] == "Parse a typed sheet."

    def test_summary_is_bounded_by_the_configured_maximum(self):
        oversized = "word " * (_MAX_DOCSTRING_CHARS // 2)
        node = _parsed_node(
            "module.py",
            ('def f():\n    """' + oversized + '"""\n').encode(),
            "f",
        )

        assert len(oversized) > _MAX_DOCSTRING_CHARS, "input must exceed the bound"
        assert len(node.extra["docstring"]) == _MAX_DOCSTRING_CHARS

    def test_embedded_text_bound_matches_the_parser_bound(self):
        """A larger embedding-side bound would be dead code, a smaller one a
        silent second truncation. Neither is intended."""
        assert _MAX_EMBEDDED_DOCSTRING_CHARS == _MAX_DOCSTRING_CHARS


class TestDocumentationEmbeddingText:
    @staticmethod
    def _node(extra: dict) -> GraphNode:
        return GraphNode(
            id=1,
            kind="Function",
            name="parse_rates",
            qualified_name="module.py::parse_rates",
            file_path="module.py",
            line_start=1,
            line_end=2,
            language="python",
            parent_name=None,
            params=None,
            return_type=None,
            is_test=False,
            file_hash=None,
            extra=extra,
        )

    def test_text_includes_normalized_bounded_summary_deterministically(self):
        overshoot = "x" * (_MAX_EMBEDDED_DOCSTRING_CHARS + 100)
        summary = "  Parse\n uploaded\t rate sheets.  " + overshoot

        first = _node_to_text(self._node({"docstring": summary}))
        second = _node_to_text(self._node({"docstring": summary}))

        assert first == second
        assert "Parse uploaded rate sheets." in first
        assert ("x" * (_MAX_EMBEDDED_DOCSTRING_CHARS + 1)) not in first

    def test_non_string_legacy_metadata_is_ignored(self):
        plain = _node_to_text(self._node({}))
        malformed = _node_to_text(self._node({"docstring": {"unexpected": "shape"}}))

        assert malformed == plain


def test_docstring_metadata_survives_full_and_incremental_persistence(
    tmp_path,
    monkeypatch,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    source = repo / "module.py"
    source.write_text('def parse():\n    """Old summary."""\n', encoding="utf-8")
    store = GraphStore(tmp_path / "graph.db")
    monkeypatch.setenv("CRG_SERIAL_PARSE", "1")

    try:
        with patch(
            "code_review_graph.incremental.collect_all_files",
            return_value=["module.py"],
        ):
            full_build(repo, store)
        full_node = next(
            node for node in store.get_nodes_by_file(str(source)) if node.name == "parse"
        )
        assert full_node.extra["docstring"] == "Old summary."

        source.write_text('def parse():\n    """New summary."""\n', encoding="utf-8")
        incremental_update(repo, store, changed_files=["module.py"])
        changed_node = next(
            node for node in store.get_nodes_by_file(str(source)) if node.name == "parse"
        )
        assert changed_node.extra["docstring"] == "New summary."

        source.write_text("def parse():\n    return None\n", encoding="utf-8")
        incremental_update(repo, store, changed_files=["module.py"])
        removed_node = next(
            node for node in store.get_nodes_by_file(str(source)) if node.name == "parse"
        )
        assert "docstring" not in removed_node.extra
    finally:
        store.close()


class TestProseParagraphCapture:
    """Python docstrings keep later paragraphs; tag-based doc comments do not.

    Measured on a 19,595-node Django/React graph: 27.5% of nodes are
    documented, the median stored summary was 66 characters, and 26.8% of
    Python docstrings had a further 258 characters (p90 692) sitting unused
    past the first paragraph. Only 2.4% used Google/NumPy section headings, so
    paragraph capture -- not section parsing -- is where the signal is.
    """

    def test_python_joins_multiple_prose_paragraphs(self):
        node = _parsed_node(
            "module.py",
            (
                b"def resolve():\n"
                b'    """Resolve the working file.\n'
                b"\n"
                b"    Falls back to the staging area when the local copy is absent.\n"
                b"\n"
                b"    Callers must hold the case lock.\n"
                b'    """\n'
                b"    return None\n"
            ),
            "resolve",
        )

        assert node.extra["docstring"] == (
            "Resolve the working file. "
            "Falls back to the staging area when the local copy is absent. "
            "Callers must hold the case lock."
        )

    def test_python_stops_at_a_parameter_section(self):
        """Parameter names already live in NodeInfo.params."""
        node = _parsed_node(
            "module.py",
            (
                b"def resolve(case_no):\n"
                b'    """Resolve the working file.\n'
                b"\n"
                b"    Prefers the local copy.\n"
                b"\n"
                b"    Args:\n"
                b"        case_no: the case reference\n"
                b'    """\n'
                b"    return None\n"
            ),
            "resolve",
        )

        docstring = node.extra["docstring"]
        assert docstring == "Resolve the working file. Prefers the local copy."
        assert "case_no" not in docstring

    def test_python_keeps_a_returns_section_when_no_parameters_precede_it(self):
        node = _parsed_node(
            "module.py",
            (
                b"def resolve():\n"
                b'    """Resolve the working file.\n'
                b"\n"
                b"    Returns:\n"
                b"        The absolute NAS path, or None when unavailable.\n"
                b'    """\n'
                b"    return None\n"
            ),
            "resolve",
        )

        assert "absolute NAS path" in node.extra["docstring"]

    def test_python_drops_numpy_section_underlines(self):
        node = _parsed_node(
            "module.py",
            (
                b"def resolve():\n"
                b'    """Resolve the working file.\n'
                b"\n"
                b"    Notes\n"
                b"    -----\n"
                b"    Requires the NAS to be awake.\n"
                b'    """\n'
                b"    return None\n"
            ),
            "resolve",
        )

        docstring = node.extra["docstring"]
        assert "-----" not in docstring
        assert "Requires the NAS to be awake." in docstring

    def test_python_bound_applies_across_joined_paragraphs(self):
        paragraphs = b"\n\n".join(
            b"    " + (b"word " * 60) for _ in range(6)
        )
        node = _parsed_node(
            "module.py",
            b'def f():\n    """Summary.\n\n' + paragraphs + b'\n    """\n',
            "f",
        )

        assert len(node.extra["docstring"]) == _MAX_DOCSTRING_CHARS

    def test_tag_based_doc_comments_still_keep_only_the_first_paragraph(self):
        """Later Javadoc/Doxygen paragraphs are implementation detail."""
        java = _parsed_node(
            "Parser.java",
            (
                b"class Parser {\n"
                b"  /**\n"
                b"   * Parse a rate sheet.\n"
                b"   *\n"
                b"   * Implementation detail that must not be embedded.\n"
                b"   */\n"
                b"  void parse() {}\n"
                b"}\n"
            ),
            "parse",
        )
        rust = _parsed_node(
            "module.rs",
            (
                b"/// Parse a sheet.\n"
                b"///\n"
                b"/// Internal notes that must not be embedded.\n"
                b"fn parse() {}\n"
            ),
            "parse",
        )

        assert java.extra["docstring"] == "Parse a rate sheet."
        assert rust.extra["docstring"] == "Parse a sheet."
