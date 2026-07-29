# Semantic search on a local Ollama GPU

End-to-end setup for vector search backed by a local Ollama endpoint, with no
`sentence-transformers` and no torch. Written to be repeatable: every step is a
command, and each has a check you can run to confirm it worked.

The `openai` provider speaks the OpenAI `/v1/embeddings` schema over stdlib
`urllib`, and Ollama serves exactly that. Localhost endpoints are recognised, so
no cloud-egress warning is emitted and nothing leaves the machine.

## 1. Prepare the model

```bash
ollama pull qwen3-embedding:4b
```

Embedded text is a short signature string plus a documentation summary — a few
hundred tokens at most — so the model's default 40K context is far larger than
needed and costs VRAM for nothing. A small-context variant is the same weights
with a smaller KV cache:

```bash
printf 'FROM qwen3-embedding:4b\nPARAMETER num_ctx 2048\n' > Modelfile.crg
ollama create qwen3-embedding-ctx2048:4b -f Modelfile.crg
```

Measured on an RTX 4080: 10.0GB resident at 40960 context versus 4.1GB at 2048,
with identical throughput (~1.4s per 100 inputs) and identical cold-load time.
Only VRAM changes. The tradeoff is that an input longer than the window is
**silently truncated** — Ollama does not error — so leave real headroom.

Keeping the model resident removes a ~2.5s cold load per query. Ollama's default
is 5 minutes:

```bash
setx OLLAMA_KEEP_ALIVE 10m          # Windows; export it on macOS/Linux
```

Ollama must be restarted to pick this up, and on Windows `setx` does not reach
running processes — so relaunch it from a shell where the variable is set, or log
out and back in.

**Check:** `ollama ps` shows the model, its context, and how long it stays loaded.

## 2. Install

```bash
pipx install --force --editable "/path/to/code-review-graph[perf]"
```

`[perf]` pulls NumPy alone, which is all the vectorised search scan needs.
`[embeddings]` also pulls sentence-transformers and torch — only required for the
`local` provider.

`--editable` means new processes pick up source edits with no reinstall. It does
**not** update an already-running MCP server; see TROUBLESHOOTING.md.

**Check:** `code-review-graph --version`, and
`python -c "import code_review_graph.embeddings as e; print(e._np is not None)"`
should print `True`.

## 3. Configure once, for every project

Write `~/.code-review-graph/embeddings.toml` (or `$CRG_HOME/embeddings.toml`):

```toml
provider  = "openai"
model     = "qwen3-embedding-ctx2048:4b"
base_url  = "http://127.0.0.1:11434/v1"
api_key   = "ollama"        # required non-empty; Ollama ignores the value
dimension = 512             # Matryoshka truncation from the model's native 2560
```

Prefer this file over exported environment variables. Git hooks inherit the shell
that ran `git commit`, and an MCP server keeps whatever environment it started
with — both routinely miss variables exported later and then silently skip
refreshing the index.

`dimension` is worth setting deliberately. Search scans every stored vector, so
its cost scales with dimension: on ~12k vectors, 512 dimensions is ~110ms of
scan, while the model's native 2560 would be ~8x that and ~5x the disk. Changing
it later changes the provider identity and forces a full re-embed, so choose
before the first pass.

**Check:** in any git repo, `code-review-graph status` reports embedding
availability without needing any `CRG_OPENAI_*` variable set.

## 4. Register the MCP server for every project

```bash
claude mcp add --scope user code-review-graph -- code-review-graph serve
```

`--scope user` makes it available in all projects. If a project also registers
the server in its own `.mcp.json`, that entry takes precedence — remove it or the
project will keep using whatever it names (commonly `uvx code-review-graph`,
which fetches the published package rather than your install).

Environment can be pinned into the registration itself as a belt-and-braces
alternative to the config file:

```bash
claude mcp add --scope user code-review-graph \
  --env CRG_OPENAI_BASE_URL=http://127.0.0.1:11434/v1 \
  --env CRG_OPENAI_API_KEY=ollama \
  --env CRG_OPENAI_MODEL=qwen3-embedding-ctx2048:4b \
  --env CRG_OPENAI_DIMENSION=512 \
  -- code-review-graph serve
```

**Check:** `claude mcp get code-review-graph` shows `Status: ✔ Connected` and the
command you expect. Restart the MCP client for changes to take effect.

## 5. Per new project

```bash
cd /path/to/project
code-review-graph build      # full parse; required to populate docstrings
code-review-graph embed      # one pass over the graph
```

`build`, not `update`: `update` only re-parses changed files, so on a graph built
by an older version the documentation summaries stay absent everywhere you have
not since touched.

Both print an `Embeddings:` line. `skipped (<reason>)` means no provider resolved
in that process — fix the reason rather than ignoring it.

After this, `build`/`update` keep the index in step automatically, so the one-off
`embed` is not needed again.

For that to happen on every commit, the project also needs the pre-commit hook,
which runs `update` and is installed per repository by `install_git_hook` —
called for you by `code-review-graph install`. Without it, refresh only happens
when you run `build`/`update` yourself.

**Check:** ask a question that shares no identifiers with its answer, e.g.
`semantic_search_nodes("work out which office a case belongs to from its
reference number")`. The response should report `"search_mode": "semantic"` — if
it says keyword, no vectors exist for the resolved identity.

## Notes on scale

- Roughly 4.5 minutes per 20k nodes on a 4080, and incremental afterwards: a
  commit touching 20 functions re-embeds only those.
- Storage is `dimension x 4` bytes per vector: ~2KB each at 512 dimensions, so
  ~24MB for 12k nodes.
- Test nodes are not embedded by default (`CRG_EMBED_KINDS` to change). On one
  real graph they were 40% of the index and crowded source out of results, while
  `TESTED_BY` edges already answer coverage questions exactly.
- On Windows, a very large first pass can exhaust ephemeral ports because Ollama
  opens a socket per input. This is retried with a long backoff and committed per
  batch, so it recovers; see TROUBLESHOOTING.md to avoid it outright.
