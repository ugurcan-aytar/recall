# Changelog

All notable changes to recall are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Repository scaffolding: Go module, Cobra command skeleton with stub subcommands, CI workflow, license, README, CONTRIBUTING, SECURITY, editorconfig, goreleaser config, GitHub issue and PR templates (Phase R0).
- SQLite store with FTS5 (WAL, 64 MB cache, prepared-statement cache for BM25), collections, documents with incremental hash-based reconciliation, path contexts, metadata (Phase R1).
- `recall collection add/remove/list/rename`, `recall ls`, `recall index [--pull]`, `recall search` (BM25 with `--json/--csv/--md/--xml/--files` output), `recall get <path|#docid>[:line]` with fuzzy suggestions, `recall multi-get` (glob or comma-separated list), `recall context add/list/rm/check`, `recall status`, `recall doctor`.
- Markdown chunking (`internal/chunk`): qmd-inspired break-point scoring with distance penalty `finalScore = baseScore × (1 − (distance/window)² × 0.7)`, 900-token target, 200-token search window, 15% overlap, full code-fence protection (never splits inside ``` / ~~~) (Phase R2).
- `chunks` table + index integration: inserts/updates re-chunk the document, store `content_hash` per chunk for later incremental re-embedding. `recall status` reports chunk count.
- Vector storage via `sqlite-vec`: `chunk_embeddings` virtual table (`vec0(chunk_id, embedding float[300])`); `sqlite_vec.Auto()` registered before any `sql.Open` (Phase R3.1).
- `internal/embed` package: `Embedder` interface (`Embed`, `EmbedSingle`, `Dimensions`, `ModelName`, `Close`), qmd-compatible prompt formatters, deterministic `MockEmbedder` shared by every test, LRU `QueryCache` (32 entries) for repeat queries, model download with SHA-256 verify and progress callback (Phase R3.2 / R3.4 / R3.10).
- Local GGUF backend (`internal/embed/local.go`) wrapping `godeps/gollama` behind `embed_llama` build tag — default builds use the stub which returns a clear "rebuild with -tags sqlite_fts5,embed_llama" error (Phase R3.3).
- `recall embed [-f]`: incremental — only chunks without an embedding row are processed; force mode drops all vectors and re-embeds; metadata `embedding_model` is checked between runs and a model change requires `-f` (Phase R3.5).
- `recall vsearch`: KNN cosine similarity (1 / (1 + distance)) via `sqlite_vec.SerializeFloat32`, with all the search formatting flags. Lazy: BM25-only commands never load the model (Phase R3.6 / R3.9).
- `recall query`: parallel BM25 + vector via goroutines, RRF fusion (k=60), top-rank bonus (+0.05 / +0.02 for #1 / #2-3), adaptive min-score floor (40% of top), `--explain` trace; degrades gracefully to BM25 when no embeddings exist or the model is unavailable (Phase R3.7).
- `recall models list/download/path`.
- Public Go API in `pkg/recall`: `NewEngine`, `AddCollection`, `Index`, `Embed`, `SearchBM25`, `SearchVector`, `SearchHybrid`, `Get`, `MultiGet`, `AddContext`, `ListContexts`, `Close` — what brain will import.
- Optional API embedding fallback (Phase R3b, opt-in only): `RECALL_EMBED_PROVIDER=openai|voyage` reaches `internal/embed/api.go`, which speaks the OpenAI (`text-embedding-3-small`, default 300 dims) or Voyage (`voyage-3-lite`) embeddings endpoint. Batches of 100, exponential backoff with retry on 429 / 5xx, model label written to metadata so cross-provider switches require `recall embed -f`. Default remains `local`; the CLI never proposes the API path.
- `recall cleanup`: drops orphan chunks, stale `chunk_embeddings` (vec0 has no FK so re-indexes leave these behind), runs SQLite VACUUM, prints before/after DB size and reclaimed bytes (Phase R4.1).
- `recall version`: shows version, commit, build date, Go version, OS/arch — populated from ldflags via `Makefile` and `.goreleaser.yaml`, with a `runtime/debug.BuildInfo` fallback for `go install` builds (Phase R4.2).
- Every CLI subcommand is now implemented — no remaining "not implemented yet" stubs.
- Code-aware chunking (Phase R2b): default glob expanded to include `.go .ts .tsx .js .jsx .py .java .rs .rb .php .c .cpp .h .hpp .cs .swift .kt .scala` plus common config formats. `internal/chunk/chunk_code.go` parses Go / Python / TypeScript / JavaScript / Java / Rust with `github.com/smacker/go-tree-sitter` and chunks at function / method / class / impl / import boundaries. Functions exceeding 900 tokens are split internally via blank-line scoring. Unsupported languages fall back to the markdown chunker silently — no errors.
- `internal/chunk/strategy.go::ChunkFile` routes per file: `auto` (default) picks AST for code with a supported grammar, regex for everything else; `regex` and `ast` force one path. `--chunk-strategy auto|regex|ast` flag on `recall embed` (with `-f` re-chunks every doc) and `recall query` (informational).
- Per-language title extraction in `ExtractTitle`: Go shows `package X — FirstExportedSymbol`, Python uses the first `class` / `def`, TypeScript / JavaScript the first `export …`, Java `public class X`, Rust the first `pub fn|struct|enum|trait|mod`. Markdown / text continues to use the H1-or-filename rule.
- Multi-collection search: `-c repo1,repo2,repo3` filters BM25, vector, and hybrid queries to a comma-separated list. Search results always show `{collection}/{path}` so the citation format works for the "many repos" use case.
- Makefile that passes the `sqlite_fts5` build tag required by `mattn/go-sqlite3`.
