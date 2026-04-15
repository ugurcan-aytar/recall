package commands

import (
	"errors"
	"fmt"
	"os"
	"strconv"

	"github.com/ugurcan-aytar/recall/internal/embed"
)

// embedWorkersOverride is set by `recall embed --workers N` so the
// flag wins over the env var. Zero means "consult $RECALL_EMBED_WORKERS,
// then fall back to the backend default".
var embedWorkersOverride int

// embedderOverride lets tests inject a pre-built Embedder so they don't
// need libbinding.a. When non-nil, openEmbedder returns it instead of
// building a real local embedder.
var embedderOverride embed.Embedder

// queryEmbedCache lives for the lifetime of the process. Since each CLI
// invocation is a one-shot process, the cache only matters for callers
// that share an Embedder across many queries (e.g. brain).
var queryEmbedCache = embed.NewQueryCache(0)

// openEmbedder returns an Embedder for the active configuration. Honours
// the test-only embedderOverride first; otherwise:
//
//   - if $RECALL_EMBED_PROVIDER is set to "openai" or "voyage", returns
//     an [embed.NewAPIEmbedder] for that provider;
//   - otherwise (the default in every reachable code path) returns the
//     local GGUF embedder rooted at $RECALL_MODELS_DIR.
//
// This is the lazy-load entry point: BM25-only commands MUST NOT call
// this. Only `recall vsearch`, `recall query`, and `recall embed` do.
func openEmbedder() (embed.Embedder, error) {
	if embedderOverride != nil {
		return embedderOverride, nil
	}
	workers := resolveWorkers()
	if provider := embed.ResolveAPIProvider(); provider != embed.ProviderLocal {
		return embed.NewAPIEmbedder(embed.APIEmbedderOptions{
			Provider: provider,
			Workers:  workers,
		})
	}
	if !embed.LocalEmbedderAvailable() {
		return nil, embed.ErrLocalEmbedderNotCompiled
	}
	modelPath, err := embed.ResolveActiveModelPath()
	if err != nil {
		return nil, err
	}
	if _, err := os.Stat(modelPath); err != nil {
		return nil, fmt.Errorf(
			"embedding model not found at %s — run `recall models download` or set RECALL_EMBED_MODEL: %w",
			modelPath, err,
		)
	}
	return embed.NewLocalEmbedder(embed.LocalEmbedderOptions{
		ModelPath: modelPath,
		Workers:   workers,
	})
}

// resolveWorkers picks the worker count, in priority order:
//
//   - explicit --workers flag (embedWorkersOverride > 0)
//   - $RECALL_EMBED_WORKERS env var
//   - 0 (backend's own default — single worker for both local and API)
//
// The local backend caps at MaxLocalWorkers and the API backend at
// MaxAPIWorkers internally; this helper just returns whatever the user
// asked for and lets the backend clamp.
func resolveWorkers() int {
	if embedWorkersOverride > 0 {
		return embedWorkersOverride
	}
	if raw := os.Getenv("RECALL_EMBED_WORKERS"); raw != "" {
		if n, err := strconv.Atoi(raw); err == nil && n > 0 {
			return n
		}
	}
	return 0
}

// embedQueryCached wraps EmbedSingle with the in-process LRU cache and
// the active embedder's family-specific query prompt format (nomic,
// Gemma, Qwen3, or raw depending on the model).
func embedQueryCached(e embed.Embedder, query string) ([]float32, error) {
	prompt := embed.FormatQueryFor(e.Family(), query)
	if v, ok := queryEmbedCache.Get(prompt); ok {
		return v, nil
	}
	v, err := e.EmbedSingle(prompt)
	if err != nil {
		return nil, err
	}
	queryEmbedCache.Put(prompt, v)
	return v, nil
}

// SetEmbedderOverride is exported for cross-package tests that need to
// inject a MockEmbedder into the commands package without resorting to
// internal-import gymnastics. Pass nil to clear.
func SetEmbedderOverride(e embed.Embedder) { embedderOverride = e }

// resetQueryCacheForTest is package-internal so tests in this package can
// clear cached state between cases.
func resetQueryCacheForTest() { queryEmbedCache = embed.NewQueryCache(0) }

// ensureNoNilEmbedderShadow prevents the linter from flagging the unused
// errors import in builds where the openEmbedder fallback path is dead.
var _ = errors.New
