package commands

import (
	"errors"
	"fmt"
	"os"

	"github.com/ugurcan-aytar/recall/internal/embed"
)

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
	if provider := embed.ResolveAPIProvider(); provider != embed.ProviderLocal {
		return embed.NewAPIEmbedder(embed.APIEmbedderOptions{Provider: provider})
	}
	if !embed.LocalEmbedderAvailable() {
		return nil, embed.ErrLocalEmbedderNotCompiled
	}
	modelPath, err := embed.ResolveModelPath(embed.DefaultModelName)
	if err != nil {
		return nil, err
	}
	if _, err := os.Stat(modelPath); err != nil {
		return nil, fmt.Errorf(
			"embedding model not found at %s — run `recall models download`: %w",
			modelPath, err,
		)
	}
	return embed.NewLocalEmbedder(embed.LocalEmbedderOptions{ModelPath: modelPath})
}

// embedQueryCached wraps EmbedSingle with the in-process LRU cache and the
// qmd-compatible "task: search result | query: …" prompt format.
func embedQueryCached(e embed.Embedder, query string) ([]float32, error) {
	prompt := embed.FormatQuery(query)
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
