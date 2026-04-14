package recall

// This file is the public re-export layer for recall's internal embedder
// package. External consumers (in particular brain) can't reach
// `internal/embed` directly — Go forbids cross-module imports of
// `internal/...`. Everything a library user needs to construct, configure,
// and drive an embedder lives below as a thin alias / wrapper.
//
// Behaviour is NOT reimplemented here; these are re-exports of the
// internal/embed API. Adding or widening a symbol here is an API change
// and requires a minor bump (pre-1.0 we're permissive, but note it in
// CHANGELOG).

import (
	"fmt"
	"os"

	"github.com/ugurcan-aytar/recall/internal/embed"
	"github.com/ugurcan-aytar/recall/internal/store"
)

// Embedder is the contract every embedding backend must satisfy.
// See the Go doc on the underlying type for the full interface.
type Embedder = embed.Embedder

// APIProvider selects the remote embedding backend. The empty string
// means "use the local GGUF embedder".
type APIProvider = embed.APIProvider

// Provider constants — pass the right value to [APIEmbedderOptions.Provider]
// or check the return of [ResolveAPIProvider].
const (
	ProviderLocal  = embed.ProviderLocal
	ProviderOpenAI = embed.ProviderOpenAI
	ProviderVoyage = embed.ProviderVoyage
)

// Model defaults and batching constants.
const (
	DefaultOpenAIModel    = embed.DefaultOpenAIModel
	DefaultVoyageModel    = embed.DefaultVoyageModel
	APIBatchSize          = embed.APIBatchSize
	MaxAPIRetries         = embed.MaxAPIRetries
	DefaultModelName      = embed.DefaultModelName
	DefaultModelURL       = embed.DefaultModelURL
	DefaultQueryCacheSize = embed.DefaultQueryCacheSize
)

// EmbeddingDimensions is the fixed vec0 width recall persists to SQLite.
// Any Embedder handed to an Engine method MUST return vectors of this
// length, or operations will return a dim-mismatch error.
const EmbeddingDimensions = store.EmbeddingDimensions

// APIEmbedderOptions configures [NewAPIEmbedder]. See the underlying
// internal type for field semantics.
type APIEmbedderOptions = embed.APIEmbedderOptions

// LocalEmbedderOptions configures [NewLocalEmbedder].
type LocalEmbedderOptions = embed.LocalEmbedderOptions

// DownloadOptions configures [DownloadModel].
type DownloadOptions = embed.DownloadOptions

// LocalModel is a discovered GGUF file under [ModelsDir].
type LocalModel = embed.LocalModel

// MockEmbedder is a deterministic hash-based Embedder for tests. Library
// consumers should use this (not a real GGUF model) in their unit tests.
type MockEmbedder = embed.MockEmbedder

// QueryCache is an LRU cache for query embeddings, keyed by formatted
// prompt text. Useful when a consumer (brain, typically) reruns the same
// query across multiple retrieval calls in a chat session.
type QueryCache = embed.QueryCache

// ErrLocalEmbedderNotCompiled is returned by [NewLocalEmbedder] in builds
// without the `embed_llama` tag. Library consumers can branch on this
// error to fall back to an API embedder or print a friendlier message.
var ErrLocalEmbedderNotCompiled = embed.ErrLocalEmbedderNotCompiled

// NewAPIEmbedder constructs an HTTP-backed embedder (OpenAI or Voyage).
func NewAPIEmbedder(opts APIEmbedderOptions) (Embedder, error) {
	return embed.NewAPIEmbedder(opts)
}

// NewLocalEmbedder constructs a GGUF-backed in-process embedder. Returns
// [ErrLocalEmbedderNotCompiled] on default builds (no `embed_llama` tag).
func NewLocalEmbedder(opts LocalEmbedderOptions) (Embedder, error) {
	return embed.NewLocalEmbedder(opts)
}

// NewMockEmbedder returns a deterministic test embedder that produces
// dims-length vectors from FNV hashes of the input text.
func NewMockEmbedder(dims int) *MockEmbedder {
	return embed.NewMockEmbedder(dims)
}

// NewQueryCache returns an LRU cache with the given capacity. Pass 0 for
// [DefaultQueryCacheSize].
func NewQueryCache(capacity int) *QueryCache {
	return embed.NewQueryCache(capacity)
}

// LocalEmbedderAvailable reports whether this binary was built with the
// `embed_llama` tag and can create a local GGUF embedder.
func LocalEmbedderAvailable() bool {
	return embed.LocalEmbedderAvailable()
}

// ResolveAPIProvider inspects $RECALL_EMBED_PROVIDER and returns the
// selected provider. Returns [ProviderLocal] when the env var is unset,
// empty, or an unknown value.
func ResolveAPIProvider() APIProvider {
	return embed.ResolveAPIProvider()
}

// ResolveModelPath joins name against [ModelsDir] and returns the absolute
// path. The file may or may not exist on disk; use [DownloadModel] or
// check with os.Stat.
func ResolveModelPath(name string) (string, error) {
	return embed.ResolveModelPath(name)
}

// ModelsDir returns the GGUF model directory, honouring $RECALL_MODELS_DIR.
// Defaults to ~/.recall/models/.
func ModelsDir() (string, error) {
	return embed.ModelsDir()
}

// ListLocalModels returns every .gguf file under [ModelsDir].
func ListLocalModels() ([]LocalModel, error) {
	return embed.ListLocalModels()
}

// DownloadModel fetches a GGUF from HuggingFace into [ModelsDir]. Idempotent:
// skips the download if the file already exists and its SHA matches.
func DownloadModel(opts DownloadOptions) (string, error) {
	return embed.DownloadModel(opts)
}

// FormatQuery prefixes a query with the nomic-embed-text-v1.5 task marker.
// Engine.SearchVector / Engine.SearchHybrid apply this internally; direct
// callers of an Embedder.EmbedSingle should use it themselves.
func FormatQuery(query string) string {
	return embed.FormatQuery(query)
}

// FormatDocument prefixes a document chunk for embedding. Applied by the
// engine when embedding chunks; direct callers should use it themselves.
func FormatDocument(title, content string) string {
	return embed.FormatDocument(title, content)
}

// ResolveEmbedder builds the Embedder implied by the current environment,
// mirroring what `recall embed` / `recall vsearch` / `recall query` do
// internally. The selection order is:
//
//  1. If $RECALL_EMBED_PROVIDER is "openai" or "voyage", return
//     [NewAPIEmbedder] for that provider. The API key is read from
//     $OPENAI_API_KEY or $VOYAGE_API_KEY respectively.
//  2. Otherwise, return [NewLocalEmbedder] rooted at the default model
//     under [ModelsDir]. When the binary lacks the `embed_llama` tag,
//     returns [ErrLocalEmbedderNotCompiled].
//
// Callers wanting a custom provider / model / dimensions should skip this
// helper and call [NewAPIEmbedder] / [NewLocalEmbedder] directly.
func ResolveEmbedder() (Embedder, error) {
	if provider := ResolveAPIProvider(); provider != ProviderLocal {
		return NewAPIEmbedder(APIEmbedderOptions{Provider: provider})
	}
	if !LocalEmbedderAvailable() {
		return nil, ErrLocalEmbedderNotCompiled
	}
	modelPath, err := ResolveModelPath(DefaultModelName)
	if err != nil {
		return nil, err
	}
	if _, err := os.Stat(modelPath); err != nil {
		return nil, fmt.Errorf(
			"embedding model not found at %s — run `recall models download`: %w",
			modelPath, err,
		)
	}
	return NewLocalEmbedder(LocalEmbedderOptions{ModelPath: modelPath})
}
