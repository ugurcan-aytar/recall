// Package embed produces vector embeddings for chunks and queries.
//
// The default path runs a local GGUF model in-process (via go-llama.cpp).
// That code lives behind the `embed_llama` build tag because it requires
// libbinding.a to be built first — see CLAUDE.md for the build steps. The
// stub file ships with the default build and returns a clear "not compiled
// in" error, so `go build` works out of the box.
//
// All tests in this project use MockEmbedder (defined in embed_test.go).
// Real GGUF models are never downloaded from tests.
package embed

import "errors"

// Embedder is the contract every embedding backend must satisfy.
//
// Implementations MUST be safe for concurrent EmbedSingle / Embed calls
// from multiple goroutines. Dimensions and ModelName are immutable for
// the lifetime of the embedder.
type Embedder interface {
	// Embed returns one vector per input text, in order. Input and output
	// slices have equal length.
	Embed(texts []string) ([][]float32, error)

	// EmbedSingle is a convenience for one-at-a-time callers.
	EmbedSingle(text string) ([]float32, error)

	// Dimensions reports the length of every returned vector.
	Dimensions() int

	// ModelName is a stable identifier (e.g. "nomic-embed-text-v1.5.Q8_0")
	// stored in the DB's metadata table so recall can detect when the
	// model has changed and existing vectors became stale.
	ModelName() string

	// Close releases any resources (mmap'd model, contexts, etc.). After
	// Close, further Embed calls return an error.
	Close() error
}

// ErrLocalEmbedderNotCompiled is returned by [NewLocalEmbedder] in builds
// without the `embed_llama` tag. Defined here (not in local_stub.go) so
// callers can compare against it regardless of which backend was linked.
var ErrLocalEmbedderNotCompiled = errors.New(
	"local GGUF embedding is not compiled into this binary; rebuild with " +
		"`-tags sqlite_fts5,embed_llama` after building libbinding.a (see CLAUDE.md)",
)

// FormatQuery returns the embedding prompt for a user query.
//
// Uses nomic-embed-text-v1.5's required task prefix ("search_query: ")
// per the model card on HuggingFace. Without this prefix the model
// emits valid but lower-quality vectors because it was fine-tuned with
// asymmetric query/document prompts.
func FormatQuery(query string) string {
	return "search_query: " + query
}

// FormatDocument returns the embedding prompt for a chunk of document
// content. Uses nomic's "search_document: " prefix; title is kept as a
// short prefix so very generic chunks still ground to the source doc.
func FormatDocument(title, content string) string {
	if title == "" {
		return "search_document: " + content
	}
	return "search_document: " + title + " — " + content
}
