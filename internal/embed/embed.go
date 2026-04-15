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

import (
	"errors"
	"os"
	"strings"
)

// Embedder is the contract every embedding backend must satisfy.
//
// Implementations MUST be safe for concurrent EmbedSingle / Embed calls
// from multiple goroutines. Dimensions, ModelName, and Family are
// immutable for the lifetime of the embedder.
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

	// Family reports which prompt-format family the underlying model
	// belongs to. Callers should wrap queries / documents with
	// FormatQueryFor / FormatDocumentFor using this value so the model
	// receives the task prefix it was fine-tuned on.
	Family() PromptFamily

	// Close releases any resources (mmap'd model, contexts, etc.). After
	// Close, further Embed calls return an error.
	Close() error
}

// PromptFamily identifies an embedding model's required prompt format.
// Different model families were trained with different task prefixes; a
// nomic-style prefix on a Qwen3-Embedding model (or vice versa) emits
// valid vectors but measurably worse retrieval quality.
type PromptFamily string

const (
	// FamilyNomic formats prompts with "search_query: " / "search_document: "
	// prefixes as described by the nomic-embed-text-v1 / v1.5 model cards.
	FamilyNomic PromptFamily = "nomic"

	// FamilyGemma formats prompts with Google's EmbeddingGemma task
	// markers: "task: search result | query: <q>" for queries and
	// "title: <t> | text: <body>" for passages.
	FamilyGemma PromptFamily = "gemma"

	// FamilyQwen3 uses Qwen3-Embedding's instruction format for queries
	// ("Instruct: <task>\nQuery:<q>") and raw text for passages.
	FamilyQwen3 PromptFamily = "qwen3"

	// FamilyGeneric emits the query or document text unchanged. Correct
	// for API embedders whose HTTP layer adds whatever prefix the model
	// needs, for MockEmbedder in tests, and as a safe fallback when we
	// can't identify the model family from its filename.
	FamilyGeneric PromptFamily = "generic"
)

// DetectFamily guesses the prompt family from a model's filename or
// identifier (e.g. "nomic-embed-text-v1.5.Q8_0", "embeddinggemma-300m",
// "Qwen3-Embedding-0.6B"). Returns FamilyNomic when nothing matches —
// nomic's prefix format is the most forgiving and won't break other
// modern BERT-family embedders.
func DetectFamily(modelName string) PromptFamily {
	name := strings.ToLower(modelName)
	switch {
	case strings.Contains(name, "nomic-embed"), strings.Contains(name, "nomic_embed"):
		return FamilyNomic
	case strings.Contains(name, "embeddinggemma"),
		strings.Contains(name, "embedding-gemma"),
		strings.Contains(name, "embedding_gemma"):
		return FamilyGemma
	case strings.Contains(name, "qwen3-embedding"),
		strings.Contains(name, "qwen3_embedding"),
		strings.Contains(name, "qwen3embedding"):
		return FamilyQwen3
	default:
		return FamilyNomic
	}
}

// ResolveFamily honours the $RECALL_EMBED_PROMPT_FORMAT override before
// falling back to DetectFamily(modelName). Accepted override values are
// "nomic", "gemma" / "embeddinggemma", "qwen" / "qwen3", and
// "generic" / "raw" / "none"; unknown values fall through to detection.
func ResolveFamily(modelName string) PromptFamily {
	if raw := strings.ToLower(strings.TrimSpace(os.Getenv("RECALL_EMBED_PROMPT_FORMAT"))); raw != "" {
		switch raw {
		case "nomic":
			return FamilyNomic
		case "gemma", "embeddinggemma":
			return FamilyGemma
		case "qwen", "qwen3":
			return FamilyQwen3
		case "generic", "raw", "none":
			return FamilyGeneric
		}
	}
	return DetectFamily(modelName)
}

// ErrLocalEmbedderNotCompiled is returned by [NewLocalEmbedder] in builds
// without the `embed_llama` tag. Defined here (not in local_stub.go) so
// callers can compare against it regardless of which backend was linked.
var ErrLocalEmbedderNotCompiled = errors.New(
	"local GGUF embedding is not compiled into this binary; rebuild with " +
		"`-tags sqlite_fts5,embed_llama` after building libbinding.a (see CLAUDE.md)",
)

// FormatQuery returns a nomic-flavoured embedding prompt. Kept for
// callers that predate the family abstraction and for tests. Prefer
// [FormatQueryFor] when the target family is known.
func FormatQuery(query string) string {
	return FormatQueryFor(FamilyNomic, query)
}

// FormatDocument returns a nomic-flavoured document prompt. See
// [FormatQuery] for the compatibility note.
func FormatDocument(title, content string) string {
	return FormatDocumentFor(FamilyNomic, title, content)
}

// FormatQueryFor returns the embedding prompt for a user query in the
// given family's native format. Picking the right family matters: nomic
// prefixes applied to a Qwen3-Embedding model (or vice versa) still
// produce valid vectors but measurably worse retrieval quality.
func FormatQueryFor(family PromptFamily, query string) string {
	switch family {
	case FamilyGemma:
		return "task: search result | query: " + query
	case FamilyQwen3:
		return "Instruct: Given a query, retrieve relevant passages that answer the query\nQuery: " + query
	case FamilyGeneric:
		return query
	case FamilyNomic:
		fallthrough
	default:
		return "search_query: " + query
	}
}

// FormatDocumentFor returns the embedding prompt for a chunk of
// document content in the given family's native format. Title is
// carried through where the family's prompt template has a slot for
// it.
func FormatDocumentFor(family PromptFamily, title, content string) string {
	switch family {
	case FamilyGemma:
		t := title
		if t == "" {
			t = "none"
		}
		return "title: " + t + " | text: " + content
	case FamilyQwen3:
		// Qwen3-Embedding documents go in unwrapped — the task prefix
		// lives on the query side only.
		return content
	case FamilyGeneric:
		return content
	case FamilyNomic:
		fallthrough
	default:
		if title == "" {
			return "search_document: " + content
		}
		return "search_document: " + title + " — " + content
	}
}
