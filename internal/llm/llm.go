// Package llm wraps a local GGUF generation model. recall's primary
// path is BM25 + vector + RRF — pure retrieval, no LLM. But three
// optional features (query expansion, cross-encoder reranking, HyDE)
// need a small text-gen model on the side. This package owns the
// generation surface so callers don't reach into gollama directly.
//
// Like the embedder, the local backend lives behind the `embed_llama`
// build tag. Default builds use the stub which returns a clear
// "not compiled in" error so `go build` and `go test` keep working
// on every machine. All tests use [MockGenerator].
package llm

import "errors"

// Generator is the contract every text-generation backend satisfies.
//
// Implementations MUST be safe for concurrent Generate calls — they
// own whatever pool of model instances they need internally.
// ModelName is immutable for the lifetime of the generator.
type Generator interface {
	// Generate runs the model on prompt and returns the generated
	// text. Options control sampling behaviour; reasonable defaults
	// kick in when no options are passed.
	Generate(prompt string, opts ...GenerateOption) (string, error)

	// ModelName is a stable identifier ("qmd-query-expansion-1.7B-q4_k_m")
	// surfaced in `recall doctor` and in error messages.
	ModelName() string

	// Close releases the loaded model and any contexts. After Close,
	// further Generate calls return an error.
	Close() error
}

// GenerateOptions configures a single Generate call. Callers compose
// them via the WithXxx functional options below.
type GenerateOptions struct {
	// MaxTokens caps generated tokens. 0 ⇒ backend default (256).
	MaxTokens int
}

// GenerateOption mutates GenerateOptions in the functional-options
// pattern recall uses elsewhere.
type GenerateOption func(*GenerateOptions)

// WithMaxTokens caps the response at n tokens.
func WithMaxTokens(n int) GenerateOption {
	return func(o *GenerateOptions) {
		if n > 0 {
			o.MaxTokens = n
		}
	}
}

// ErrLocalGeneratorNotCompiled is returned by [NewLocalGenerator] in
// builds without the `embed_llama` tag. Callers can compare against
// it (errors.Is) to print friendlier guidance instead of cascading
// the cryptic gollama load failure.
var ErrLocalGeneratorNotCompiled = errors.New(
	"local GGUF generation is not compiled into this binary; rebuild with " +
		"`-tags sqlite_fts5,embed_llama` after building libbinding.a (see CLAUDE.md)",
)
