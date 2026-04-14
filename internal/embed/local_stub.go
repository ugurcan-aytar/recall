//go:build !embed_llama

package embed

import "fmt"

// LocalEmbedderOptions is exported for API symmetry with the embed_llama
// build. In the stub build it is unused.
type LocalEmbedderOptions struct {
	ModelPath string
	Threads   int
	Context   int
}

// NewLocalEmbedder is the stub default. It always returns
// ErrLocalEmbedderNotCompiled. Tests should use [NewMockEmbedder] instead.
func NewLocalEmbedder(opts LocalEmbedderOptions) (Embedder, error) {
	return nil, fmt.Errorf("%w (model=%q)", ErrLocalEmbedderNotCompiled, opts.ModelPath)
}

// LocalEmbedderAvailable reports whether this binary has the local GGUF
// backend compiled in. Callers can branch on this to print friendlier
// errors instead of opening a model that doesn't exist.
func LocalEmbedderAvailable() bool { return false }
