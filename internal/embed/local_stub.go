//go:build !embed_llama

package embed

import "fmt"

// LocalEmbedderOptions is exported for API symmetry with the
// embed_llama build. In the stub build the only field that ever gets
// echoed back is ModelPath (in the error message); the rest exist so
// callers can construct an options literal without conditional code
// per build tag.
type LocalEmbedderOptions struct {
	ModelPath string
	Threads   int
	Context   int
	Workers   int
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

// Compile-time check that MockEmbedder satisfies the Family method on
// stub builds too — the real localEmbedder lives only when embed_llama
// is on, so without this the interface would look incomplete to any
// cross-module consumer that does `var _ Embedder = (*MockEmbedder)(nil)`
// in a stub-build test.
var _ PromptFamily = FamilyGeneric
