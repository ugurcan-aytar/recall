//go:build !embed_llama

package llm

import "fmt"

// LocalGeneratorOptions is exported for API symmetry with the
// embed_llama build. In the stub build the only field that gets
// echoed back is ModelPath (in the error message); the rest exist
// so callers can build an options literal without conditional code
// per build tag.
type LocalGeneratorOptions struct {
	ModelPath string
	Threads   int
	Context   int
}

// NewLocalGenerator is the stub default. It always returns
// ErrLocalGeneratorNotCompiled. Tests should use [MockGenerator]
// instead.
func NewLocalGenerator(opts LocalGeneratorOptions) (Generator, error) {
	return nil, fmt.Errorf("%w (model=%q)", ErrLocalGeneratorNotCompiled, opts.ModelPath)
}

// LocalGeneratorAvailable reports whether this binary has the local
// GGUF generation backend compiled in. Callers branch on this to
// print friendlier errors instead of opening a model that doesn't
// exist.
func LocalGeneratorAvailable() bool { return false }
