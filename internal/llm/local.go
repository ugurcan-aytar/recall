//go:build embed_llama

// Local GGUF generation backend wrapping godeps/gollama. Compiled in
// only when the `embed_llama` build tag is set, because it requires
// libbinding.a (see CLAUDE.md "Build" section). Default builds use
// local_stub.go which returns a clear "not compiled in" error.
//
// Generators are deliberately single-threaded for now: query expansion
// / HyDE / reranking each fire one Generate call per `recall query`,
// so the worker-pool optimisation that pays off for embedding (where
// you process hundreds of chunks per `recall embed`) wouldn't move
// the needle here.

package llm

import (
	"errors"
	"fmt"
	"sync"

	llama "github.com/godeps/gollama"
)

// LocalGeneratorOptions configures the GGUF backend.
type LocalGeneratorOptions struct {
	ModelPath string // absolute path to the .gguf file
	Threads   int    // 0 ⇒ gollama default
	Context   int    // 0 ⇒ 4096 (generation needs more context than embedding)
}

type localGenerator struct {
	model     *llama.Model
	ctx       *llama.Context
	modelName string

	mu     sync.Mutex
	closed bool
}

// NewLocalGenerator loads the GGUF model at opts.ModelPath. The first
// call is slow (mmap + warmup); subsequent Generate calls reuse the
// loaded model.
func NewLocalGenerator(opts LocalGeneratorOptions) (Generator, error) {
	if opts.ModelPath == "" {
		return nil, errors.New("local generator: model path is required")
	}

	model, err := llama.LoadModel(opts.ModelPath)
	if err != nil {
		return nil, fmt.Errorf("load gguf model %s: %w", opts.ModelPath, err)
	}

	ctxSize := opts.Context
	if ctxSize <= 0 {
		ctxSize = 4096
	}
	ctxOpts := []llama.ContextOption{
		llama.WithContext(ctxSize),
		llama.WithBatch(ctxSize),
	}
	if opts.Threads > 0 {
		ctxOpts = append(ctxOpts, llama.WithThreads(opts.Threads))
	}

	ctx, err := model.NewContext(ctxOpts...)
	if err != nil {
		_ = model.Close()
		return nil, fmt.Errorf("new context: %w", err)
	}

	return &localGenerator{
		model:     model,
		ctx:       ctx,
		modelName: deriveModelName(opts.ModelPath),
	}, nil
}

// LocalGeneratorAvailable reports whether this binary has the local
// generation backend compiled in.
func LocalGeneratorAvailable() bool { return true }

func (g *localGenerator) Generate(prompt string, opts ...GenerateOption) (string, error) {
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.closed {
		return "", errors.New("local generator is closed")
	}

	cfg := GenerateOptions{MaxTokens: 256}
	for _, o := range opts {
		o(&cfg)
	}

	out, err := g.ctx.Generate(prompt, llama.WithMaxTokens(cfg.MaxTokens))
	if err != nil {
		return "", fmt.Errorf("generate: %w", err)
	}
	return out, nil
}

func (g *localGenerator) ModelName() string { return g.modelName }

func (g *localGenerator) Close() error {
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.closed {
		return nil
	}
	g.closed = true
	if g.ctx != nil {
		_ = g.ctx.Close()
	}
	if g.model != nil {
		return g.model.Close()
	}
	return nil
}

// deriveModelName turns "/x/y/qmd-query-expansion-1.7B-q4_k_m.gguf"
// into "qmd-query-expansion-1.7B-q4_k_m".
func deriveModelName(path string) string {
	base := path
	for i := len(path) - 1; i >= 0; i-- {
		if path[i] == '/' || path[i] == '\\' {
			base = path[i+1:]
			break
		}
	}
	if dot := lastDot(base); dot >= 0 {
		return base[:dot]
	}
	return base
}

func lastDot(s string) int {
	for i := len(s) - 1; i >= 0; i-- {
		if s[i] == '.' {
			return i
		}
	}
	return -1
}
