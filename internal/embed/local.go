//go:build embed_llama

// Local GGUF embedding backend wrapping godeps/gollama. Compiled in only
// when the `embed_llama` build tag is set, because it requires
// libbinding.a (see CLAUDE.md "Build" section).
//
// The default build uses local_stub.go which returns a clear "not compiled
// in" error. All tests in this project use MockEmbedder; this file is never
// exercised by go test.

package embed

import (
	"errors"
	"fmt"
	"sync"

	llama "github.com/godeps/gollama"
)

// LocalEmbedderOptions configures the GGUF backend.
type LocalEmbedderOptions struct {
	ModelPath string // absolute path to the .gguf file
	Threads   int    // 0 ⇒ gollama default
	Context   int    // 0 ⇒ 2048
}

type localEmbedder struct {
	mu        sync.Mutex
	model     *llama.Model
	ctx       *llama.Context
	modelName string
	dims      int
	closed    bool
}

// NewLocalEmbedder loads the GGUF model at opts.ModelPath. The first call
// is slow (mmap + warmup); subsequent Embed calls reuse the loaded model.
func NewLocalEmbedder(opts LocalEmbedderOptions) (Embedder, error) {
	if opts.ModelPath == "" {
		return nil, errors.New("local embedder: model path is required")
	}

	model, err := llama.LoadModel(opts.ModelPath)
	if err != nil {
		return nil, fmt.Errorf("load gguf model %s: %w", opts.ModelPath, err)
	}

	ctxSize := opts.Context
	if ctxSize <= 0 {
		ctxSize = 2048
	}
	ctxOpts := []llama.ContextOption{
		llama.WithEmbeddings(),
		llama.WithContext(ctxSize),
		// Force one sequence per context. gollama otherwise defaults to
		// n_seq_max=8, splitting n_ctx (2048) across 8 sequences = 256
		// tokens per chunk — SIGTRAPs on any chunk over ~250 tokens.
		llama.WithParallel(1),
		// Match batch to context so a single chunk fits one llama_batch.
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

	// Probe dim with a one-token throwaway so callers can check Dimensions()
	// before any user-facing embed call.
	probe, err := ctx.GetEmbeddings("hello")
	if err != nil {
		_ = ctx.Close()
		_ = model.Close()
		return nil, fmt.Errorf("probe embedding: %w", err)
	}

	return &localEmbedder{
		model:     model,
		ctx:       ctx,
		modelName: deriveModelName(opts.ModelPath),
		dims:      len(probe),
	}, nil
}

// LocalEmbedderAvailable reports whether this binary has the local backend.
func LocalEmbedderAvailable() bool { return true }

func (e *localEmbedder) Embed(texts []string) ([][]float32, error) {
	// gollama's GetEmbeddingsBatch packs all input tokens into a single
	// llama_batch, which a small BERT-like embedder (nomic-embed at
	// 2048 ctx) exceeds when given many chunks at once. One-at-a-time
	// is plenty fast for our scale and avoids "llama_batch size
	// exceeded" / "n_ubatch >= n_tokens" aborts.
	out := make([][]float32, len(texts))
	for i, t := range texts {
		v, err := e.EmbedSingle(t)
		if err != nil {
			return nil, err
		}
		out[i] = v
	}
	return out, nil
}

func (e *localEmbedder) EmbedSingle(text string) ([]float32, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return nil, errors.New("local embedder is closed")
	}
	v, err := e.ctx.GetEmbeddings(text)
	if err != nil {
		return nil, fmt.Errorf("embed: %w", err)
	}
	return v, nil
}

func (e *localEmbedder) Dimensions() int   { return e.dims }
func (e *localEmbedder) ModelName() string { return e.modelName }

func (e *localEmbedder) Close() error {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return nil
	}
	e.closed = true
	if e.ctx != nil {
		_ = e.ctx.Close()
	}
	if e.model != nil {
		return e.model.Close()
	}
	return nil
}

// deriveModelName turns "/x/y/nomic-embed-text-v1.5.Q8_0.gguf" into
// "nomic-embed-text-v1.5.Q8_0".
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
