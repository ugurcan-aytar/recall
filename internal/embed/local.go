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
	Workers   int    // worker pool size; 0/1 ⇒ single shared model + mutex (current default)
}

// MaxLocalWorkers caps the number of GGUF model instances we'll load
// in parallel. Each instance mmaps the full ~146 MB nomic file plus
// its own context (KV cache); a hard upper bound prevents a typo'd
// RECALL_EMBED_WORKERS=64 from OOM-ing the user's laptop.
const MaxLocalWorkers = 8

// localEmbedder owns one or more llama.Model + Context pairs and
// dispatches Embed() across them via a buffered semaphore. With
// Workers <= 1 it degenerates to a single model + sync.Mutex —
// identical to the pre-parallel implementation, no extra RAM.
type localEmbedder struct {
	pool      chan *workerCtx // semaphore + worker handle queue
	all       []*workerCtx    // every owned context, for Close()
	modelName string
	family    PromptFamily
	dims      int

	mu     sync.Mutex
	closed bool
}

// workerCtx pairs a Model with its Context. Each worker owns one
// pair so concurrent EmbedSingle calls don't trip gollama's
// single-context invariants.
type workerCtx struct {
	model *llama.Model
	ctx   *llama.Context
}

// NewLocalEmbedder loads opts.ModelPath. With Workers <= 1 it loads
// a single model + context (the v0.1 default — minimum RAM, single-
// threaded inference). With Workers > 1 it loads N independent
// model+context pairs so Embed() can dispatch chunks concurrently
// across them. Each extra worker costs ~146 MB (nomic Q8) + the
// context's KV cache.
func NewLocalEmbedder(opts LocalEmbedderOptions) (Embedder, error) {
	if opts.ModelPath == "" {
		return nil, errors.New("local embedder: model path is required")
	}

	workers := opts.Workers
	if workers <= 0 {
		workers = 1
	}
	if workers > MaxLocalWorkers {
		workers = MaxLocalWorkers
	}

	ctxSize := opts.Context
	if ctxSize <= 0 {
		ctxSize = 2048
	}
	buildContextOpts := func() []llama.ContextOption {
		o := []llama.ContextOption{
			llama.WithEmbeddings(),
			llama.WithContext(ctxSize),
			// Force one sequence per context. gollama otherwise defaults
			// to n_seq_max=8, splitting n_ctx (2048) across 8 sequences =
			// 256 tokens per chunk — SIGTRAPs on any chunk over ~250
			// tokens.
			llama.WithParallel(1),
			// Match batch to context so a single chunk fits one
			// llama_batch.
			llama.WithBatch(ctxSize),
		}
		if opts.Threads > 0 {
			o = append(o, llama.WithThreads(opts.Threads))
		}
		return o
	}

	pool := make(chan *workerCtx, workers)
	all := make([]*workerCtx, 0, workers)
	var dims int
	for i := 0; i < workers; i++ {
		model, err := llama.LoadModel(opts.ModelPath)
		if err != nil {
			closeWorkers(all)
			return nil, fmt.Errorf("load gguf model %s (worker %d): %w", opts.ModelPath, i, err)
		}
		ctx, err := model.NewContext(buildContextOpts()...)
		if err != nil {
			_ = model.Close()
			closeWorkers(all)
			return nil, fmt.Errorf("new context (worker %d): %w", i, err)
		}
		// Probe the first worker's dim. Subsequent workers must produce
		// the same width; the model is identical so this is more of a
		// belt-and-braces invariant than a real check.
		probe, err := ctx.GetEmbeddings("hello")
		if err != nil {
			_ = ctx.Close()
			_ = model.Close()
			closeWorkers(all)
			return nil, fmt.Errorf("probe embedding (worker %d): %w", i, err)
		}
		if i == 0 {
			dims = len(probe)
		} else if len(probe) != dims {
			_ = ctx.Close()
			_ = model.Close()
			closeWorkers(all)
			return nil, fmt.Errorf("worker %d returned %d-dim probe (expected %d)", i, len(probe), dims)
		}
		w := &workerCtx{model: model, ctx: ctx}
		all = append(all, w)
		pool <- w
	}

	name := deriveModelName(opts.ModelPath)
	return &localEmbedder{
		pool:      pool,
		all:       all,
		modelName: name,
		family:    ResolveFamily(name),
		dims:      dims,
	}, nil
}

func closeWorkers(ws []*workerCtx) {
	for _, w := range ws {
		if w.ctx != nil {
			_ = w.ctx.Close()
		}
		if w.model != nil {
			_ = w.model.Close()
		}
	}
}

// LocalEmbedderAvailable reports whether this binary has the local backend.
func LocalEmbedderAvailable() bool { return true }

// Embed dispatches one text per worker via the pool semaphore.
// gollama's GetEmbeddingsBatch packs all tokens into a single
// llama_batch which a 2048-ctx BERT-class encoder overflows on the
// first long chunk; we deliberately stay one-text-per-call and rely
// on the worker pool for concurrency instead.
func (e *localEmbedder) Embed(texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	if e.isClosed() {
		return nil, errors.New("local embedder is closed")
	}
	out := make([][]float32, len(texts))
	if cap(e.pool) <= 1 {
		// Single-worker fast path: no goroutines, no channel ops.
		w := <-e.pool
		defer func() { e.pool <- w }()
		for i, t := range texts {
			v, err := w.ctx.GetEmbeddings(t)
			if err != nil {
				return nil, fmt.Errorf("embed: %w", err)
			}
			out[i] = v
		}
		return out, nil
	}

	errs := make(chan error, len(texts))
	var wg sync.WaitGroup
	wg.Add(len(texts))
	for i, t := range texts {
		i, t := i, t
		go func() {
			defer wg.Done()
			w := <-e.pool
			defer func() { e.pool <- w }()
			v, err := w.ctx.GetEmbeddings(t)
			if err != nil {
				errs <- fmt.Errorf("embed (text %d): %w", i, err)
				return
			}
			out[i] = v
		}()
	}
	wg.Wait()
	close(errs)
	if err := <-errs; err != nil {
		return nil, err
	}
	return out, nil
}

func (e *localEmbedder) EmbedSingle(text string) ([]float32, error) {
	if e.isClosed() {
		return nil, errors.New("local embedder is closed")
	}
	w := <-e.pool
	defer func() { e.pool <- w }()
	v, err := w.ctx.GetEmbeddings(text)
	if err != nil {
		return nil, fmt.Errorf("embed: %w", err)
	}
	return v, nil
}

func (e *localEmbedder) isClosed() bool {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.closed
}

func (e *localEmbedder) Dimensions() int       { return e.dims }
func (e *localEmbedder) ModelName() string     { return e.modelName }
func (e *localEmbedder) Family() PromptFamily  { return e.family }

func (e *localEmbedder) Close() error {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return nil
	}
	e.closed = true
	closeWorkers(e.all)
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
