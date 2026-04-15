// Local GGUF embedding backend on top of dianlight/gollama.cpp.
//
// dianlight/gollama.cpp is a purego (no-CGo) wrapper around llama.cpp.
// The package downloads the platform-appropriate llama.cpp shared
// library on first use into ~/.cache/gollama/libs/ (override with
// $GOLLAMA_CACHE_DIR). After that recall talks to the runtime via
// purego + libffi — no static archive, no rpath flags, no libbinding.a
// build step. Compiles unconditionally (no `embed_llama` build tag);
// the binary always ships with local-embedding capability.
//
// Lifecycle: gollama.Backend_init / Backend_free are process-global
// (the underlying llama.cpp API state is too). We call Backend_init
// lazily on the first NewLocalEmbedder, never call Backend_free —
// recall's CLI is one-shot, the OS reclaims everything on exit.

package embed

import (
	"errors"
	"fmt"
	"math"
	"runtime"
	"sync"
	"unsafe"

	gollama "github.com/dianlight/gollama.cpp"

	"github.com/ugurcan-aytar/recall/internal/llamacpp"
)

// LocalEmbedderOptions configures the GGUF backend.
type LocalEmbedderOptions struct {
	ModelPath string // absolute path to the .gguf file
	Threads   int    // 0 ⇒ runtime.NumCPU()
	Context   int    // 0 ⇒ 2048
	Workers   int    // worker pool size; 0/1 ⇒ single context (current default)
}

// MaxLocalWorkers caps how many independent llama.cpp contexts we
// hold open per Embedder. Each context owns its own KV cache; total
// RAM scales linearly. The model weights are mmap'd ONCE and shared
// across all contexts (a big win vs the godeps/gollama pattern,
// which loaded N model copies).
const MaxLocalWorkers = 8

// localEmbedder owns one llama.cpp Model + N Context handles. The
// pool channel doubles as a semaphore: workers grab a context, run
// inference, return it.
type localEmbedder struct {
	model     gollama.LlamaModel
	pool      chan gollama.LlamaContext
	all       []gollama.LlamaContext
	modelName string
	family    PromptFamily
	dims      int

	mu     sync.Mutex
	closed bool
}

// Backend init lives in internal/llamacpp so embed and llm share a
// single sync.Once across the process. The first NewLocalEmbedder
// or NewLocalGenerator call pays the (~100 MB) shared-library
// download cost; subsequent calls return immediately.

// NewLocalEmbedder loads opts.ModelPath. The model weights mmap once
// and are shared across the worker contexts; Workers controls how
// many independent inference contexts we open. With Workers <= 1
// behaviour matches the v0.1 single-context default.
func NewLocalEmbedder(opts LocalEmbedderOptions) (Embedder, error) {
	if opts.ModelPath == "" {
		return nil, errors.New("local embedder: model path is required")
	}
	if err := llamacpp.EnsureBackend(); err != nil {
		return nil, err
	}

	workers := opts.Workers
	if workers <= 0 {
		workers = 1
	}
	if workers > MaxLocalWorkers {
		workers = MaxLocalWorkers
	}

	threads := opts.Threads
	if threads <= 0 {
		threads = runtime.NumCPU()
	}

	ctxSize := opts.Context
	if ctxSize <= 0 {
		ctxSize = 2048
	}

	modelParams := gollama.Model_default_params()
	model, err := gollama.Model_load_from_file(opts.ModelPath, modelParams)
	if err != nil {
		return nil, fmt.Errorf("load gguf model %s: %w", opts.ModelPath, err)
	}

	pool := make(chan gollama.LlamaContext, workers)
	all := make([]gollama.LlamaContext, 0, workers)
	dims := int(gollama.Model_n_embd(model))
	if dims <= 0 {
		gollama.Model_free(model)
		return nil, fmt.Errorf("model %s reports %d embedding dimensions", opts.ModelPath, dims)
	}

	for i := 0; i < workers; i++ {
		ctxParams := gollama.Context_default_params()
		ctxParams.NCtx = uint32(ctxSize)
		// NBatch and NUbatch must match — gollama's old wrapper
		// silently kept NUbatch at the 512 default while NBatch
		// climbed, blowing up on chunks > ~256 tokens. dianlight
		// exposes both fields directly so we set them in lockstep.
		ctxParams.NBatch = uint32(ctxSize)
		ctxParams.NUbatch = uint32(ctxSize)
		// One sequence per context. llama.cpp's old auto-bump of
		// n_parallel to 8 when embeddings were on (which divided
		// n_ctx across 8 sequences and SIGTRAP'd long chunks)
		// doesn't exist in dianlight — but we still pin to 1
		// explicitly so future llama.cpp default changes don't
		// surprise us.
		ctxParams.NSeqMax = 1
		ctxParams.NThreads = int32(threads)
		ctxParams.NThreadsBatch = int32(threads)
		ctxParams.Embeddings = 1

		ctx, err := gollama.Init_from_model(model, ctxParams)
		if err != nil {
			closeContexts(all)
			gollama.Model_free(model)
			return nil, fmt.Errorf("new context (worker %d): %w", i, err)
		}
		all = append(all, ctx)
		pool <- ctx
	}

	name := deriveModelName(opts.ModelPath)
	return &localEmbedder{
		model:     model,
		pool:      pool,
		all:       all,
		modelName: name,
		family:    ResolveFamily(name),
		dims:      dims,
	}, nil
}

// LocalEmbedderAvailable always returns true now that the backend
// downloads itself. The function is kept for API compatibility with
// the v0.1 build-tag stub.
func LocalEmbedderAvailable() bool { return true }

// Embed dispatches one text per worker via the context pool.
// Callers wanting parallelism construct the Embedder with
// LocalEmbedderOptions.Workers > 1; otherwise the channel has a
// single context and Embed is sequential.
func (e *localEmbedder) Embed(texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	if e.isClosed() {
		return nil, errors.New("local embedder is closed")
	}
	out := make([][]float32, len(texts))

	if cap(e.pool) <= 1 {
		ctx := <-e.pool
		defer func() { e.pool <- ctx }()
		for i, t := range texts {
			v, err := embedOne(ctx, e.model, e.dims, t)
			if err != nil {
				return nil, fmt.Errorf("embed (text %d): %w", i, err)
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
			ctx := <-e.pool
			defer func() { e.pool <- ctx }()
			v, err := embedOne(ctx, e.model, e.dims, t)
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
	ctx := <-e.pool
	defer func() { e.pool <- ctx }()
	return embedOne(ctx, e.model, e.dims, text)
}

func (e *localEmbedder) Dimensions() int       { return e.dims }
func (e *localEmbedder) ModelName() string     { return e.modelName }
func (e *localEmbedder) Family() PromptFamily  { return e.family }

func (e *localEmbedder) isClosed() bool {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.closed
}

func (e *localEmbedder) Close() error {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return nil
	}
	e.closed = true
	closeContexts(e.all)
	gollama.Model_free(e.model)
	return nil
}

func closeContexts(ctxs []gollama.LlamaContext) {
	for _, c := range ctxs {
		gollama.Free(c)
	}
}

// embedOne runs a full tokenize → batch → decode → get_embeddings
// cycle for a single text and returns the L2-normalised vector.
//
// dianlight's batch struct is a thin shell over the C llama_batch;
// fields like Token / Pos / SeqId / Logits are pointers we have to
// fill via unsafe.Pointer slice gymnastics. The pattern is taken
// from dianlight's own examples/embedding/main.go; if dianlight
// ever wraps it in a higher-level helper we should switch to that.
func embedOne(ctx gollama.LlamaContext, model gollama.LlamaModel, dims int, text string) ([]float32, error) {
	tokens, err := gollama.Tokenize(model, text, true, true)
	if err != nil {
		return nil, fmt.Errorf("tokenize: %w", err)
	}
	if len(tokens) == 0 {
		return nil, errors.New("tokenizer produced zero tokens")
	}
	if len(tokens) > math.MaxInt32 {
		return nil, fmt.Errorf("text too long: %d tokens", len(tokens))
	}

	batch := gollama.Batch_init(int32(len(tokens)), 0, 1)
	defer gollama.Batch_free(batch)

	// Fill the batch in place. Each token at position i, sequence 0,
	// with logits flag set so the encoder produces an output for it.
	addTokensToBatch(&batch, tokens, 0)

	if err := gollama.Decode(ctx, batch); err != nil {
		return nil, fmt.Errorf("decode: %w", err)
	}

	embPtr := gollama.Get_embeddings(ctx)
	if embPtr == nil {
		return nil, errors.New("get_embeddings returned nil")
	}
	src := unsafe.Slice(embPtr, dims)
	out := make([]float32, dims)
	copy(out, src)
	l2Normalize(out)
	return out, nil
}

// addTokensToBatch mirrors the helper from dianlight's embedding
// example. The C llama_batch struct exposes its arrays as raw
// pointers; we cast them to large arrays via unsafe.Pointer and
// write fields in place.
func addTokensToBatch(batch *gollama.LlamaBatch, tokens []gollama.LlamaToken, seqID gollama.LlamaSeqId) {
	tokensPtr := (*[1 << 20]gollama.LlamaToken)(unsafe.Pointer(batch.Token))
	posPtr := (*[1 << 20]gollama.LlamaPos)(unsafe.Pointer(batch.Pos))
	seqIDPtr := (*[1 << 20]*gollama.LlamaSeqId)(unsafe.Pointer(batch.SeqId))
	logitsPtr := (*[1 << 20]int8)(unsafe.Pointer(batch.Logits))

	seq := seqID
	for i, tok := range tokens {
		tokensPtr[i] = tok
		posPtr[i] = gollama.LlamaPos(i)
		seqIDPtr[i] = &seq
		logitsPtr[i] = 1
	}
	batch.NTokens = int32(len(tokens))
}

// l2Normalize scales v in place so its L2 norm is 1. nomic-embed-text
// (and most modern BERT-class embedders) return normalised vectors
// only when explicitly asked; we normalise here so cosine similarity
// at the vec0 layer matches what the model card describes.
func l2Normalize(v []float32) {
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	norm := math.Sqrt(sum)
	if norm == 0 {
		return
	}
	for i := range v {
		v[i] = float32(float64(v[i]) / norm)
	}
}

// deriveModelName turns "/x/y/nomic-embed-text-v1.5.Q8_0.gguf" into
// "nomic-embed-text-v1.5.Q8_0" — same as the godeps-era helper, kept
// here so call sites don't need to change.
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
