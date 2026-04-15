// Local GGUF embedding via llama.cpp's official prebuilt llama-server
// running as a subprocess and talking over a Unix socket.
//
// Pre-v0.2.2 recall linked llama.cpp in-process via dianlight/gollama.cpp.
// That worked on paper but the dianlight Go struct drifted from the
// bundled C library (a leading `Seed` field was kept after llama.cpp
// dropped it from llama_context_params), so libffi shifted every
// argument one slot to the left. The visible failure mode: every
// embedding came back as a 768-dim zero vector, sqlite-vec scored
// every query at cosine 1.0, retrieval was useless.
//
// The fix swaps to the simplest possible IPC: spawn the upstream
// prebuilt llama-server, send /v1/embeddings (OpenAI-compatible
// batch input), get vectors back. Trade-offs:
//
//   * Pro: no CGo on the inference path, no struct ABI surface area,
//     no fork to maintain. Model loads ONCE per Embedder lifetime.
//   * Con: ~2s server boot before the first vector. Subsequent
//     batches are pure HTTP round-trips on a Unix socket.
//
// Workers maps to llama-server's --parallel flag (server slots).
// We then fire that many concurrent /v1/embeddings requests when
// the caller hands us > Workers texts.
package embed

import (
	"context"
	"errors"
	"fmt"
	"os"
	"runtime"
	"strings"
	"sync"

	"github.com/ugurcan-aytar/recall/internal/llamacpp"
)

// LocalEmbedderOptions configures the llama-server subprocess.
type LocalEmbedderOptions struct {
	ModelPath string // absolute path to the .gguf file
	Threads   int    // 0 ⇒ llama.cpp default
	Context   int    // 0 ⇒ let llama.cpp infer from the model
	Workers   int    // server slots and concurrent request count; 0 ⇒ 1
}

// MaxLocalWorkers caps concurrent server slots. Each slot owns a
// KV cache; total RAM scales linearly. Above 8 the marginal speed
// gain falls off sharply for embedding-only workloads on Apple
// Silicon (the GPU bottlenecks before the slots saturate).
const MaxLocalWorkers = 8

// embedServer abstracts the llamacpp.Server surface localEmbedder
// actually uses, so tests can swap in an httptest-backed fake
// without spinning up a real subprocess.
type embedServer interface {
	PostJSON(ctx context.Context, path string, body, out any) error
	Close() error
}

type localEmbedder struct {
	server    embedServer
	modelName string
	family    PromptFamily
	dims      int
	workers   int
	sem       chan struct{}

	mu     sync.Mutex
	closed bool
}

// embeddingsRequest mirrors the OpenAI /v1/embeddings request body
// schema that llama-server accepts. Input is a string slice; the
// server returns one vector per input in the same order.
type embeddingsRequest struct {
	Input []string `json:"input"`
	Model string   `json:"model"`
}

type embeddingsResponse struct {
	Data []struct {
		Embedding []float32 `json:"embedding"`
		Index     int       `json:"index"`
	} `json:"data"`
}

// NewLocalEmbedder boots a llama-server with --embedding bound to a
// Unix socket and probes /v1/embeddings to learn the model's vector
// dimensions. The first call may take several seconds (binary
// download on first use, then ~2s model load); subsequent calls
// reuse the already-running server.
func NewLocalEmbedder(opts LocalEmbedderOptions) (Embedder, error) {
	if opts.ModelPath == "" {
		return nil, errors.New("local embedder: model path is required")
	}
	if _, err := os.Stat(opts.ModelPath); err != nil {
		return nil, fmt.Errorf("local embedder: model file: %w", err)
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
		// Nomic-embed-text-v1.5 trained with n_ctx=2048, so
		// embeddings of inputs longer than that drift quality-
		// wise. But our chunker's token estimator (1.3×words) can
		// undercount real BERT WordPieces by 3-5× on dense source
		// code, producing the occasional 2050-2100 token outlier.
		// Default ctxSize=4096 (1× headroom) keeps those running
		// through the same llama-server invocation — degraded
		// quality on a tiny long tail beats hard-failing the whole
		// embed run.
		ctxSize = 4096
	}

	// llama-server divides --ctx-size across --parallel slots
	// (n_ctx_slot = n_ctx / n_parallel). To give every slot a full
	// ctxSize for long chunks we multiply on the way in.
	totalCtx := ctxSize * workers

	srv, err := llamacpp.StartServer(context.Background(), llamacpp.ServerOptions{
		ModelPath:  opts.ModelPath,
		Embedding:  true,
		Pooling:    "mean", // nomic + most BERT-class embedders want mean pooling
		Context:    totalCtx,
		BatchSize:  ctxSize, // -b ≥ a single chunk so one decode fits
		UBatchSize: ctxSize, // -ub ≥ n_tokens for encoder models like nomic
		Parallel:   workers,
		Threads:    threads,
		GPULayers:  -1, // let llama.cpp pick (Metal-on by default for Apple Silicon)
	})
	if err != nil {
		return nil, fmt.Errorf("start llama-server: %w", err)
	}

	name := deriveModelName(opts.ModelPath)
	dims, err := probeDims(srv, name)
	if err != nil {
		_ = srv.Close()
		return nil, fmt.Errorf("probe embedding dimensions: %w", err)
	}

	e := &localEmbedder{
		server:    srv,
		modelName: name,
		family:    ResolveFamily(name),
		dims:      dims,
		workers:   workers,
		sem:       make(chan struct{}, workers),
	}
	return e, nil
}

// LocalEmbedderAvailable returns true when this binary was built
// with local-embedding support. With the subprocess pattern the
// answer is always yes (the actual llama.cpp binary is fetched at
// runtime), so the function exists only for API compatibility with
// callers from the v0.1 build-tag era.
func LocalEmbedderAvailable() bool { return true }

// probeDims sends a one-token embed request so we learn the vector
// length the server will return for this model. Caching is the
// caller's job (we stash the result on the embedder struct).
func probeDims(srv embedServer, model string) (int, error) {
	var resp embeddingsResponse
	err := srv.PostJSON(context.Background(), "/v1/embeddings", embeddingsRequest{
		Input: []string{"recall dim probe"},
		Model: model,
	}, &resp)
	if err != nil {
		return 0, err
	}
	if len(resp.Data) == 0 || len(resp.Data[0].Embedding) == 0 {
		return 0, errors.New("server returned no embedding for the dim probe")
	}
	return len(resp.Data[0].Embedding), nil
}

func (e *localEmbedder) Embed(texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	if e.isClosed() {
		return nil, errors.New("local embedder is closed")
	}

	out := make([][]float32, len(texts))

	// Sequential path: keep the implementation simple when there's
	// only one slot or one text.
	if e.workers <= 1 || len(texts) == 1 {
		for i, t := range texts {
			v, err := e.embedBatch([]string{t})
			if err != nil {
				return nil, fmt.Errorf("embed (text %d): %w", i, err)
			}
			out[i] = v[0]
		}
		return out, nil
	}

	// Parallel path: split into per-slot batches and fan out. The
	// server's --parallel slots already pipeline within one batch,
	// but firing N concurrent batches keeps every slot busy when
	// the caller hands us many short texts.
	errs := make(chan error, len(texts))
	var wg sync.WaitGroup
	wg.Add(len(texts))
	for i, t := range texts {
		i, t := i, t
		e.sem <- struct{}{}
		go func() {
			defer wg.Done()
			defer func() { <-e.sem }()
			v, err := e.embedBatch([]string{t})
			if err != nil {
				errs <- fmt.Errorf("embed (text %d): %w", i, err)
				return
			}
			out[i] = v[0]
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
	v, err := e.embedBatch([]string{text})
	if err != nil {
		return nil, err
	}
	return v[0], nil
}

// maxInputChars caps how many bytes of source text we hand to
// llama-server per input. nomic-embed-text-v1.5 has n_ctx_train=2048,
// and llama-server's /v1/embeddings rejects inputs over the model's
// training context with HTTP 400 ("exceed_context_size_error") even
// when n_ctx_seq is larger. BERT WordPieces average ~2.2 bytes/token
// on dense code (worst case observed), so 4000 bytes is a hard upper
// bound that keeps even pathological chunks under the 2048 token
// model limit with margin. Normal markdown chunks (<2 chars/token
// average is rare for English prose) are nowhere near this cap.
const maxInputChars = 4000

// truncatedRetryFactor halves the per-input length on a single
// retry when llama-server still reports exceed_context_size_error
// after the initial cap. Belt-and-braces — covers genuinely weird
// inputs (binary blobs, base64, etc.) where bytes-per-token can
// drop below 1.
const truncatedRetryFactor = 2

func (e *localEmbedder) embedBatch(texts []string) ([][]float32, error) {
	payload := capInputLengths(texts, maxInputChars)
	var resp embeddingsResponse
	err := e.server.PostJSON(context.Background(), "/v1/embeddings", embeddingsRequest{
		Input: payload,
		Model: e.modelName,
	}, &resp)
	if err != nil && isExceedContextErr(err) {
		// Retry once with a much tighter cap. A second 400 means
		// the input has near-1-byte-per-token density (binary,
		// base64) and we should skip rather than spin.
		shorter := capInputLengths(payload, maxInputChars/truncatedRetryFactor)
		err = e.server.PostJSON(context.Background(), "/v1/embeddings", embeddingsRequest{
			Input: shorter,
			Model: e.modelName,
		}, &resp)
	}
	if err != nil {
		return nil, err
	}
	if len(resp.Data) != len(texts) {
		return nil, fmt.Errorf("server returned %d embeddings for %d inputs", len(resp.Data), len(texts))
	}
	out := make([][]float32, len(texts))
	// llama-server returns the vectors with a per-item Index field; trust
	// the index when it's set, otherwise rely on positional order (the
	// server preserves input order in practice).
	for _, item := range resp.Data {
		idx := item.Index
		if idx < 0 || idx >= len(out) {
			return nil, fmt.Errorf("server returned out-of-range index %d", idx)
		}
		if len(item.Embedding) != e.dims {
			return nil, fmt.Errorf("server returned %d-dim vector at index %d, expected %d", len(item.Embedding), idx, e.dims)
		}
		out[idx] = item.Embedding
	}
	for i, v := range out {
		if v == nil {
			return nil, fmt.Errorf("server returned no embedding for input %d", i)
		}
	}
	return out, nil
}

func (e *localEmbedder) Dimensions() int      { return e.dims }
func (e *localEmbedder) ModelName() string    { return e.modelName }
func (e *localEmbedder) Family() PromptFamily { return e.family }

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
	return e.server.Close()
}

// capInputLengths returns texts with each entry truncated to at
// most max bytes. Allocates a new slice only when at least one
// entry needs trimming so the common case stays zero-copy.
func capInputLengths(texts []string, max int) []string {
	var capped []string
	for i, t := range texts {
		if len(t) <= max {
			continue
		}
		if capped == nil {
			capped = make([]string, len(texts))
			copy(capped, texts)
		}
		capped[i] = t[:max]
	}
	if capped == nil {
		return texts
	}
	return capped
}

// isExceedContextErr matches the specific llama-server error
// returned when an input tokenizes to more than n_ctx_train. The
// substring is stable across the b6xxx / b7xxx / b8xxx release
// lines we've validated against.
func isExceedContextErr(err error) bool {
	if err == nil {
		return false
	}
	msg := err.Error()
	return strings.Contains(msg, "exceed_context_size_error") ||
		strings.Contains(msg, "is larger than the max context size") ||
		strings.Contains(msg, "is too large to process")
}

// deriveModelName turns "/x/y/nomic-embed-text-v1.5.Q8_0.gguf" into
// "nomic-embed-text-v1.5.Q8_0". Used as the metadata table's model
// label so reconcileModelName can detect when the user swapped models.
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
