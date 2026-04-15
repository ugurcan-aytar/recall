// Local cross-encoder reranker via llama.cpp's `--reranking` flag
// and the /v1/rerank HTTP endpoint. Shipping since v0.2.4; replaces
// the Qwen2.5-1.5B-Instruct binary-yes/no fallback we were forced to
// use while the in-process binding didn't expose llama.cpp's rank-
// pooling surface.
//
// Contract:
//
//	POST /v1/rerank
//	  {"query": "...", "top_n": N, "documents": ["...", ...]}
//	→ {"results": [{"index": i, "relevance_score": <float logit>}, ...]}
//
// The endpoint returns a score per input document, sorted by score
// desc. Scores are raw cross-encoder logits (roughly [-12, +8] in
// practice for bge-reranker-v2-m3) — real gradient, not binary.
// Callers that need a probability-like [0,1] score should sigmoid
// or min-max normalise within the candidate set; recall's rerank
// package does the latter before feeding the position-aware blender.

package llm

import (
	"context"
	"errors"
	"fmt"
	"os"
	"runtime"

	"github.com/ugurcan-aytar/recall/internal/llamacpp"
)

// Reranker is the contract cross-encoder reranker backends satisfy.
//
// Implementations MUST be safe for concurrent Rerank calls.
type Reranker interface {
	// Rerank scores each document against the query. Returns one
	// float64 per input document, in input order. Higher score ⇒
	// more relevant. The scale is implementation-defined (raw
	// logits, sigmoid, 0-1, …); callers that need a fixed range
	// should normalise.
	Rerank(ctx context.Context, query string, documents []string) ([]float64, error)

	// ModelName is a stable identifier surfaced in recall doctor
	// and in error messages.
	ModelName() string

	// Close releases the loaded model and the subprocess. After
	// Close, further Rerank calls return an error.
	Close() error
}

// ErrLocalRerankerNotAvailable signals that the local reranker
// can't come up (model file missing, binary download failed,
// subprocess boot failed). Callers that want graceful fallback
// should check errors.Is against this sentinel.
var ErrLocalRerankerNotAvailable = errors.New(
	"local reranker unavailable — run `recall models download --reranker`",
)

// LocalRerankerOptions configures NewLocalReranker.
type LocalRerankerOptions struct {
	ModelPath string // absolute path to the .gguf reranker (e.g. bge-reranker-v2-m3)
	Threads   int    // 0 ⇒ runtime.NumCPU()
	Context   int    // 0 ⇒ 4096 (plenty for query + single passage)
}

// rerankerServer mirrors the abstraction used by localEmbedder /
// localGenerator — lets tests inject a fake HTTP layer.
type rerankerServer interface {
	PostJSON(ctx context.Context, path string, body, out any) error
	Close() error
}

type localReranker struct {
	server    rerankerServer
	modelName string
}

// rerankRequest mirrors llama-server's /v1/rerank body.
type rerankRequest struct {
	Query     string   `json:"query"`
	Documents []string `json:"documents"`
	TopN      int      `json:"top_n"`
}

type rerankResult struct {
	Index          int     `json:"index"`
	RelevanceScore float64 `json:"relevance_score"`
}

type rerankResponse struct {
	Results []rerankResult `json:"results"`
}

// NewLocalReranker boots a llama-server subprocess in `--reranking`
// mode. The server exposes /v1/rerank which takes a query + N
// documents and returns N cross-encoder logits. Use it for any
// reranker model that llama.cpp supports (bge-reranker-v2-m3,
// Qwen3-Reranker-0.6B, gte-multilingual-reranker-base, …).
func NewLocalReranker(opts LocalRerankerOptions) (Reranker, error) {
	if opts.ModelPath == "" {
		return nil, errors.New("local reranker: model path is required")
	}
	if _, err := os.Stat(opts.ModelPath); err != nil {
		return nil, fmt.Errorf("%w: model file %s: %v", ErrLocalRerankerNotAvailable, opts.ModelPath, err)
	}

	threads := opts.Threads
	if threads <= 0 {
		threads = runtime.NumCPU()
	}
	ctxSize := opts.Context
	if ctxSize <= 0 {
		// Cross-encoder reranking sends (query + single document)
		// through the encoder; bge-reranker-v2-m3's max_position is
		// 8192 but the realistic (query,passage) token budget is
		// much lower. 4096 is a conservative cap that keeps KV
		// cache small while never cutting off real recall passages
		// (the embedder's maxInputChars=4000 upper-bounds passage
		// tokens to ~2048 WordPieces).
		ctxSize = 4096
	}

	srv, err := llamacpp.StartServer(context.Background(), llamacpp.ServerOptions{
		ModelPath:  opts.ModelPath,
		Reranking:  true,
		Context:    ctxSize,
		BatchSize:  ctxSize,
		UBatchSize: ctxSize,
		Parallel:   1, // rerank calls are batched server-side; one slot is fine
		Threads:    threads,
		GPULayers:  -1,
	})
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrLocalRerankerNotAvailable, err)
	}
	return &localReranker{
		server:    srv,
		modelName: deriveModelName(opts.ModelPath),
	}, nil
}

// LocalRerankerAvailable reports whether NewLocalReranker can
// succeed in this build. Always true on v0.2.4+ (subprocess
// pattern), kept for API symmetry with LocalEmbedderAvailable.
func LocalRerankerAvailable() bool { return true }

func (r *localReranker) Rerank(ctx context.Context, query string, documents []string) ([]float64, error) {
	if len(documents) == 0 {
		return nil, nil
	}
	req := rerankRequest{
		Query:     query,
		Documents: documents,
		TopN:      len(documents),
	}
	var resp rerankResponse
	if err := r.server.PostJSON(ctx, "/v1/rerank", req, &resp); err != nil {
		return nil, fmt.Errorf("llama-server /v1/rerank: %w", err)
	}
	if len(resp.Results) != len(documents) {
		return nil, fmt.Errorf("reranker returned %d scores for %d documents", len(resp.Results), len(documents))
	}
	// The endpoint returns results sorted by score desc with an
	// `index` field pointing into the input. Reassemble in input
	// order so callers can zip straight against their candidate
	// slice.
	out := make([]float64, len(documents))
	filled := make([]bool, len(documents))
	for _, r := range resp.Results {
		if r.Index < 0 || r.Index >= len(out) {
			return nil, fmt.Errorf("reranker returned out-of-range index %d", r.Index)
		}
		out[r.Index] = r.RelevanceScore
		filled[r.Index] = true
	}
	for i, ok := range filled {
		if !ok {
			return nil, fmt.Errorf("reranker returned no score for document %d", i)
		}
	}
	return out, nil
}

func (r *localReranker) ModelName() string { return r.modelName }

func (r *localReranker) Close() error {
	if r.server == nil {
		return nil
	}
	return r.server.Close()
}
