// Local GGUF generation via the same prebuilt llama-server
// subprocess the embedder uses, minus the --embedding flag. Each
// generator owns one server tied to its own Unix socket; Generate
// POSTs to /completion (llama.cpp's native endpoint — simpler and
// prompt-agnostic vs /v1/chat/completions which imposes the model's
// chat template on raw prompts).
//
// Like the embedder, this path swapped in v0.2.2 from in-process
// dianlight/gollama.cpp to subprocess-over-Unix-socket. See
// internal/embed/local.go for the full backstory; the motivations
// (no CGo, no struct ABI drift, one shared binary across embed +
// generate) apply equally here.

package llm

import (
	"context"
	"errors"
	"fmt"
	"os"
	"runtime"
	"sync"

	"github.com/ugurcan-aytar/recall/internal/llamacpp"
)

// LocalGeneratorOptions configures the llama-server subprocess.
type LocalGeneratorOptions struct {
	ModelPath string // absolute path to the .gguf file
	Threads   int    // 0 ⇒ llama.cpp default
	Context   int    // 0 ⇒ let llama.cpp infer from the model
}

// DefaultMaxTokens mirrors the pre-v0.2.2 behaviour — recall's
// generation use cases (query expansion, reranker yes/no, HyDE
// passage) all fit well below this.
const DefaultMaxTokens = 256

// generatorServer abstracts the llamacpp.Server surface the
// generator needs, so tests can stub the HTTP layer without
// spawning a real subprocess.
type generatorServer interface {
	PostJSON(ctx context.Context, path string, body, out any) error
	Close() error
}

type localGenerator struct {
	server    generatorServer
	modelName string

	// Generation is sequential per generator — llama-server handles
	// concurrency internally if we ever want to open more slots,
	// but recall's paths (expand = 1 call, rerank = N sequential
	// calls) see no benefit from parallelism on a single model
	// load. Serialise here with a mutex.
	mu     sync.Mutex
	closed bool
}

// chatMessage is one turn in an OpenAI-style chat exchange.
type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// chatRequest is the subset of /v1/chat/completions we send.
// Chat completions applies the model's own chat template around
// the prompt — critical for instruct-tuned models (qmd-query-
// expansion, Qwen2.5-Instruct), which produce the wrong format
// when their prompt is passed raw through /completion.
type chatRequest struct {
	Model       string        `json:"model"`
	Messages    []chatMessage `json:"messages"`
	MaxTokens   int           `json:"max_tokens"`
	Temperature float32       `json:"temperature"`
	TopP        float32       `json:"top_p"`
	TopK        int           `json:"top_k"`
	// Seed keeps greedy sampling deterministic across runs.
	Seed int `json:"seed"`
}

type chatResponseChoice struct {
	Message chatMessage `json:"message"`
}

type chatResponse struct {
	Choices []chatResponseChoice `json:"choices"`
}

// NewLocalGenerator boots a llama-server (no --embedding flag)
// bound to a Unix socket, ready to accept /completion requests.
// The first call blocks for model load (3-10 s depending on model
// size); subsequent Generate calls reuse the hot server.
func NewLocalGenerator(opts LocalGeneratorOptions) (Generator, error) {
	if opts.ModelPath == "" {
		return nil, errors.New("local generator: model path is required")
	}
	if _, err := os.Stat(opts.ModelPath); err != nil {
		return nil, fmt.Errorf("local generator: model file: %w", err)
	}

	threads := opts.Threads
	if threads <= 0 {
		threads = runtime.NumCPU()
	}
	ctxSize := opts.Context
	if ctxSize <= 0 {
		// Generation needs more headroom than embedding: prompt +
		// output + a safety margin. 4096 fits recall's longest
		// expand / rerank / hyde prompts with room to spare on
		// Qwen3 / Qwen2.5 models (their training context is much
		// larger, we're nowhere near pushing the model).
		ctxSize = 4096
	}

	srv, err := llamacpp.StartServer(context.Background(), llamacpp.ServerOptions{
		ModelPath:  opts.ModelPath,
		Embedding:  false, // generation mode — exposes /completion and /v1/chat/completions
		Context:    ctxSize,
		BatchSize:  ctxSize,
		UBatchSize: ctxSize,
		Parallel:   1, // one slot; recall's generator paths are sequential
		Threads:    threads,
		GPULayers:  -1,
	})
	if err != nil {
		return nil, fmt.Errorf("start llama-server: %w", err)
	}

	return &localGenerator{
		server:    srv,
		modelName: deriveModelName(opts.ModelPath),
	}, nil
}

// LocalGeneratorAvailable reports whether NewLocalGenerator can
// succeed in this build. v0.2.2 ships the subprocess path on
// every platform we build prebuilts for, so always true.
func LocalGeneratorAvailable() bool { return true }

// Generate runs the prompt through the loaded model with greedy
// sampling (temperature=0, top_k=1) so recall's parsing code can
// rely on deterministic output across runs.
func (g *localGenerator) Generate(prompt string, opts ...GenerateOption) (string, error) {
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.closed {
		return "", errors.New("local generator is closed")
	}

	cfg := GenerateOptions{MaxTokens: DefaultMaxTokens}
	for _, o := range opts {
		o(&cfg)
	}
	if cfg.MaxTokens <= 0 {
		cfg.MaxTokens = DefaultMaxTokens
	}

	req := chatRequest{
		Model:       g.modelName,
		Messages:    []chatMessage{{Role: "user", Content: prompt}},
		MaxTokens:   cfg.MaxTokens,
		Temperature: 0,
		TopP:        1,
		TopK:        1,
		Seed:        0,
	}
	var resp chatResponse
	if err := g.server.PostJSON(context.Background(), "/v1/chat/completions", req, &resp); err != nil {
		return "", fmt.Errorf("llama-server /v1/chat/completions: %w", err)
	}
	if len(resp.Choices) == 0 {
		return "", errors.New("llama-server /v1/chat/completions: empty choices")
	}
	return resp.Choices[0].Message.Content, nil
}

func (g *localGenerator) ModelName() string { return g.modelName }

func (g *localGenerator) Close() error {
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.closed {
		return nil
	}
	g.closed = true
	return g.server.Close()
}

// deriveModelName mirrors the embed package helper — keeps the
// rerank / expand UX consistent regardless of which backend is
// loaded. Kept duplicated rather than imported to avoid a cycle
// between internal/llm and internal/embed.
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
