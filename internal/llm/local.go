// Local GGUF generation backend on top of dianlight/gollama.cpp.
// See internal/embed/local.go for the backend overview — same
// purego runtime, same auto-download lifecycle, same single-Model
// + multiple-Context pattern. The generator path differs from the
// embedder in two ways:
//
//   1. Embeddings flag is OFF; Logits flag is ON so we can sample
//      from the model.
//   2. Generation is a tokenize → decode-prompt → loop(sample +
//      decode) cycle, not a one-shot encoder pass.
//
// Backend lifecycle is shared with the embedder (one process-wide
// gollama.Backend_init) so the two packages cooperate cleanly when
// `recall query --expand` opens both an embedder and a generator.

package llm

import (
	"errors"
	"fmt"
	"math"
	"runtime"
	"strings"
	"sync"

	gollama "github.com/dianlight/gollama.cpp"

	"github.com/ugurcan-aytar/recall/internal/llamacpp"
)

// LocalGeneratorOptions configures the GGUF generation backend.
type LocalGeneratorOptions struct {
	ModelPath string // absolute path to the .gguf file
	Threads   int    // 0 ⇒ runtime.NumCPU()
	Context   int    // 0 ⇒ 4096 (generation needs more headroom than embedding)
}

type localGenerator struct {
	model     gollama.LlamaModel
	ctx       gollama.LlamaContext
	modelName string
	ctxSize   int

	mu     sync.Mutex
	closed bool
}

// NewLocalGenerator loads opts.ModelPath and prepares one inference
// context. recall's generation paths (--expand / --hyde / --rerank)
// are sequential per query, so we don't pool contexts here — a
// single mutex-guarded context keeps the surface simple.
func NewLocalGenerator(opts LocalGeneratorOptions) (Generator, error) {
	if opts.ModelPath == "" {
		return nil, errors.New("local generator: model path is required")
	}
	if err := llamacpp.EnsureBackend(); err != nil {
		return nil, err
	}

	threads := opts.Threads
	if threads <= 0 {
		threads = runtime.NumCPU()
	}
	ctxSize := opts.Context
	if ctxSize <= 0 {
		ctxSize = 4096
	}

	modelParams := gollama.Model_default_params()
	model, err := gollama.Model_load_from_file(opts.ModelPath, modelParams)
	if err != nil {
		return nil, fmt.Errorf("load gguf model %s: %w", opts.ModelPath, err)
	}

	ctxParams := gollama.Context_default_params()
	ctxParams.NCtx = uint32(ctxSize)
	ctxParams.NBatch = uint32(ctxSize)
	ctxParams.NUbatch = uint32(ctxSize)
	ctxParams.NSeqMax = 1
	ctxParams.NThreads = int32(threads)
	ctxParams.NThreadsBatch = int32(threads)
	ctxParams.Logits = 1 // generation needs logits per token

	ctx, err := gollama.Init_from_model(model, ctxParams)
	if err != nil {
		gollama.Model_free(model)
		return nil, fmt.Errorf("new context: %w", err)
	}

	return &localGenerator{
		model:     model,
		ctx:       ctx,
		modelName: deriveModelName(opts.ModelPath),
		ctxSize:   ctxSize,
	}, nil
}

// LocalGeneratorAvailable always returns true now that the backend
// downloads itself. Kept for API compatibility with the v0.1
// build-tag stub.
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

	// Tokenize the prompt; addSpecial=true so chat / instruct models
	// that expect BOS get it. parseSpecial=false so user text isn't
	// re-interpreted as control tokens.
	tokens, err := gollama.Tokenize(g.model, prompt, true, false)
	if err != nil {
		return "", fmt.Errorf("tokenize: %w", err)
	}
	if len(tokens) == 0 {
		return "", errors.New("tokenizer produced zero tokens")
	}
	if len(tokens) > math.MaxInt32 {
		return "", fmt.Errorf("prompt too long: %d tokens", len(tokens))
	}

	// Feed the prompt into the model's KV cache in one batch.
	promptBatch := gollama.Batch_get_one(tokens)
	defer gollama.Batch_free(promptBatch)
	if err := gollama.Decode(g.ctx, promptBatch); err != nil {
		return "", fmt.Errorf("decode prompt: %w", err)
	}

	// Greedy sampler — recall's generation use cases (query
	// expansion, HyDE, reranker yes/no) want deterministic
	// outputs. A future caller-controlled temperature can swap
	// in a different sampler.
	sampler := gollama.Sampler_init_greedy()
	defer gollama.Sampler_free(sampler)

	var b strings.Builder
	b.Grow(cfg.MaxTokens * 4) // rough rune estimate

	// dianlight v0.2.x doesn't expose Vocab_is_eog / Token_is_eog,
	// so we rely on (a) MaxTokens as the hard cap and (b) the
	// pre-computed ctxSize as the safety overflow check. The model
	// will keep emitting tokens past its natural EOS — Token_to_piece
	// returns an empty string for control tokens, so the leak is
	// invisible bytes rather than garbled text. recall's
	// generation use cases (yes/no rerank, ≤256-token expansion,
	// HyDE passage) all fit well below MaxTokens.
	nCur := len(tokens)
	for produced := 0; produced < cfg.MaxTokens; produced++ {
		if nCur >= g.ctxSize {
			break
		}
		newToken := gollama.Sampler_sample(sampler, g.ctx, -1)
		piece := gollama.Token_to_piece(g.model, newToken, false)
		b.WriteString(piece)

		// Feed the sampled token back so the next iteration
		// extends from the right KV state.
		batch := gollama.Batch_get_one([]gollama.LlamaToken{newToken})
		if err := gollama.Decode(g.ctx, batch); err != nil {
			gollama.Batch_free(batch)
			return "", fmt.Errorf("decode token at position %d: %w", nCur, err)
		}
		gollama.Batch_free(batch)
		nCur++
	}
	return b.String(), nil
}

func (g *localGenerator) ModelName() string { return g.modelName }

func (g *localGenerator) Close() error {
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.closed {
		return nil
	}
	g.closed = true
	gollama.Free(g.ctx)
	gollama.Model_free(g.model)
	return nil
}

// deriveModelName turns "/x/y/qmd-query-expansion-1.7B-q4_k_m.gguf"
// into "qmd-query-expansion-1.7B-q4_k_m". Same logic as the embed
// package; duplicated here to avoid a sub-package import cycle.
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
