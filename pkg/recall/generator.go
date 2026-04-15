package recall

// Public re-export of the LLM generation surface. Library consumers
// (brain in particular) drive query expansion / HyDE / reranking
// through these aliases without reaching into internal/llm or
// internal/expand. Behaviour lives in the internal packages — this
// file is purely the exported facade.

import (
	"context"

	"github.com/ugurcan-aytar/recall/internal/embed"
	"github.com/ugurcan-aytar/recall/internal/expand"
	"github.com/ugurcan-aytar/recall/internal/llm"
	"github.com/ugurcan-aytar/recall/internal/rerank"
)

// Generator is recall's text-generation contract. See [llm.Generator]
// for the full doc.
type Generator = llm.Generator

// GenerateOption mutates a single Generate call.
type GenerateOption = llm.GenerateOption

// WithMaxTokens caps the response at n tokens.
func WithMaxTokens(n int) GenerateOption { return llm.WithMaxTokens(n) }

// MockGenerator is the test-only canned-response generator.
type MockGenerator = llm.MockGenerator

// NewMockGenerator returns a MockGenerator wired with the given canned
// responses (or no responses, if nil — Default is then returned for
// every prompt).
func NewMockGenerator(responses map[string]string) *MockGenerator {
	return llm.NewMockGenerator(responses)
}

// LocalGeneratorOptions configures [NewLocalGenerator].
type LocalGeneratorOptions = llm.LocalGeneratorOptions

// NewLocalGenerator constructs a GGUF-backed generator. Returns
// [ErrLocalGeneratorNotCompiled] on default builds (no embed_llama
// tag).
func NewLocalGenerator(opts LocalGeneratorOptions) (Generator, error) {
	return llm.NewLocalGenerator(opts)
}

// LocalGeneratorAvailable reports whether this binary was built with
// the `embed_llama` tag and can construct a local LLM.
func LocalGeneratorAvailable() bool {
	return llm.LocalGeneratorAvailable()
}

// ErrLocalGeneratorNotCompiled is returned by [NewLocalGenerator] in
// builds without the `embed_llama` tag.
var ErrLocalGeneratorNotCompiled = llm.ErrLocalGeneratorNotCompiled

// Expansion model defaults — point at qmd-query-expansion-1.7B by
// default, override with $RECALL_EXPAND_MODEL.
const (
	DefaultExpansionModelName = embed.DefaultExpansionModelName
	DefaultExpansionModelURL  = embed.DefaultExpansionModelURL
)

// ResolveActiveExpansionModelPath returns the GGUF file recall should
// load for query expansion / HyDE, honouring $RECALL_EXPAND_MODEL.
// Bare filename joins with the models directory; absolute path passes
// through.
func ResolveActiveExpansionModelPath() (string, error) {
	return embed.ResolveActiveExpansionModelPath()
}

// Reranker model defaults — point at Qwen2.5-1.5B-Instruct by
// default, override with $RECALL_RERANK_MODEL.
const (
	DefaultRerankerModelName = embed.DefaultRerankerModelName
	DefaultRerankerModelURL  = embed.DefaultRerankerModelURL
)

// ResolveActiveRerankerModelPath returns the GGUF file recall should
// load for the --rerank reranker, honouring $RECALL_RERANK_MODEL.
func ResolveActiveRerankerModelPath() (string, error) {
	return embed.ResolveActiveRerankerModelPath()
}

// BlendBands lets external consumers retune the position-aware
// reranker blend (--rerank). DefaultRerankBlendBands matches qmd:
// 75/25 for ranks 0-2, 60/40 for 3-9, 40/60 for 10+.
type BlendBands = rerank.BlendBands

// BlendBand is one rank-range entry in BlendBands.
type BlendBand = rerank.BlendBand

// DefaultRerankBlendBands is the qmd-mirrored default. Document
// any change as a semver-minor event.
var DefaultRerankBlendBands = rerank.DefaultBlendBands

// Expanded is the parsed result of [Expand]. Lex / Vec / Hyde slices
// hold the model's variants for each retrieval surface; Original is
// the user's literal query.
type Expanded = expand.Expanded

// ExpandOptions tweaks one [Expand] call.
type ExpandOptions = expand.Options

// Expand drives gen with a query-expansion prompt and parses the
// structured response. See [expand.Expand] for the full doc.
func Expand(gen Generator, query string, opts ExpandOptions) (*Expanded, error) {
	return expand.Expand(gen, query, opts)
}

// Reranker is the cross-encoder reranking contract — see
// [llm.Reranker]. brain and other library consumers construct one
// via [NewLocalReranker] and hand it to [Rerank].
type Reranker = llm.Reranker

// LocalRerankerOptions configures [NewLocalReranker].
type LocalRerankerOptions = llm.LocalRerankerOptions

// NewLocalReranker boots a llama-server subprocess in `--reranking`
// mode against the GGUF at opts.ModelPath. Returns a Reranker that
// talks to `/v1/rerank` under the hood.
func NewLocalReranker(opts LocalRerankerOptions) (Reranker, error) {
	return llm.NewLocalReranker(opts)
}

// LocalRerankerAvailable reports whether this binary can spawn a
// local reranker. Always true on v0.2.4+ (subprocess pattern).
func LocalRerankerAvailable() bool { return llm.LocalRerankerAvailable() }

// ErrLocalRerankerNotAvailable signals that the reranker can't come
// up (model file missing, llama.cpp binary fetch failed, server
// boot failed). Callers that want graceful fallback should check
// errors.Is against this sentinel.
var ErrLocalRerankerNotAvailable = llm.ErrLocalRerankerNotAvailable

// Scored pairs a retrieved candidate with its reranker relevance
// score. See [rerank.Scored] for field semantics.
type Scored = rerank.Scored

// RerankOptions tweaks one [Rerank] call.
type RerankOptions = rerank.Options

// Rerank scores each candidate against the query with a cross-
// encoder reranker and returns the slice sorted by normalised
// score desc. See [rerank.Rerank] for the full doc.
func Rerank(ctx context.Context, rr Reranker, query string, candidates []SearchResult, opts RerankOptions) ([]Scored, error) {
	return rerank.Rerank(ctx, rr, query, candidates, opts)
}

// PositionAwareBlend fuses RRF rank with reranker score using
// per-rank-band weights. See [rerank.PositionAwareBlend].
func PositionAwareBlend(scored []Scored, bands BlendBands) []Scored {
	return rerank.PositionAwareBlend(scored, bands)
}
