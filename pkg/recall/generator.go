package recall

// Public re-export of the LLM generation surface. Library consumers
// (brain in particular) drive query expansion / HyDE / reranking
// through these aliases without reaching into internal/llm or
// internal/expand. Behaviour lives in the internal packages — this
// file is purely the exported facade.

import (
	"github.com/ugurcan-aytar/recall/internal/embed"
	"github.com/ugurcan-aytar/recall/internal/expand"
	"github.com/ugurcan-aytar/recall/internal/llm"
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
