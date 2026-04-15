// Package expand turns a single user query into a small set of
// alternative queries that downstream retrieval can fuse together.
// recall delegates the actual generation to a [llm.Generator] —
// typically the qmd-query-expansion-1.7B model recall ships under
// `recall models download --expansion`. Output is parsed into three
// buckets that map directly onto recall's retrieval pipeline:
//
//   lex   → BM25 alternative queries
//   vec   → vector-search alternative queries
//   hyde  → hypothetical answer passages (consumed by feature #6)
//
// Callers wanting only expansion (#3) ignore the Hyde slice; callers
// wanting only HyDE (#6) ignore Lex/Vec. One Generate call covers
// both, so callers running both features pay for one model
// invocation.
package expand

import (
	"fmt"
	"strings"

	"github.com/ugurcan-aytar/recall/internal/llm"
)

// PromptPrefix is the literal token the qmd-query-expansion model
// expects. The leading "/no_think" is a Qwen3 directive that skips
// internal reasoning so the model emits the structured output
// directly. Other generation models will see it as a no-op token
// and behave the same way they would on the bare prompt.
const PromptPrefix = "/no_think Expand this search query: "

// DefaultMaxTokens is the response cap recall asks for. Three
// structured lines × ~80 tokens each gives plenty of headroom.
const DefaultMaxTokens = 256

// Expanded is the parsed result of one Expand call. Original is the
// query the user typed; Lex / Vec / Hyde hold the model's variants.
// Empty slices are valid and mean "the model didn't produce that
// kind of variant" — callers should fall back to Original in that
// case.
type Expanded struct {
	Original string
	Lex      []string
	Vec      []string
	Hyde     []string
}

// Options tweaks one Expand call.
type Options struct {
	// Intent is an optional one-line description of what the user is
	// actually after. When set, recall sends a second prompt line
	// ("Query intent: <intent>") so the model can disambiguate. qmd
	// uses this for things like "intent: API rate limiting" alongside
	// a bare-noun query.
	Intent string

	// MaxTokens caps the generator response. 0 ⇒ DefaultMaxTokens.
	MaxTokens int

	// IncludeLex retains lex entries in the parsed output. Set to
	// false when the caller already has the user's literal query
	// covered (e.g. recall's hybrid path always BM25-searches the
	// original) and only wants vec/hyde variants.
	IncludeLex bool
}

// Expand drives gen with a query-expansion prompt and parses the
// structured response. Returns the original query in Expanded.Original
// regardless of what the model produced.
func Expand(gen llm.Generator, query string, opts Options) (*Expanded, error) {
	if gen == nil {
		return nil, fmt.Errorf("expand: generator is required")
	}
	if strings.TrimSpace(query) == "" {
		return nil, fmt.Errorf("expand: query is empty")
	}

	maxTokens := opts.MaxTokens
	if maxTokens <= 0 {
		maxTokens = DefaultMaxTokens
	}

	prompt := PromptPrefix + query
	if opts.Intent != "" {
		prompt += "\nQuery intent: " + opts.Intent
	}

	raw, err := gen.Generate(prompt, llm.WithMaxTokens(maxTokens))
	if err != nil {
		return nil, fmt.Errorf("expand: %w", err)
	}
	return Parse(raw, query, opts.IncludeLex), nil
}

// Parse walks the model's raw response line by line, extracting
// `lex: …`, `vec: …`, `hyde: …` entries. Lines that don't match the
// `<type>: <content>` shape are silently dropped — robust against
// the model emitting a header, a closing token, or accidental
// whitespace. Original is set on the returned struct so callers don't
// need to plumb the user's query through separately.
//
// includeLex=false drops the lex bucket entirely (the caller's
// pipeline already runs BM25 on the original query, so duplicate-ish
// lex variants are noise).
func Parse(raw, original string, includeLex bool) *Expanded {
	out := &Expanded{Original: original}
	originalLower := strings.ToLower(original)
	originalTokens := tokenSet(originalLower)

	for _, line := range strings.Split(raw, "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		idx := strings.Index(line, ":")
		if idx <= 0 || idx == len(line)-1 {
			continue
		}
		kind := strings.ToLower(strings.TrimSpace(line[:idx]))
		content := strings.TrimSpace(line[idx+1:])
		if content == "" {
			continue
		}

		switch kind {
		case "lex":
			if !includeLex {
				continue
			}
			// Lex variants that share zero tokens with the original
			// are usually hallucinated drift — qmd filters them out
			// the same way.
			if !shareToken(content, originalTokens) {
				continue
			}
			out.Lex = append(out.Lex, content)
		case "vec":
			out.Vec = append(out.Vec, content)
		case "hyde":
			out.Hyde = append(out.Hyde, content)
		default:
			// Unknown bucket. Drop silently.
		}
	}
	return out
}

// tokenSet lowercases s and returns its set of word tokens. Used for
// the "share-a-token-with-original" filter on lex variants.
func tokenSet(s string) map[string]struct{} {
	out := map[string]struct{}{}
	for _, f := range strings.Fields(stripPunct(s)) {
		if len(f) > 1 {
			out[f] = struct{}{}
		}
	}
	return out
}

// shareToken reports whether content has at least one token in common
// with the original query's token set.
func shareToken(content string, originalTokens map[string]struct{}) bool {
	for _, f := range strings.Fields(stripPunct(strings.ToLower(content))) {
		if _, ok := originalTokens[f]; ok {
			return true
		}
	}
	return false
}

// stripPunct replaces every non-letter / non-digit / non-whitespace
// rune with a space so Fields can split cleanly. We don't unicode-
// folding here; a future improvement could share the unicode-aware
// tokeniser used by the BM25 sanitiser.
func stripPunct(s string) string {
	var b strings.Builder
	b.Grow(len(s))
	for _, r := range s {
		switch {
		case r >= 'a' && r <= 'z',
			r >= '0' && r <= '9',
			r == ' ', r == '\t':
			b.WriteRune(r)
		default:
			b.WriteByte(' ')
		}
	}
	return b.String()
}
