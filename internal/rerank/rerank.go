// Package rerank scores (query, passage) pairs through a small
// generation LLM and returns the candidate list sorted by relevance.
//
// recall ideally would use a true cross-encoder reranker (Qwen3-
// Reranker via llama.cpp's `--pooling rank`), but gollama doesn't
// expose that surface yet. Until it does, we fall back to a binary
// yes/no prompt against an instruction-tuned generation model and
// derive a 0.0 / 1.0 score per candidate. Empirical POC against
// Qwen2.5-1.5B-Instruct showed 5/5 correct discrimination on a
// realistic 5-doc query, with deterministic single-token output and
// ~70 ms latency per call — see CLAUDE.md "Reranker fallback" for
// the test corpus.
//
// The binary score is then fed into the position-aware blender
// (feature #5) which fuses it with the RRF rank to produce a
// gradient signal even though the underlying classifier is binary.
package rerank

import (
	"context"
	"fmt"
	"sort"
	"strings"
	"sync"

	"github.com/ugurcan-aytar/recall/internal/llm"
	"github.com/ugurcan-aytar/recall/internal/store"
)

// Default knobs.
const (
	// DefaultTopN is how many candidates the reranker scores by
	// default. The first N RRF hits get the LLM treatment; anything
	// past N keeps its RRF rank unchanged. 30 mirrors qmd's window —
	// large enough to catch promotions from rank 25, small enough to
	// stay under a few seconds of LLM cost.
	DefaultTopN = 30

	// DefaultMaxTokens caps the model's response. "yes\n" or "no\n"
	// fits easily; we leave a little headroom for chatty models that
	// echo the question first.
	DefaultMaxTokens = 8

	// DefaultPassageBudget is the chunk-content cap (runes) sent to
	// the model. Long chunks balloon prompt tokens and slow inference
	// without changing the relevance verdict materially. 800 runes ≈
	// 200 tokens — comfortably inside any 2k-context model.
	DefaultPassageBudget = 800
)

// PromptTemplate is the binary-yes/no prompt recall sends per
// candidate. The single-line "Reply with exactly one word" framing
// was chosen empirically: a graded 0-10 scale at this model size
// produced too-generous scores (irrelevant docs landing 5-7), while
// the binary form gave 5/5 correct verdicts on the POC corpus.
const PromptTemplate = `Does the passage answer the query? Reply with exactly one word: yes or no.

Query: %s
Passage: %s`

// Scored pairs an input candidate with its 0.0 / 1.0 rerank verdict
// (Score) and the position-aware fused result (BlendedScore).
// RRFRank is the original 0-based rank in the input slice — used by
// [PositionAwareBlend] to pick the rerank-vs-rank weight band.
//
// BlendedScore is zero until PositionAwareBlend has been called;
// callers that don't run the blender can sort by Score directly.
type Scored struct {
	Result       store.SearchResult
	Score        float64
	BlendedScore float64
	RRFRank      int
}

// Options tweaks one Rerank call.
type Options struct {
	// TopN limits how many candidates we send to the LLM. 0 ⇒
	// DefaultTopN. The remaining tail is appended to the output
	// with a 0.5 "unscored" verdict so callers can still display it.
	TopN int

	// PassageBudget caps each passage at this many runes before
	// embedding it in the prompt. 0 ⇒ DefaultPassageBudget.
	PassageBudget int

	// MaxTokens caps each generation. 0 ⇒ DefaultMaxTokens.
	MaxTokens int

	// Workers controls how many concurrent Generate calls fire.
	// 0/1 ⇒ sequential. The wrapper does no internal pool by
	// itself; callers wanting parallelism should ensure the
	// Generator is safe for concurrent use (gollama's localGenerator
	// already serialises via mutex, so workers > 1 against a single
	// model gives no speedup — load multiple Generators if you need
	// real parallelism).
	Workers int
}

// Rerank scores each candidate with gen and returns the slice
// sorted by Score desc. Uses ctx for cancellation; falls through to
// gen.Generate (which doesn't yet take a ctx in the gollama wrapper)
// but checks ctx.Err() between candidates so a SIGINT during a long
// rerank pass aborts cleanly.
func Rerank(ctx context.Context, gen llm.Generator, query string, candidates []store.SearchResult, opts Options) ([]Scored, error) {
	if gen == nil {
		return nil, fmt.Errorf("rerank: generator is required")
	}
	topN := opts.TopN
	if topN <= 0 {
		topN = DefaultTopN
	}
	if topN > len(candidates) {
		topN = len(candidates)
	}
	budget := opts.PassageBudget
	if budget <= 0 {
		budget = DefaultPassageBudget
	}
	maxTokens := opts.MaxTokens
	if maxTokens <= 0 {
		maxTokens = DefaultMaxTokens
	}

	out := make([]Scored, len(candidates))
	for i, c := range candidates {
		out[i] = Scored{Result: c, Score: 0.5, BlendedScore: 0, RRFRank: i}
	}

	scoreOne := func(i int) {
		passage := truncateRunes(candidates[i].Snippet, budget)
		prompt := fmt.Sprintf(PromptTemplate, query, passage)
		raw, err := gen.Generate(prompt, llm.WithMaxTokens(maxTokens))
		if err != nil {
			// Treat per-call errors as "uncertain" — leaves Score at 0.5
			// so the position-aware blender falls back to RRF rank
			// for this candidate.
			return
		}
		out[i].Score = parseYesNo(raw)
	}

	workers := opts.Workers
	if workers <= 1 {
		for i := 0; i < topN; i++ {
			if ctx.Err() != nil {
				return out, ctx.Err()
			}
			scoreOne(i)
		}
	} else {
		jobs := make(chan int, topN)
		for i := 0; i < topN; i++ {
			jobs <- i
		}
		close(jobs)
		var wg sync.WaitGroup
		wg.Add(workers)
		for w := 0; w < workers; w++ {
			go func() {
				defer wg.Done()
				for i := range jobs {
					if ctx.Err() != nil {
						return
					}
					scoreOne(i)
				}
			}()
		}
		wg.Wait()
		if ctx.Err() != nil {
			return out, ctx.Err()
		}
	}

	// Stable sort: rerank score desc, ties broken by original RRF
	// rank (so the input order survives within a tied verdict
	// bucket).
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].Score != out[j].Score {
			return out[i].Score > out[j].Score
		}
		return out[i].RRFRank < out[j].RRFRank
	})
	return out, nil
}

// parseYesNo returns 1.0 for a "yes" answer, 0.0 for "no", and 0.5
// when the model emits something else. Lowercases + scans the first
// few hundred chars; we don't insist on the FIRST token being a
// yes/no because chatty models sometimes echo the question.
func parseYesNo(raw string) float64 {
	low := strings.ToLower(raw)
	// Scan window — long enough for a chatty echo + the answer.
	if len(low) > 256 {
		low = low[:256]
	}
	yes := indexWord(low, "yes")
	no := indexWord(low, "no")
	switch {
	case yes >= 0 && (no < 0 || yes < no):
		return 1.0
	case no >= 0:
		return 0.0
	default:
		return 0.5
	}
}

// indexWord returns the byte index of the first whole-word occurrence
// of word in s, or -1 if absent. Whole-word means surrounded by
// non-letter bytes (or string boundaries) so "no" doesn't match
// inside "noticed" or "another".
func indexWord(s, word string) int {
	for i := 0; i+len(word) <= len(s); i++ {
		if s[i:i+len(word)] != word {
			continue
		}
		// Check left boundary.
		if i > 0 && isLetter(s[i-1]) {
			continue
		}
		// Check right boundary.
		if i+len(word) < len(s) && isLetter(s[i+len(word)]) {
			continue
		}
		return i
	}
	return -1
}

func isLetter(b byte) bool {
	return (b >= 'a' && b <= 'z') || (b >= 'A' && b <= 'Z')
}

// truncateRunes caps s at n runes, appending an ellipsis when it
// actually trims. Avoids byte-slicing so multi-byte UTF-8 sequences
// survive intact.
func truncateRunes(s string, n int) string {
	if n <= 0 {
		return s
	}
	count := 0
	for i := range s {
		if count == n {
			return s[:i] + "…"
		}
		count++
	}
	return s
}
