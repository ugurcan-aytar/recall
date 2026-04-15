// Package rerank scores (query, passage) pairs through a cross-
// encoder reranker model and returns the candidate list sorted by
// relevance. v0.2.4 swapped the pre-existing binary-yes/no prompt
// fallback for a real cross-encoder via llama.cpp's /v1/rerank
// endpoint and bge-reranker-v2-m3 — continuous gradient scores
// instead of noisy 0/1 verdicts.
//
// The raw cross-encoder score is a logit (roughly [-12, +8] for
// bge-reranker-v2-m3). We min-max normalise within the candidate
// set to [0,1] before feeding the position-aware blender, so the
// blender's weights keep their documented semantics (RRF weight +
// reranker weight sum to 1.0 per band).
package rerank

import (
	"context"
	"fmt"
	"sort"

	"github.com/ugurcan-aytar/recall/internal/llm"
	"github.com/ugurcan-aytar/recall/internal/store"
)

// Default knobs.
const (
	// DefaultTopN is how many candidates go through the reranker.
	// The first N RRF hits get the cross-encoder treatment;
	// anything past N keeps its RRF rank unchanged. 30 mirrors
	// qmd's window and is well within a 200-300 ms budget for
	// bge-reranker-v2-m3 Q4_K_M on Apple Silicon (batched in a
	// single /v1/rerank call).
	DefaultTopN = 30

	// DefaultPassageBudget caps passage length (in runes) before
	// it's sent to the reranker. Long passages balloon KV cache
	// with no quality gain — cross-encoders look at sentence-level
	// relevance, not whole chapters. 800 runes ≈ 200 tokens.
	DefaultPassageBudget = 800
)

// Scored pairs an input candidate with its rerank score (raw logit
// from the cross-encoder) and the position-aware fused result
// (BlendedScore, populated by PositionAwareBlend).
//
// Score is the normalised [0,1] relevance (min-max within the
// candidate set). RawScore preserves the pre-normalisation logit
// for --explain traces and downstream debugging. RRFRank is the
// 0-based rank in the input, used by the blender to pick a
// weighting band.
type Scored struct {
	Result       store.SearchResult
	Score        float64
	RawScore     float64
	BlendedScore float64
	RRFRank      int
}

// Options tweaks one Rerank call.
type Options struct {
	// TopN limits how many candidates go through the reranker.
	// 0 ⇒ DefaultTopN. Candidates beyond TopN stay at Score=0.5
	// (middle of the normalised range) so the blender falls back
	// to RRF rank for the tail.
	TopN int

	// PassageBudget caps each passage at this many runes before
	// submission. 0 ⇒ DefaultPassageBudget.
	PassageBudget int
}

// Rerank submits the top-N candidates to the cross-encoder in one
// /v1/rerank call, normalises the returned logits to [0,1] within
// the candidate set, and returns the slice sorted by Score desc.
// Candidates past TopN keep their input order and get Score=0.5
// so the blender treats them as "uncertain".
func Rerank(ctx context.Context, rr llm.Reranker, query string, candidates []store.SearchResult, opts Options) ([]Scored, error) {
	if rr == nil {
		return nil, fmt.Errorf("rerank: reranker is required")
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

	out := make([]Scored, len(candidates))
	for i, c := range candidates {
		out[i] = Scored{Result: c, Score: 0.5, RawScore: 0, RRFRank: i}
	}

	if topN == 0 {
		return out, nil
	}

	passages := make([]string, topN)
	for i := 0; i < topN; i++ {
		passages[i] = truncateRunes(candidates[i].Snippet, budget)
	}

	raw, err := rr.Rerank(ctx, query, passages)
	if err != nil {
		// Leave the default Score=0.5 in place so the blender
		// falls back to RRF; caller gets the error so they can
		// surface it as a warning.
		return out, fmt.Errorf("rerank: %w", err)
	}
	if len(raw) != topN {
		return out, fmt.Errorf("rerank: got %d scores for %d candidates", len(raw), topN)
	}

	// Min-max normalise the returned logits into [0,1] within the
	// candidate set. Cross-encoder logits have no fixed range —
	// bge-reranker-v2-m3 in particular spans roughly [-12, +8] —
	// so feeding them raw into the position-aware blender would
	// swamp the RRF contribution. Normalisation preserves ordering
	// AND keeps the blender's documented "per-band weights sum to
	// 1.0" semantics intact.
	minScore, maxScore := raw[0], raw[0]
	for _, v := range raw {
		if v < minScore {
			minScore = v
		}
		if v > maxScore {
			maxScore = v
		}
	}
	span := maxScore - minScore
	for i := 0; i < topN; i++ {
		out[i].RawScore = raw[i]
		if span > 0 {
			out[i].Score = (raw[i] - minScore) / span
		} else {
			// All scores equal — no signal to rank by. Keep the
			// 0.5 default so RRF carries.
			out[i].Score = 0.5
		}
	}

	sort.SliceStable(out, func(i, j int) bool {
		if out[i].Score != out[j].Score {
			return out[i].Score > out[j].Score
		}
		return out[i].RRFRank < out[j].RRFRank
	})
	return out, nil
}

// truncateRunes caps s at n runes, appending an ellipsis when it
// actually trims. Byte-slicing would corrupt multi-byte UTF-8.
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
