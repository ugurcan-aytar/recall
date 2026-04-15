package rerank

import "sort"

// BlendBands sets the (rrfWeight, rerankerWeight) pairs applied to
// each candidate based on its original RRF rank. qmd's defaults:
//
//   top 1-3   → 75% RRF / 25% reranker (RRF protects strong hits)
//   top 4-10  → 60% RRF / 40% reranker
//   top 11+   → 40% RRF / 60% reranker (reranker can promote)
//
// Custom bands let callers retune per use case. Bands MUST be
// sorted by RRFRankUpper ascending; the first band whose upper
// bound covers the candidate's rank wins. The last band typically
// has RRFRankUpper = math.MaxInt to act as the "rank 11+" catch-all.
type BlendBands []BlendBand

// BlendBand describes one rank-range weighting rule.
type BlendBand struct {
	// RRFRankUpper is the inclusive upper bound (0-based) of the
	// rank range this band covers. A band with RRFRankUpper=2
	// catches RRF ranks 0, 1, 2 (the "top 3").
	RRFRankUpper int

	// RRFWeight is multiplied with the normalised RRF rank score.
	RRFWeight float64

	// RerankerWeight is multiplied with the binary rerank verdict
	// (Scored.Score).
	RerankerWeight float64
}

// DefaultBlendBands matches qmd's published weights. Document changes
// here as a semver-minor event — the values directly shape what
// `recall query --rerank` returns.
var DefaultBlendBands = BlendBands{
	{RRFRankUpper: 2, RRFWeight: 0.75, RerankerWeight: 0.25},
	{RRFRankUpper: 9, RRFWeight: 0.60, RerankerWeight: 0.40},
	{RRFRankUpper: 1 << 31, RRFWeight: 0.40, RerankerWeight: 0.60},
}

// PositionAwareBlend computes Scored.BlendedScore for each input
// using the band weights, then re-sorts the slice by BlendedScore
// descending (ties broken by original RRFRank). Returns a NEW
// slice; the input is not mutated.
//
// The RRF half of the blend is a linear normalisation of RRFRank:
// rank 0 maps to 1.0, the worst rank maps to 0. This gives the
// "RRF protects high-confidence hits" property — a top-3 doc that
// the reranker disagrees with still scores ~0.75, while a rank-15
// doc the reranker agrees with scores ~0.625 (computed against
// DefaultBlendBands and a 30-candidate input).
func PositionAwareBlend(scored []Scored, bands BlendBands) []Scored {
	if len(scored) == 0 {
		return nil
	}
	if len(bands) == 0 {
		bands = DefaultBlendBands
	}

	n := len(scored)
	out := make([]Scored, n)
	for i, s := range scored {
		rrfNorm := 1.0
		if n > 1 {
			rrfNorm = 1.0 - float64(s.RRFRank)/float64(n-1)
		}
		w := pickBand(s.RRFRank, bands)
		blended := w.RRFWeight*rrfNorm + w.RerankerWeight*s.Score
		out[i] = Scored{
			Result:       s.Result,
			Score:        s.Score,
			BlendedScore: blended,
			RRFRank:      s.RRFRank,
		}
	}
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].BlendedScore != out[j].BlendedScore {
			return out[i].BlendedScore > out[j].BlendedScore
		}
		return out[i].RRFRank < out[j].RRFRank
	})
	return out
}

// pickBand returns the first band whose upper bound covers rank.
// Falls through to the last band when nothing matches (the standard
// "catch-all" pattern with RRFRankUpper=MaxInt).
func pickBand(rank int, bands BlendBands) BlendBand {
	for _, b := range bands {
		if rank <= b.RRFRankUpper {
			return b
		}
	}
	return bands[len(bands)-1]
}
