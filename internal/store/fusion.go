package store

import (
	"sort"
)

// FusionOptions controls the RRF + bonus + adaptive-floor pipeline. Zero
// values use defaults documented in ROADMAP.md Phase R3.7.
type FusionOptions struct {
	// K is the rank-shift constant in 1/(K+rank). qmd / Cormack et al.
	// use 60.
	K float64

	// TopRankBonusOne is added to the fused score of the top-ranked result
	// in either input list.
	TopRankBonusOne float64

	// TopRankBonusTwoThree is added for ranks 2 and 3.
	TopRankBonusTwoThree float64

	// MinScoreFloorPct is the adaptive cutoff: any fused result whose score
	// is below MinScoreFloorPct × top.Score is dropped. 0.4 = 40%.
	MinScoreFloorPct float64
}

// DefaultFusionOptions matches the values ROADMAP R3.7 spells out.
func DefaultFusionOptions() FusionOptions {
	return FusionOptions{
		K:                    60,
		TopRankBonusOne:      0.05,
		TopRankBonusTwoThree: 0.02,
		MinScoreFloorPct:     0.4,
	}
}

// MergeRankLists folds N same-side result lists (e.g. BM25 results
// from the original query plus its expansion variants) into one
// ranked list. Each doc's score becomes the RRF-style sum of
// 1/(k+rank) across every list it appears in, with k taken from
// FusionOptions.K. The first list is given an extra weight of
// firstListBonus on top of its rank contribution — this matches qmd's
// "first query gets 2× weight" rule and protects the user's literal
// query from being out-voted by aggressive expansions.
//
// The Score field on each output SearchResult is overwritten with
// the merged rank-fusion score (callers that need the absolute BM25
// or vector score should look at the input lists). The Snippet,
// Title, Path, etc. come from whichever list ranked the doc highest.
func MergeRankLists(lists [][]SearchResult, opts FusionOptions, firstListBonus float64) []SearchResult {
	if len(lists) == 0 {
		return nil
	}
	if len(lists) == 1 {
		return append([]SearchResult(nil), lists[0]...)
	}
	k := opts.K
	if k <= 0 {
		k = 60
	}

	scores := map[string]float64{}
	bestSample := map[string]SearchResult{}
	bestRank := map[string]int{}

	for li, list := range lists {
		bonus := 1.0
		if li == 0 && firstListBonus > 0 {
			bonus = 1.0 + firstListBonus
		}
		for rank, r := range list {
			doc := docKey(r)
			scores[doc] += bonus / (k + float64(rank+1))
			if cur, ok := bestRank[doc]; !ok || rank < cur {
				bestRank[doc] = rank
				bestSample[doc] = r
			}
		}
	}

	out := make([]SearchResult, 0, len(scores))
	for doc, s := range scores {
		r := bestSample[doc]
		r.Score = s
		out = append(out, r)
	}
	sort.Slice(out, func(i, j int) bool {
		return out[i].Score > out[j].Score
	})
	return out
}

// docKey returns a stable identifier for a SearchResult suitable for
// cross-list deduplication. Falls back to path+collection when the
// upstream doc_id is empty (rare — should only happen for synthetic
// fixtures in tests).
func docKey(r SearchResult) string {
	if r.DocID != "" {
		return r.DocID
	}
	return r.CollectionName + "/" + r.Path
}

// FusedResult is one row of fused output. Trace is populated when the
// caller requests an --explain breakdown.
type FusedResult struct {
	SearchResult
	FusedScore float64
	Trace      FusionTrace
}

// FusionTrace records each input list's contribution.
type FusionTrace struct {
	BM25Rank   int     // 0 ⇒ not present in BM25 list
	BM25Score  float64
	VecRank    int     // 0 ⇒ not present in vector list
	VecScore   float64
	BonusUsed  float64
	FloorUsed  float64
	RRFOnly    float64 // fused score before bonus / floor
}

// FuseRRF combines BM25 and vector result lists with Reciprocal Rank
// Fusion plus a top-rank bonus and an adaptive minimum-score floor.
//
//	RRF: score(d) = Σ 1 / (K + rank(d, list) + 1)         — k=60 default
//	Bonus:    +TopRankBonusOne for rank 1 of either list,
//	          +TopRankBonusTwoThree for ranks 2-3.
//	Floor:    drop results with score < MinScoreFloorPct × top.Score.
//
// The function deduplicates by (CollectionName, Path) — the same document
// appearing in both lists adds both rank contributions.
func FuseRRF(bm25, vec []SearchResult, opts FusionOptions) []FusedResult {
	if opts.K <= 0 {
		opts.K = 60
	}
	if opts.MinScoreFloorPct < 0 {
		opts.MinScoreFloorPct = 0
	}

	type acc struct {
		res       SearchResult
		rrf       float64
		bm25Rank  int
		bm25Score float64
		vecRank   int
		vecScore  float64
	}
	merged := map[string]*acc{}

	addList := func(list []SearchResult, isBM25 bool) {
		for i, r := range list {
			rank := i + 1
			key := r.CollectionName + "/" + r.Path
			a, ok := merged[key]
			if !ok {
				a = &acc{res: r}
				merged[key] = a
			}
			a.rrf += 1.0 / (opts.K + float64(rank))
			if isBM25 {
				a.bm25Rank = rank
				a.bm25Score = r.Score
				// BM25 snippet wins (it has match markers).
				a.res.Snippet = r.Snippet
			} else {
				a.vecRank = rank
				a.vecScore = r.Score
				if a.res.Snippet == "" {
					a.res.Snippet = r.Snippet
				}
			}
		}
	}
	addList(bm25, true)
	addList(vec, false)

	out := make([]FusedResult, 0, len(merged))
	for _, a := range merged {
		bonus := 0.0
		if a.bm25Rank == 1 || a.vecRank == 1 {
			bonus = opts.TopRankBonusOne
		} else if isTopThree(a.bm25Rank) || isTopThree(a.vecRank) {
			bonus = opts.TopRankBonusTwoThree
		}
		out = append(out, FusedResult{
			SearchResult: a.res,
			FusedScore:   a.rrf + bonus,
			Trace: FusionTrace{
				BM25Rank:  a.bm25Rank,
				BM25Score: a.bm25Score,
				VecRank:   a.vecRank,
				VecScore:  a.vecScore,
				BonusUsed: bonus,
				RRFOnly:   a.rrf,
			},
		})
	}
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].FusedScore == out[j].FusedScore {
			// Tie-break: prefer the doc that ranked higher in BM25
			// (BM25 ties have unambiguous ordering, vector cosine doesn't).
			return better(out[i].Trace.BM25Rank, out[j].Trace.BM25Rank)
		}
		return out[i].FusedScore > out[j].FusedScore
	})

	if len(out) == 0 {
		return out
	}

	// Adaptive floor.
	floor := out[0].FusedScore * opts.MinScoreFloorPct
	for i := range out {
		out[i].Trace.FloorUsed = floor
	}
	if floor <= 0 {
		return out
	}
	cut := 0
	for cut < len(out) && out[cut].FusedScore >= floor {
		cut++
	}
	return out[:cut]
}

// isTopThree returns true for ranks 2 or 3. (Rank 1 is handled separately
// because its bonus is larger.)
func isTopThree(rank int) bool { return rank == 2 || rank == 3 }

// better returns true when rank a is "better" than rank b. Rank 0 means
// "absent from this list" and loses to any positive rank.
func better(a, b int) bool {
	switch {
	case a == 0 && b == 0:
		return false
	case a == 0:
		return false
	case b == 0:
		return true
	default:
		return a < b
	}
}
