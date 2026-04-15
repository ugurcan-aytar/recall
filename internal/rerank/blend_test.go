package rerank

import (
	"math"
	"testing"

	"github.com/ugurcan-aytar/recall/internal/store"
)

func TestPickBandMatchesQmdDefaults(t *testing.T) {
	cases := []struct {
		rank          int
		wantRRFWeight float64
	}{
		{0, 0.75},  // top-1
		{2, 0.75},  // edge of top-3
		{3, 0.60},  // start of top-4-10
		{9, 0.60},  // edge of top-10
		{10, 0.40}, // start of top-11+
		{50, 0.40}, // far tail
	}
	for _, tc := range cases {
		b := pickBand(tc.rank, DefaultBlendBands)
		if b.RRFWeight != tc.wantRRFWeight {
			t.Errorf("rank %d → RRFWeight %g, want %g", tc.rank, b.RRFWeight, tc.wantRRFWeight)
		}
	}
}

func TestPositionAwareBlendEmpty(t *testing.T) {
	if got := PositionAwareBlend(nil, DefaultBlendBands); got != nil {
		t.Errorf("nil input → %+v, want nil", got)
	}
}

// TestPositionAwareBlendSingleCandidate is the degenerate case: with
// only one input, RRFRank is always 0 and rrfNorm collapses to 1.0.
// blended = 0.75*1 + 0.25*Score. Used to be an off-by-one trap when
// the normalisation divided by n instead of n-1.
func TestPositionAwareBlendSingleCandidate(t *testing.T) {
	in := []Scored{{Result: store.SearchResult{DocID: "a"}, Score: 1.0, RRFRank: 0}}
	got := PositionAwareBlend(in, DefaultBlendBands)
	if len(got) != 1 {
		t.Fatalf("len = %d, want 1", len(got))
	}
	want := 0.75*1.0 + 0.25*1.0 // = 1.0
	if math.Abs(got[0].BlendedScore-want) > 1e-9 {
		t.Errorf("BlendedScore = %g, want %g", got[0].BlendedScore, want)
	}
}

// TestPositionAwareBlendRRFProtectsTopHits is the headline guarantee:
// a top-rank doc the reranker dislikes still beats a deep-rank doc
// the reranker agrees with — IF the deep doc is far enough back that
// even the boost can't overtake. Within that "RRF protects" zone the
// blended order should mirror the RRF order.
func TestPositionAwareBlendRRFProtectsTopHits(t *testing.T) {
	// 5 candidates, all at the same reranker verdict (no), so the
	// only differentiator is RRF rank. Blended order must match RRF
	// rank order ascending (best rank → top of result list).
	in := []Scored{
		{Result: store.SearchResult{DocID: "a"}, Score: 0, RRFRank: 0},
		{Result: store.SearchResult{DocID: "b"}, Score: 0, RRFRank: 1},
		{Result: store.SearchResult{DocID: "c"}, Score: 0, RRFRank: 2},
		{Result: store.SearchResult{DocID: "d"}, Score: 0, RRFRank: 3},
		{Result: store.SearchResult{DocID: "e"}, Score: 0, RRFRank: 4},
	}
	got := PositionAwareBlend(in, DefaultBlendBands)
	for i, s := range got {
		want := string(rune('a' + i))
		if s.Result.DocID != want {
			t.Errorf("position %d = %s, want %s (full = %+v)", i, s.Result.DocID, want, got)
			break
		}
	}
}

// TestPositionAwareBlendRerankerLiftsAcrossNoVotes verifies the
// reranker's "correction" property: when everything else is "no", a
// single yes vote — even one buried deep in the RRF tail — moves
// to the front. With qmd's weights a yes vote contributes ≥0.6
// regardless of band, while every no vote tops out at 0.75 (top-3)
// and shrinks fast at deeper bands. So the lone yes overtakes
// every no.
func TestPositionAwareBlendRerankerLiftsAcrossNoVotes(t *testing.T) {
	in := make([]Scored, 30)
	for i := range in {
		in[i] = Scored{Result: store.SearchResult{DocID: lblFromInt(i)}, Score: 0, RRFRank: i}
	}
	in[10].Score = 1.0 // the one yes voter, deep in the tail

	got := PositionAwareBlend(in, DefaultBlendBands)
	if got[0].RRFRank != 10 {
		t.Errorf("expected rank-10 yes vote at position 0; got rank %d (full = %+v)",
			got[0].RRFRank, got)
	}
}

// TestPositionAwareBlendProtectsTopAgainstSameVerdictTail captures
// the OTHER half of the property: at the SAME reranker verdict, a
// top-3 doc must outrank a tail doc. RRF rank breaks ties between
// equally-confident reranker votes.
func TestPositionAwareBlendProtectsTopAgainstSameVerdictTail(t *testing.T) {
	// All no votes: top-3 must keep the top of the result list.
	in := make([]Scored, 30)
	for i := range in {
		in[i] = Scored{Result: store.SearchResult{DocID: lblFromInt(i)}, Score: 0, RRFRank: i}
	}
	got := PositionAwareBlend(in, DefaultBlendBands)
	for i := 0; i < 3; i++ {
		if got[i].RRFRank != i {
			t.Errorf("at all-no, blended position %d = RRFRank %d, want %d (full = %+v)",
				i, got[i].RRFRank, i, got)
			break
		}
	}

	// All yes votes: same property — top-3 stays on top.
	in2 := make([]Scored, 30)
	for i := range in2 {
		in2[i] = Scored{Result: store.SearchResult{DocID: lblFromInt(i)}, Score: 1.0, RRFRank: i}
	}
	got2 := PositionAwareBlend(in2, DefaultBlendBands)
	for i := 0; i < 3; i++ {
		if got2[i].RRFRank != i {
			t.Errorf("at all-yes, blended position %d = RRFRank %d, want %d", i, got2[i].RRFRank, i)
			break
		}
	}
}

// TestPositionAwareBlendCustomBands proves the BlendBands argument
// is honoured. With a degenerate "100% reranker" band, blended order
// reduces to the rerank score order regardless of RRFRank.
func TestPositionAwareBlendCustomBands(t *testing.T) {
	allReranker := BlendBands{
		{RRFRankUpper: 1 << 31, RRFWeight: 0, RerankerWeight: 1.0},
	}
	in := []Scored{
		{Result: store.SearchResult{DocID: "top-no"}, Score: 0, RRFRank: 0},
		{Result: store.SearchResult{DocID: "deep-yes"}, Score: 1, RRFRank: 20},
	}
	got := PositionAwareBlend(in, allReranker)
	if got[0].Result.DocID != "deep-yes" {
		t.Errorf("first = %s, want deep-yes (100%% reranker weights)", got[0].Result.DocID)
	}
}

func lblFromInt(i int) string {
	if i < 10 {
		return string(rune('0' + i))
	}
	return string(rune('a' + i - 10))
}
