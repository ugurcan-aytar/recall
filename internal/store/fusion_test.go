package store

import (
	"math"
	"testing"
)

func mkResult(coll, path string, score float64) SearchResult {
	return SearchResult{
		DocID: path[:1], Path: path, CollectionName: coll, Score: score,
		Title: path,
	}
}

func TestRRFBasic(t *testing.T) {
	bm25 := []SearchResult{
		mkResult("c", "a.md", 0.9),
		mkResult("c", "b.md", 0.8),
		mkResult("c", "x.md", 0.1),
	}
	vec := []SearchResult{
		mkResult("c", "b.md", 0.95),
		mkResult("c", "a.md", 0.7),
	}
	out := FuseRRF(bm25, vec, DefaultFusionOptions())
	if len(out) == 0 {
		t.Fatal("no fused output")
	}
	// "b.md" appears at rank 1 of vector AND rank 2 of bm25 → very high.
	// "a.md" appears at rank 1 of bm25 AND rank 2 of vector → also high.
	// One of these wins; the floor (40% of top) should retain at least both.
	top := out[0].Path
	if top != "a.md" && top != "b.md" {
		t.Errorf("unexpected top result: %s", top)
	}
}

func TestRRFTopRankBonus(t *testing.T) {
	opts := DefaultFusionOptions()
	// One document at rank 1 of one list, another at ranks 2 of both lists.
	bm25 := []SearchResult{mkResult("c", "boost.md", 1)}
	vec := []SearchResult{
		mkResult("c", "other.md", 1),
		mkResult("c", "boost.md", 0.5),
	}
	out := FuseRRF(bm25, vec, opts)
	var boost, other *FusedResult
	for i := range out {
		switch out[i].Path {
		case "boost.md":
			boost = &out[i]
		case "other.md":
			other = &out[i]
		}
	}
	if boost == nil || other == nil {
		t.Fatalf("expected both docs in fused output, got %+v", out)
	}
	if boost.Trace.BonusUsed != opts.TopRankBonusOne {
		t.Errorf("boost bonus = %.4f, want %.4f", boost.Trace.BonusUsed, opts.TopRankBonusOne)
	}
	if other.Trace.BonusUsed != opts.TopRankBonusOne {
		t.Errorf("other bonus = %.4f, want %.4f", other.Trace.BonusUsed, opts.TopRankBonusOne)
	}
}

func TestRRFRank2or3GetsSmallerBonus(t *testing.T) {
	opts := DefaultFusionOptions()
	bm25 := []SearchResult{
		mkResult("c", "a.md", 1),
		mkResult("c", "b.md", 0.5),
		mkResult("c", "c.md", 0.3),
	}
	out := FuseRRF(bm25, nil, opts)
	if len(out) < 3 {
		t.Fatalf("want 3 fused, got %d", len(out))
	}
	for _, r := range out {
		switch r.Path {
		case "a.md":
			if r.Trace.BonusUsed != opts.TopRankBonusOne {
				t.Errorf("rank-1 bonus = %.4f, want %.4f", r.Trace.BonusUsed, opts.TopRankBonusOne)
			}
		case "b.md", "c.md":
			if r.Trace.BonusUsed != opts.TopRankBonusTwoThree {
				t.Errorf("rank-2/3 bonus on %s = %.4f, want %.4f", r.Path, r.Trace.BonusUsed, opts.TopRankBonusTwoThree)
			}
		}
	}
}

func TestRRFOriginalQueryWeight(t *testing.T) {
	// Concrete RRF check: with K=60 and a single top hit in BM25,
	// rrf = 1/(60+1) ≈ 0.0163934426.
	opts := DefaultFusionOptions()
	bm25 := []SearchResult{mkResult("c", "a.md", 1)}
	out := FuseRRF(bm25, nil, opts)
	if len(out) != 1 {
		t.Fatalf("want 1 result, got %d", len(out))
	}
	expectedRRF := 1.0 / (opts.K + 1)
	if math.Abs(out[0].Trace.RRFOnly-expectedRRF) > 1e-9 {
		t.Errorf("RRFOnly = %.10f, want %.10f", out[0].Trace.RRFOnly, expectedRRF)
	}
	if math.Abs(out[0].FusedScore-(expectedRRF+opts.TopRankBonusOne)) > 1e-9 {
		t.Errorf("FusedScore = %.10f, want %.10f",
			out[0].FusedScore, expectedRRF+opts.TopRankBonusOne)
	}
}

func TestRRFDisjointLists(t *testing.T) {
	bm25 := []SearchResult{
		mkResult("c", "a.md", 1),
		mkResult("c", "b.md", 0.8),
	}
	vec := []SearchResult{
		mkResult("c", "x.md", 0.95),
		mkResult("c", "y.md", 0.7),
	}
	out := FuseRRF(bm25, vec, DefaultFusionOptions())
	// All four should appear after fusion (subject to floor).
	got := map[string]bool{}
	for _, r := range out {
		got[r.Path] = true
	}
	for _, p := range []string{"a.md", "b.md", "x.md", "y.md"} {
		if !got[p] {
			t.Errorf("%s missing from fused output", p)
		}
	}
}

func TestRRFIdenticalLists(t *testing.T) {
	list := []SearchResult{
		mkResult("c", "a.md", 1),
		mkResult("c", "b.md", 0.5),
	}
	out := FuseRRF(list, list, DefaultFusionOptions())

	// "a.md" appears at rank 1 of BOTH lists → rrf = 2 × 1/61 ≈ 0.0327869,
	// plus the top-rank bonus (rank 1 in either ⇒ +0.05).
	if out[0].Path != "a.md" {
		t.Errorf("top = %s, want a.md", out[0].Path)
	}
	expected := 2.0/(60+1) + 0.05
	if math.Abs(out[0].FusedScore-expected) > 1e-9 {
		t.Errorf("identical-list fusion = %.10f, want %.10f", out[0].FusedScore, expected)
	}
}

func TestRRFEmptyList(t *testing.T) {
	bm25 := []SearchResult{mkResult("c", "a.md", 1)}
	out := FuseRRF(bm25, nil, DefaultFusionOptions())
	if len(out) != 1 || out[0].Path != "a.md" {
		t.Fatalf("empty-vector branch: %+v", out)
	}

	out2 := FuseRRF(nil, bm25, DefaultFusionOptions())
	if len(out2) != 1 || out2[0].Path != "a.md" {
		t.Fatalf("empty-bm25 branch: %+v", out2)
	}

	out3 := FuseRRF(nil, nil, DefaultFusionOptions())
	if len(out3) != 0 {
		t.Errorf("both empty: got %+v", out3)
	}
}

func TestRRFAdaptiveFloor(t *testing.T) {
	opts := DefaultFusionOptions()
	// Build a list where the lowest entries should be filtered by floor.
	// Top: bonus 0.05 + 1/61 ≈ 0.066. Low: 1/65 ≈ 0.0154 (no bonus, rank 5).
	// Floor at 40% of top ≈ 0.0264 → low gets cut.
	bm25 := []SearchResult{
		mkResult("c", "1.md", 1),
		mkResult("c", "2.md", 1),
		mkResult("c", "3.md", 1),
		mkResult("c", "4.md", 1),
		mkResult("c", "5.md", 1),
	}
	out := FuseRRF(bm25, nil, opts)
	if len(out) >= 5 {
		t.Errorf("floor did not trim; got %d results", len(out))
	}
}

func TestDefaultFusionOptions(t *testing.T) {
	o := DefaultFusionOptions()
	if o.K != 60 {
		t.Errorf("K = %v", o.K)
	}
	if o.TopRankBonusOne != 0.05 {
		t.Errorf("TopRankBonusOne = %v", o.TopRankBonusOne)
	}
	if o.TopRankBonusTwoThree != 0.02 {
		t.Errorf("TopRankBonusTwoThree = %v", o.TopRankBonusTwoThree)
	}
	if o.MinScoreFloorPct != 0.4 {
		t.Errorf("MinScoreFloorPct = %v", o.MinScoreFloorPct)
	}
}

func TestMergeRankListsEmpty(t *testing.T) {
	if got := MergeRankLists(nil, DefaultFusionOptions(), 0); got != nil {
		t.Errorf("nil input → %+v, want nil", got)
	}
}

func TestMergeRankListsSinglePassThrough(t *testing.T) {
	in := []SearchResult{mkResult("c", "a.md", 0.9), mkResult("c", "b.md", 0.5)}
	got := MergeRankLists([][]SearchResult{in}, DefaultFusionOptions(), 0)
	if len(got) != 2 || got[0].Path != "a.md" {
		t.Errorf("single-list = %+v, want pass-through", got)
	}
}

func TestMergeRankListsUnionsByDoc(t *testing.T) {
	listA := []SearchResult{mkResult("c", "a.md", 0.9), mkResult("c", "b.md", 0.5)}
	listB := []SearchResult{mkResult("c", "b.md", 0.4), mkResult("c", "c.md", 0.3)}
	got := MergeRankLists([][]SearchResult{listA, listB}, DefaultFusionOptions(), 0)
	if len(got) != 3 {
		t.Fatalf("merge = %d docs, want 3", len(got))
	}
	// b.md appears in both lists at rank 1 (listA) and rank 0 (listB);
	// it should outrank the docs that appear in only one list.
	if got[0].Path != "b.md" {
		t.Errorf("expected b.md (in both lists) on top, got %s; full = %+v", got[0].Path, got)
	}
}

func TestMergeRankListsFirstListBonus(t *testing.T) {
	// Same shape on both lists: doc x at rank 0, doc y at rank 1.
	// With a first-list bonus, the original list's rank-0 doc must
	// stay ahead even when the variant pushes y above its own x.
	listA := []SearchResult{mkResult("c", "x.md", 0), mkResult("c", "y.md", 0)}
	listB := []SearchResult{mkResult("c", "y.md", 0), mkResult("c", "x.md", 0)}

	noBonus := MergeRankLists([][]SearchResult{listA, listB}, DefaultFusionOptions(), 0)
	withBonus := MergeRankLists([][]SearchResult{listA, listB}, DefaultFusionOptions(), 1.0)
	// Without bonus the two docs are symmetric so order can go either
	// way (we just check both are present).
	if len(noBonus) != 2 || len(withBonus) != 2 {
		t.Fatalf("unexpected lengths: noBonus=%d withBonus=%d", len(noBonus), len(withBonus))
	}
	// With bonus, x (rank 0 in listA) should outrank y (rank 0 in listB
	// but only rank 1 in listA).
	if withBonus[0].Path != "x.md" {
		t.Errorf("first-list bonus should put x.md on top; got %s", withBonus[0].Path)
	}
}

func TestMergeRankListsSampleFromBestRank(t *testing.T) {
	// Doc x ranks 1 in listA but 0 in listB; the snippet/title we
	// keep should come from listB (better rank).
	listA := []SearchResult{
		{DocID: "noise", Path: "noise.md", CollectionName: "c"},
		{DocID: "x", Path: "x.md", Title: "from-A", Snippet: "A-snippet", CollectionName: "c"},
	}
	listB := []SearchResult{
		{DocID: "x", Path: "x.md", Title: "from-B", Snippet: "B-snippet", CollectionName: "c"},
	}
	got := MergeRankLists([][]SearchResult{listA, listB}, DefaultFusionOptions(), 0)
	var x SearchResult
	for _, r := range got {
		if r.DocID == "x" {
			x = r
			break
		}
	}
	if x.Title != "from-B" {
		t.Errorf("Title = %q, want from-B (better rank)", x.Title)
	}
	if x.Snippet != "B-snippet" {
		t.Errorf("Snippet = %q, want B-snippet", x.Snippet)
	}
}
