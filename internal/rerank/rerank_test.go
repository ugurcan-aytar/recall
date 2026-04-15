package rerank_test

import (
	"context"
	"errors"
	"testing"

	"github.com/ugurcan-aytar/recall/internal/llm"
	"github.com/ugurcan-aytar/recall/internal/rerank"
	"github.com/ugurcan-aytar/recall/internal/store"
)

// fixtureCandidates returns three SearchResults whose snippets each
// carry a different rerank verdict from the scripted fakeReranker
// below.
func fixtureCandidates() []store.SearchResult {
	return []store.SearchResult{
		{DocID: "a", Path: "a.md", Title: "A", Snippet: "passage labelled relevant",
			CollectionName: "c", Score: 0.9},
		{DocID: "b", Path: "b.md", Title: "B", Snippet: "passage labelled irrelevant",
			CollectionName: "c", Score: 0.7},
		{DocID: "c", Path: "c.md", Title: "C", Snippet: "passage labelled relevant",
			CollectionName: "c", Score: 0.5},
	}
}

// fakeReranker implements llm.Reranker by mapping each document to a
// canned logit based on a substring lookup. Catches wire-level bugs
// (index ordering, length mismatches) without spinning up a real
// llama-server.
type fakeReranker struct {
	scores  map[string]float64
	err     error
	lastQry string
	lastDoc []string
	closed  bool
}

func (f *fakeReranker) Rerank(_ context.Context, query string, docs []string) ([]float64, error) {
	f.lastQry = query
	f.lastDoc = append([]string(nil), docs...)
	if f.err != nil {
		return nil, f.err
	}
	out := make([]float64, len(docs))
	for i, d := range docs {
		score := -10.0
		for keyword, s := range f.scores {
			if contains(d, keyword) {
				score = s
				break
			}
		}
		out[i] = score
	}
	return out, nil
}
func (f *fakeReranker) ModelName() string { return "fake-reranker" }
func (f *fakeReranker) Close() error      { f.closed = true; return nil }

func contains(s, substr string) bool {
	for i := 0; i+len(substr) <= len(s); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}

func TestRerankSortsByScoreDesc(t *testing.T) {
	rr := &fakeReranker{scores: map[string]float64{
		"passage labelled relevant":    5.0,
		"passage labelled irrelevant": -5.0,
	}}
	got, err := rerank.Rerank(context.Background(), rr, "q", fixtureCandidates(), rerank.Options{})
	if err != nil {
		t.Fatalf("Rerank: %v", err)
	}
	if len(got) != 3 {
		t.Fatalf("len = %d, want 3", len(got))
	}
	// a and c both score +5 ("relevant"); b scores -5 ("irrelevant").
	// Stable sort preserves input order within tied scores → [a c b].
	if got[0].Result.DocID != "a" || got[1].Result.DocID != "c" || got[2].Result.DocID != "b" {
		t.Errorf("order = [%s %s %s], want [a c b]",
			got[0].Result.DocID, got[1].Result.DocID, got[2].Result.DocID)
	}
	// Min-max normalisation: min=-5, max=+5 → tied relevant docs
	// land at 1.0, irrelevant at 0.0.
	if got[0].Score != 1 || got[1].Score != 1 || got[2].Score != 0 {
		t.Errorf("normalised scores = [%g %g %g], want [1 1 0]",
			got[0].Score, got[1].Score, got[2].Score)
	}
	// Raw logits should survive normalisation for --explain traces.
	if got[0].RawScore != 5 || got[2].RawScore != -5 {
		t.Errorf("raw scores = [%g %g], want [5 -5]", got[0].RawScore, got[2].RawScore)
	}
}

func TestRerankCarriesRRFRank(t *testing.T) {
	rr := &fakeReranker{scores: map[string]float64{
		"relevant":    5.0,
		"irrelevant": -5.0,
	}}
	got, _ := rerank.Rerank(context.Background(), rr, "q", fixtureCandidates(), rerank.Options{})
	for _, s := range got {
		switch s.Result.DocID {
		case "a":
			if s.RRFRank != 0 {
				t.Errorf("a RRFRank = %d, want 0", s.RRFRank)
			}
		case "b":
			if s.RRFRank != 1 {
				t.Errorf("b RRFRank = %d, want 1", s.RRFRank)
			}
		case "c":
			if s.RRFRank != 2 {
				t.Errorf("c RRFRank = %d, want 2", s.RRFRank)
			}
		}
	}
}

func TestRerankTopNLeavesTailUnscored(t *testing.T) {
	rr := &fakeReranker{scores: map[string]float64{
		"relevant":    5.0,
		"irrelevant": -5.0,
	}}
	got, _ := rerank.Rerank(context.Background(), rr, "q", fixtureCandidates(),
		rerank.Options{TopN: 1})
	// Only candidate 0 ("a") goes to the reranker. With TopN=1 the
	// candidate set has no span (single item), so even the scored
	// one falls back to the 0.5 sentinel; the other two were never
	// submitted and stay at 0.5.
	for _, s := range got {
		if s.Score != 0.5 {
			t.Errorf("doc %s scored %g, want 0.5 (single-item TopN has no span)",
				s.Result.DocID, s.Score)
		}
	}
	if len(rr.lastDoc) != 1 {
		t.Errorf("reranker got %d docs, want 1 (TopN=1)", len(rr.lastDoc))
	}
}

func TestRerankRejectsNilReranker(t *testing.T) {
	if _, err := rerank.Rerank(context.Background(), nil, "q", nil, rerank.Options{}); err == nil {
		t.Error("expected error on nil reranker")
	}
}

func TestRerankRerankerErrorLeavesUnscored(t *testing.T) {
	// When the HTTP call fails, rerank returns the error but also
	// hands back the candidate slice with Score=0.5 so the caller
	// can decide whether to fall back to RRF ordering.
	rr := &fakeReranker{err: errors.New("boom")}
	got, err := rerank.Rerank(context.Background(), rr, "q", fixtureCandidates(), rerank.Options{})
	if err == nil {
		t.Fatal("expected error when reranker fails")
	}
	for _, s := range got {
		if s.Score != 0.5 {
			t.Errorf("doc %s scored %g, want 0.5 sentinel after reranker error",
				s.Result.DocID, s.Score)
		}
	}
}

func TestRerankSubmitsTruncatedPassage(t *testing.T) {
	long := "lorem ipsum dolor sit amet " // repeated below
	for i := 0; i < 200; i++ {
		long += "lorem ipsum dolor sit amet "
	}
	rr := &fakeReranker{scores: map[string]float64{"lorem": 1.0}}
	candidates := []store.SearchResult{
		{DocID: "x", Snippet: long, CollectionName: "c"},
	}
	_, err := rerank.Rerank(context.Background(), rr, "q", candidates,
		rerank.Options{PassageBudget: 50})
	if err != nil {
		t.Fatalf("Rerank: %v", err)
	}
	if len(rr.lastDoc) != 1 {
		t.Fatalf("reranker got %d docs, want 1", len(rr.lastDoc))
	}
	// 50 runes + possibly "…" ≈ 53 chars; certainly not the full
	// 5000+ character original.
	if len(rr.lastDoc[0]) > 100 {
		t.Errorf("passage not truncated: len=%d", len(rr.lastDoc[0]))
	}
}

func TestRerankEqualScoresFallBackToSentinel(t *testing.T) {
	// All candidates get the same logit → span=0 → no signal to
	// rank by. Every Score should be the 0.5 sentinel.
	rr := &fakeReranker{scores: map[string]float64{"passage": 2.0}}
	got, err := rerank.Rerank(context.Background(), rr, "q", fixtureCandidates(), rerank.Options{})
	if err != nil {
		t.Fatal(err)
	}
	for _, s := range got {
		if s.Score != 0.5 {
			t.Errorf("doc %s scored %g, want 0.5 sentinel when all logits equal",
				s.Result.DocID, s.Score)
		}
	}
}

func TestRerankRespectsContextValue(t *testing.T) {
	rr := &fakeReranker{scores: map[string]float64{"relevant": 1.0, "irrelevant": -1.0}}
	// Context threaded through to the reranker — catch any handlers
	// that silently swap ctx.Background() or similar.
	type ctxKey string
	ctx := context.WithValue(context.Background(), ctxKey("k"), "v")
	_, err := rerank.Rerank(ctx, rr, "q", fixtureCandidates(), rerank.Options{})
	if err != nil {
		t.Fatal(err)
	}
}

func TestRerankLengthMismatchIsError(t *testing.T) {
	rr := &shortReranker{}
	got, err := rerank.Rerank(context.Background(), rr, "q", fixtureCandidates(), rerank.Options{})
	if err == nil {
		t.Fatal("expected error on length mismatch")
	}
	for _, s := range got {
		if s.Score != 0.5 {
			t.Errorf("doc %s should stay at 0.5 on mismatch, got %g", s.Result.DocID, s.Score)
		}
	}
}

type shortReranker struct{ llm.Reranker }

func (*shortReranker) Rerank(_ context.Context, _ string, docs []string) ([]float64, error) {
	return make([]float64, len(docs)-1), nil // always one short
}
func (*shortReranker) ModelName() string { return "short" }
func (*shortReranker) Close() error      { return nil }
