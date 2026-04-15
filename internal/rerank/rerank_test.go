package rerank_test

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/ugurcan-aytar/recall/internal/llm"
	"github.com/ugurcan-aytar/recall/internal/rerank"
	"github.com/ugurcan-aytar/recall/internal/store"
)

// fixtureCandidates returns three SearchResults whose snippets each
// trigger a specific verdict from the keyword-yes/no MockGenerator
// rule below.
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

// keywordGen returns a generator whose response depends on whether
// the prompt contains "relevant" (yes) or "irrelevant" (no). Lets us
// rerank a fixture without mocking the exact prompt-to-response map.
type keywordGen struct {
	llm.MockGenerator
}

func (k *keywordGen) Generate(prompt string, _ ...llm.GenerateOption) (string, error) {
	low := strings.ToLower(prompt)
	if strings.Contains(low, "irrelevant") {
		return "no", nil
	}
	if strings.Contains(low, "relevant") {
		return "yes", nil
	}
	return "no", nil
}

func TestRerankSortsByScoreDesc(t *testing.T) {
	gen := &keywordGen{}
	got, err := rerank.Rerank(context.Background(), gen, "q", fixtureCandidates(), rerank.Options{})
	if err != nil {
		t.Fatalf("Rerank: %v", err)
	}
	if len(got) != 3 {
		t.Fatalf("len = %d, want 3", len(got))
	}
	// a and c are "relevant" → 1.0; b is "irrelevant" → 0.0. The
	// stable sort should preserve a then c (input order).
	if got[0].Result.DocID != "a" || got[1].Result.DocID != "c" || got[2].Result.DocID != "b" {
		t.Errorf("order = [%s %s %s], want [a c b]",
			got[0].Result.DocID, got[1].Result.DocID, got[2].Result.DocID)
	}
	if got[0].Score != 1 || got[1].Score != 1 || got[2].Score != 0 {
		t.Errorf("scores = [%g %g %g], want [1 1 0]",
			got[0].Score, got[1].Score, got[2].Score)
	}
}

func TestRerankCarriesRRFRank(t *testing.T) {
	gen := &keywordGen{}
	got, _ := rerank.Rerank(context.Background(), gen, "q", fixtureCandidates(), rerank.Options{})
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
	gen := &keywordGen{}
	got, _ := rerank.Rerank(context.Background(), gen, "q", fixtureCandidates(),
		rerank.Options{TopN: 1})
	// Only candidate 0 ("a"/relevant) gets the LLM, so it scores
	// 1.0. The other two stay at the 0.5 "unscored" sentinel.
	var scoredA, unscoredB, unscoredC bool
	for _, s := range got {
		switch s.Result.DocID {
		case "a":
			scoredA = s.Score == 1.0
		case "b":
			unscoredB = s.Score == 0.5
		case "c":
			unscoredC = s.Score == 0.5
		}
	}
	if !scoredA {
		t.Error("a should have been scored 1.0")
	}
	if !unscoredB || !unscoredC {
		t.Errorf("b/c should have stayed at 0.5 unscored sentinel; got = %+v", got)
	}
}

func TestRerankRejectsNilGenerator(t *testing.T) {
	if _, err := rerank.Rerank(context.Background(), nil, "q", nil, rerank.Options{}); err == nil {
		t.Error("expected error on nil generator")
	}
}

func TestRerankRespectsContextCancellation(t *testing.T) {
	gen := &keywordGen{}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := rerank.Rerank(ctx, gen, "q", fixtureCandidates(), rerank.Options{})
	if !errors.Is(err, context.Canceled) {
		t.Errorf("err = %v, want context.Canceled", err)
	}
}

func TestRerankPerCallErrorLeavesUnscored(t *testing.T) {
	// erroringGen always errors — every candidate must be left at
	// the 0.5 sentinel and the function returns nil error (rerank
	// degrades gracefully on per-call failures so a flaky model
	// doesn't kill the whole query).
	gen := &erroringGen{err: errors.New("boom")}
	got, err := rerank.Rerank(context.Background(), gen, "q", fixtureCandidates(), rerank.Options{})
	if err != nil {
		t.Fatalf("expected nil error, got %v", err)
	}
	for _, s := range got {
		if s.Score != 0.5 {
			t.Errorf("doc %s scored %g, want 0.5 sentinel after generator error",
				s.Result.DocID, s.Score)
		}
	}
}

type erroringGen struct{ err error }

func (e *erroringGen) Generate(string, ...llm.GenerateOption) (string, error) {
	return "", e.err
}
func (e *erroringGen) ModelName() string { return "erroring" }
func (e *erroringGen) Close() error      { return nil }

// Public-API regression test for parseYesNo via the keyword-gen
// path. Locked down to whole-word matching so "no" doesn't latch
// onto "another" or "noticed".
func TestRerankWholeWordYesNo(t *testing.T) {
	cases := []struct {
		name    string
		response string
		want    float64
	}{
		{"plain yes", "yes", 1.0},
		{"plain no", "no", 0.0},
		{"yes with capital", "Yes", 1.0},
		{"yes inside chatty echo", "The answer is yes, this is relevant", 1.0},
		{"no inside chatty echo", "the answer is no, this passage is unrelated", 0.0},
		{"no should not match noticed", "I noticed nothing relevant", 0.5},
		{"no should not match another", "another reason", 0.5},
		{"yes wins on order", "yes and no but yes overall", 1.0},
		{"no wins on order", "no, although someone might say yes", 0.0},
		{"empty falls back to 0.5", "", 0.5},
		{"unrelated text falls back to 0.5", "I'm not sure how to answer", 0.5},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			gen := llm.NewMockGenerator(nil)
			gen.Default = tc.response
			got, err := rerank.Rerank(context.Background(), gen, "q", []store.SearchResult{
				{DocID: "x", Snippet: "anything"},
			}, rerank.Options{})
			if err != nil {
				t.Fatal(err)
			}
			if got[0].Score != tc.want {
				t.Errorf("response %q → %g, want %g", tc.response, got[0].Score, tc.want)
			}
		})
	}
}
