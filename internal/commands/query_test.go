package commands

import (
	"path/filepath"
	"strings"
	"testing"

	"github.com/ugurcan-aytar/recall/internal/llm"
	"github.com/ugurcan-aytar/recall/internal/store"
)

func TestQueryCmdFlags(t *testing.T) {
	if queryCmd.Use == "" {
		t.Fatal("queryCmd.Use is empty")
	}
	for _, name := range []string{
		"limit", "collection", "all", "min-score", "full", "explain",
		"json", "csv", "md", "xml", "files", "chunk-strategy",
		"expand", "hyde", "intent", "rerank", "rerank-top-n",
	} {
		if f := queryCmd.Flags().Lookup(name); f == nil {
			t.Errorf("query missing --%s", name)
		}
	}
}

// TestRunExpansionDispatchesGenerator verifies the wiring end-to-end:
// inject a MockGenerator via SetGeneratorOverride, call runExpansion,
// confirm the expected prompt landed and the parsed Expanded came
// back with the model's lex / vec / hyde buckets populated.
func TestRunExpansionDispatchesGenerator(t *testing.T) {
	gen := llm.NewMockGenerator(map[string]string{
		"/no_think Expand this search query: rate limiter": strings.Join([]string{
			"lex: rate limiter",
			"vec: how do rate limiters cap traffic",
			"hyde: A rate limiter using token bucket caps requests per second.",
		}, "\n"),
	})
	SetGeneratorOverride(gen)
	t.Cleanup(func() { SetGeneratorOverride(nil) })

	got, err := runExpansion(nil, "rate limiter")
	if err != nil {
		t.Fatalf("runExpansion: %v", err)
	}
	if got == nil {
		t.Fatal("runExpansion returned nil expansion")
	}
	if len(got.Lex) != 1 {
		t.Errorf("Lex = %+v", got.Lex)
	}
	if len(got.Vec) != 1 {
		t.Errorf("Vec = %+v", got.Vec)
	}
	if len(got.Hyde) != 1 {
		t.Errorf("Hyde = %+v", got.Hyde)
	}
	if calls := gen.Calls(); len(calls) != 1 {
		t.Errorf("expected one Generate call, got %d", len(calls))
	}
}

// TestRunExpansionWithoutModelDegrades verifies the graceful path:
// without an override and without the embed_llama tag, runExpansion
// returns (nil, nil) so the rest of `recall query` continues with
// the original query.
func TestRunExpansionWithoutModelDegrades(t *testing.T) {
	if llm.LocalGeneratorAvailable() {
		t.Skip("binary built with embed_llama — degradation path not reachable")
	}
	SetGeneratorOverride(nil)
	got, err := runExpansion(nil, "anything")
	if err != nil {
		t.Errorf("expected nil error on stub build, got %v", err)
	}
	if got != nil {
		t.Errorf("expected nil expansion on stub build, got %+v", got)
	}
}

// TestApplyRerankReordersByBlendedScore verifies the
// rerank → position-aware blend wire-up. With six candidates and
// a keyword-based MockGenerator that returns "yes" for the matching
// half, the blend's "RRF protects top + reranker corrects deep"
// shape collapses to: yes-and-top-RRF wins, no-and-tail-RRF loses.
func TestApplyRerankReordersByBlendedScore(t *testing.T) {
	gen := &keywordYesNoGen{}
	SetRerankGeneratorOverride(gen)
	t.Cleanup(func() { SetRerankGeneratorOverride(nil) })

	// Top 3 RRF positions are "match" (yes), bottom 3 are not (no).
	in := []store.FusedResult{
		{SearchResult: store.SearchResult{DocID: "y0", Snippet: "match passage A", CollectionName: "c"}, FusedScore: 0.9},
		{SearchResult: store.SearchResult{DocID: "y1", Snippet: "another match here", CollectionName: "c"}, FusedScore: 0.8},
		{SearchResult: store.SearchResult{DocID: "y2", Snippet: "match passage C", CollectionName: "c"}, FusedScore: 0.7},
		{SearchResult: store.SearchResult{DocID: "n3", Snippet: "no signal here", CollectionName: "c"}, FusedScore: 0.6},
		{SearchResult: store.SearchResult{DocID: "n4", Snippet: "irrelevant text", CollectionName: "c"}, FusedScore: 0.5},
		{SearchResult: store.SearchResult{DocID: "n5", Snippet: "definitely not", CollectionName: "c"}, FusedScore: 0.4},
	}
	out := applyRerank(in, "q")
	if len(out) != 6 {
		t.Fatalf("len = %d, want 6", len(out))
	}
	// The three "yes" docs all carry the top-3 RRF rank AND the
	// top reranker verdict, so they should claim the top 3 blended
	// positions. Order within can shuffle by tie.
	yesSet := map[string]struct{}{"y0": {}, "y1": {}, "y2": {}}
	for i := 0; i < 3; i++ {
		if _, ok := yesSet[out[i].DocID]; !ok {
			t.Errorf("blended position %d = %s, want one of y0/y1/y2", i, out[i].DocID)
		}
	}
	// And the three "no" docs should occupy the bottom three.
	noSet := map[string]struct{}{"n3": {}, "n4": {}, "n5": {}}
	for i := 3; i < 6; i++ {
		if _, ok := noSet[out[i].DocID]; !ok {
			t.Errorf("blended position %d = %s, want one of n3/n4/n5", i, out[i].DocID)
		}
	}
	// FusedScore must now hold the BLENDED value, not the raw
	// rerank verdict. Top-of-list y0 (RRFRank 0 + yes) should
	// score 1.0; bottom-of-list n5 (RRFRank 5 + no) should be near 0.
	var topY0, bottomN5 float64
	for _, r := range out {
		if r.DocID == "y0" {
			topY0 = r.FusedScore
		}
		if r.DocID == "n5" {
			bottomN5 = r.FusedScore
		}
	}
	if topY0 != 1.0 {
		t.Errorf("y0 blended = %g, want 1.0", topY0)
	}
	if bottomN5 != 0 {
		t.Errorf("n5 blended = %g, want 0 (RRFRank 5 + no on a 6-doc set)", bottomN5)
	}
}

// TestApplyRerankWithoutModelLeavesInputAlone verifies graceful
// degradation: no model, no override → applyRerank returns the
// input unchanged so the user still gets RRF-ordered results.
func TestApplyRerankWithoutModelLeavesInputAlone(t *testing.T) {
	if llm.LocalGeneratorAvailable() {
		t.Skip("binary built with embed_llama — degradation path not reachable")
	}
	SetRerankGeneratorOverride(nil)
	in := []store.FusedResult{
		{SearchResult: store.SearchResult{DocID: "a"}, FusedScore: 0.9},
		{SearchResult: store.SearchResult{DocID: "b"}, FusedScore: 0.5},
	}
	out := applyRerank(in, "q")
	if len(out) != 2 {
		t.Fatalf("len = %d, want 2", len(out))
	}
	if out[0].DocID != "a" || out[0].FusedScore != 0.9 {
		t.Errorf("first = %+v, want unchanged a@0.9", out[0])
	}
}

type keywordYesNoGen struct{ llm.MockGenerator }

func (k *keywordYesNoGen) Generate(prompt string, _ ...llm.GenerateOption) (string, error) {
	if strings.Contains(strings.ToLower(prompt), "match") {
		return "yes", nil
	}
	return "no", nil
}

// recordingGen captures every prompt it sees so collection-context
// auto-intent can be asserted on directly.
type recordingGen struct {
	llm.MockGenerator
	last string
}

func (r *recordingGen) Generate(prompt string, _ ...llm.GenerateOption) (string, error) {
	r.last = prompt
	// Return a canned expansion that has hyde lines so HyDE wiring
	// has something to embed downstream.
	return "lex: rate limiter\nvec: how does the rate limiter handle bursts\nhyde: A rate limiter caps requests per second using token bucket.", nil
}

// TestRunExpansionAutoFillsIntentFromCollectionContext exercises
// the qmd-fix: when --intent is unset and a single -c collection
// has a Context blurb, runExpansion threads it as the intent line
// so HyDE / expansion stays domain-aware.
func TestRunExpansionAutoFillsIntentFromCollectionContext(t *testing.T) {
	rec := &recordingGen{}
	SetGeneratorOverride(rec)
	t.Cleanup(func() { SetGeneratorOverride(nil) })

	dir := t.TempDir()
	s, err := store.Open(filepath.Join(dir, "i.db"))
	if err != nil {
		t.Fatalf("store.Open: %v", err)
	}
	defer s.Close()
	colDir := t.TempDir()
	if _, err := s.AddCollection("notes", colDir, "", "API rate limiting and resilience patterns"); err != nil {
		t.Fatalf("AddCollection: %v", err)
	}

	// Force the query to target our single collection so
	// auto-intent kicks in. queryIntent stays empty.
	prevCol, prevIntent := queryOpts.Collection, queryIntent
	queryOpts.Collection = "notes"
	queryIntent = ""
	t.Cleanup(func() { queryOpts.Collection, queryIntent = prevCol, prevIntent })

	if _, err := runExpansion(s, "rate limiter"); err != nil {
		t.Fatalf("runExpansion: %v", err)
	}
	if !strings.Contains(rec.last, "Query intent: API rate limiting and resilience patterns") {
		t.Errorf("expected auto-filled intent from collection context; prompt was:\n%s", rec.last)
	}
}

// TestRunExpansionUserIntentBeatsCollectionContext confirms that
// an explicit --intent flag wins over the auto-fill — users keep
// the override knob.
func TestRunExpansionUserIntentBeatsCollectionContext(t *testing.T) {
	rec := &recordingGen{}
	SetGeneratorOverride(rec)
	t.Cleanup(func() { SetGeneratorOverride(nil) })

	dir := t.TempDir()
	s, err := store.Open(filepath.Join(dir, "i.db"))
	if err != nil {
		t.Fatalf("store.Open: %v", err)
	}
	defer s.Close()
	if _, err := s.AddCollection("notes", t.TempDir(), "", "auto-intent collection context"); err != nil {
		t.Fatal(err)
	}

	prevCol, prevIntent := queryOpts.Collection, queryIntent
	queryOpts.Collection = "notes"
	queryIntent = "explicit user intent"
	t.Cleanup(func() { queryOpts.Collection, queryIntent = prevCol, prevIntent })

	if _, err := runExpansion(s, "anything"); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(rec.last, "Query intent: explicit user intent") {
		t.Errorf("expected user --intent to win; prompt:\n%s", rec.last)
	}
	if strings.Contains(rec.last, "auto-intent collection context") {
		t.Errorf("collection context should NOT leak when --intent is set; prompt:\n%s", rec.last)
	}
}
