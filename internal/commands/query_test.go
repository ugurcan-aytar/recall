package commands

import (
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
		"expand", "intent", "rerank", "rerank-top-n",
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

	got, err := runExpansion("rate limiter")
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
	got, err := runExpansion("anything")
	if err != nil {
		t.Errorf("expected nil error on stub build, got %v", err)
	}
	if got != nil {
		t.Errorf("expected nil expansion on stub build, got %+v", got)
	}
}

// TestApplyRerankReordersByScore verifies the rerank wire-up:
// inject a keyword-based MockGenerator that returns "yes" iff the
// passage contains "match" and "no" otherwise, hand applyRerank a
// 3-doc fused list, and confirm the matching docs land on top with
// FusedScore = 1.0.
func TestApplyRerankReordersByScore(t *testing.T) {
	gen := &keywordYesNoGen{}
	SetRerankGeneratorOverride(gen)
	t.Cleanup(func() { SetRerankGeneratorOverride(nil) })

	in := []store.FusedResult{
		{SearchResult: store.SearchResult{DocID: "x", Path: "x.md", Snippet: "no signal here", CollectionName: "c"}, FusedScore: 0.8},
		{SearchResult: store.SearchResult{DocID: "y", Path: "y.md", Snippet: "this is the match passage", CollectionName: "c"}, FusedScore: 0.6},
		{SearchResult: store.SearchResult{DocID: "z", Path: "z.md", Snippet: "another match here", CollectionName: "c"}, FusedScore: 0.4},
	}
	out := applyRerank(in, "q")
	if len(out) != 3 {
		t.Fatalf("len = %d, want 3", len(out))
	}
	if out[0].DocID != "y" && out[0].DocID != "z" {
		t.Errorf("first = %s, want y or z (the matching docs)", out[0].DocID)
	}
	if out[2].DocID != "x" {
		t.Errorf("last = %s, want x (no match)", out[2].DocID)
	}
	// Matching docs get FusedScore overwritten to 1.0; non-match to 0.0.
	for _, r := range out {
		if r.DocID == "x" && r.FusedScore != 0 {
			t.Errorf("x FusedScore = %g, want 0", r.FusedScore)
		}
		if (r.DocID == "y" || r.DocID == "z") && r.FusedScore != 1 {
			t.Errorf("%s FusedScore = %g, want 1", r.DocID, r.FusedScore)
		}
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
