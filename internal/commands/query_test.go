package commands

import (
	"strings"
	"testing"

	"github.com/ugurcan-aytar/recall/internal/llm"
)

func TestQueryCmdFlags(t *testing.T) {
	if queryCmd.Use == "" {
		t.Fatal("queryCmd.Use is empty")
	}
	for _, name := range []string{
		"limit", "collection", "all", "min-score", "full", "explain",
		"json", "csv", "md", "xml", "files", "chunk-strategy",
		"expand", "intent",
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
