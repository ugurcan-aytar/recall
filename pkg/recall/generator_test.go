package recall_test

// External-consumer smoke for the Generator + Expand surface. If any
// alias / factory below stops compiling, brain's expansion path
// breaks — that's the signal we're guarding.

import (
	"testing"

	"github.com/ugurcan-aytar/recall/pkg/recall"
)

// Compile-time conformance: the re-exported MockGenerator satisfies
// the public Generator interface.
var _ recall.Generator = (*recall.MockGenerator)(nil)

func TestMockGeneratorFromPublicAPI(t *testing.T) {
	gen := recall.NewMockGenerator(map[string]string{
		"hello": "world",
	})
	gen.Default = "fallback"

	got, err := gen.Generate("hello", recall.WithMaxTokens(64))
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if got != "world" {
		t.Errorf("mapped Generate = %q, want world", got)
	}

	got, err = gen.Generate("anything-else")
	if err != nil {
		t.Fatalf("Generate(default): %v", err)
	}
	if got != "fallback" {
		t.Errorf("default Generate = %q, want fallback", got)
	}
}

func TestExpandFromPublicAPI(t *testing.T) {
	gen := recall.NewMockGenerator(map[string]string{
		"/no_think Expand this search query: rate limiter": "lex: rate limiter\nvec: how rate limiters cap traffic\nhyde: A rate limiter caps requests.",
	})
	got, err := recall.Expand(gen, "rate limiter", recall.ExpandOptions{IncludeLex: true})
	if err != nil {
		t.Fatalf("Expand: %v", err)
	}
	if len(got.Lex) != 1 || len(got.Vec) != 1 || len(got.Hyde) != 1 {
		t.Errorf("Expand buckets = %+v", got)
	}
}

func TestLocalGeneratorAvailableMatchesBuildTag(t *testing.T) {
	// We don't assert on the value — just confirm the public symbol
	// is callable from outside the module.
	_ = recall.LocalGeneratorAvailable()
}

func TestNewLocalGeneratorStubReturnsNotCompiled(t *testing.T) {
	if recall.LocalGeneratorAvailable() {
		t.Skip("binary built with embed_llama")
	}
	_, err := recall.NewLocalGenerator(recall.LocalGeneratorOptions{ModelPath: "/tmp/nope.gguf"})
	if err == nil {
		t.Fatal("expected error on stub build")
	}
}

func TestResolveActiveExpansionModelPathFromPublicAPI(t *testing.T) {
	tmp := t.TempDir()
	t.Setenv("RECALL_MODELS_DIR", tmp)

	t.Setenv("RECALL_EXPAND_MODEL", "")
	got, err := recall.ResolveActiveExpansionModelPath()
	if err != nil {
		t.Fatalf("default: %v", err)
	}
	if got == "" {
		t.Fatal("ResolveActiveExpansionModelPath returned empty")
	}

	t.Setenv("RECALL_EXPAND_MODEL", "my-llm.gguf")
	got, err = recall.ResolveActiveExpansionModelPath()
	if err != nil {
		t.Fatalf("override: %v", err)
	}
	if got == "" || got[len(got)-len("my-llm.gguf"):] != "my-llm.gguf" {
		t.Errorf("override = %q, want suffix my-llm.gguf", got)
	}
}
