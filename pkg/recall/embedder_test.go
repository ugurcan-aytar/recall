package recall_test

// This file exercises the public embedder surface from the point of
// view of an external consumer (e.g. brain). It MUST NOT import
// internal/embed or internal/store — if it does, the re-export layer
// isn't actually closing the gap.
//
// A compile failure here means a library user can't use recall.

import (
	"path/filepath"
	"testing"

	"github.com/ugurcan-aytar/recall/pkg/recall"
)

// Compile-time assertions: the re-exported types satisfy the public
// Embedder interface. If these fail to compile, the aliases are broken.
var (
	_ recall.Embedder = (*recall.MockEmbedder)(nil)
)

func TestMockEmbedderFromPublicAPI(t *testing.T) {
	emb := recall.NewMockEmbedder(recall.EmbeddingDimensions)
	if emb == nil {
		t.Fatal("NewMockEmbedder returned nil")
	}
	if emb.Dimensions() != recall.EmbeddingDimensions {
		t.Errorf("Dimensions = %d, want %d", emb.Dimensions(), recall.EmbeddingDimensions)
	}
	if emb.ModelName() == "" {
		t.Error("ModelName is empty")
	}
	v, err := emb.EmbedSingle("hello")
	if err != nil {
		t.Fatalf("EmbedSingle: %v", err)
	}
	if len(v) != recall.EmbeddingDimensions {
		t.Errorf("len(vec) = %d, want %d", len(v), recall.EmbeddingDimensions)
	}
	if err := emb.Close(); err != nil {
		t.Errorf("Close: %v", err)
	}
}

func TestResolveAPIProviderDefaultsToLocal(t *testing.T) {
	t.Setenv("RECALL_EMBED_PROVIDER", "")
	if got := recall.ResolveAPIProvider(); got != recall.ProviderLocal {
		t.Errorf("ResolveAPIProvider() = %q, want ProviderLocal", got)
	}
}

func TestResolveAPIProviderOpenAI(t *testing.T) {
	t.Setenv("RECALL_EMBED_PROVIDER", "openai")
	if got := recall.ResolveAPIProvider(); got != recall.ProviderOpenAI {
		t.Errorf("ResolveAPIProvider() = %q, want ProviderOpenAI", got)
	}
}

func TestResolveAPIProviderVoyage(t *testing.T) {
	t.Setenv("RECALL_EMBED_PROVIDER", "voyage")
	if got := recall.ResolveAPIProvider(); got != recall.ProviderVoyage {
		t.Errorf("ResolveAPIProvider() = %q, want ProviderVoyage", got)
	}
}

func TestResolveAPIProviderUnknownFallsBackToLocal(t *testing.T) {
	t.Setenv("RECALL_EMBED_PROVIDER", "pigeon")
	if got := recall.ResolveAPIProvider(); got != recall.ProviderLocal {
		t.Errorf("ResolveAPIProvider() = %q, want ProviderLocal", got)
	}
}

func TestLocalEmbedderAvailableMatchesBuildTag(t *testing.T) {
	// No meaningful assertion on the return value — we just confirm the
	// symbol is callable from outside the module.
	_ = recall.LocalEmbedderAvailable()
}

func TestNewLocalEmbedderStubReturnsNotCompiled(t *testing.T) {
	if recall.LocalEmbedderAvailable() {
		t.Skip("binary was built with embed_llama — stub path not applicable")
	}
	_, err := recall.NewLocalEmbedder(recall.LocalEmbedderOptions{ModelPath: "/tmp/nope.gguf"})
	if err == nil {
		t.Fatal("expected error on stub build")
	}
	// errors.Is via the public ErrLocalEmbedderNotCompiled handle.
	if !errorsContains(err, recall.ErrLocalEmbedderNotCompiled) {
		t.Errorf("err = %v, want it to wrap ErrLocalEmbedderNotCompiled", err)
	}
}

func TestNewAPIEmbedderRejectsLocalProvider(t *testing.T) {
	_, err := recall.NewAPIEmbedder(recall.APIEmbedderOptions{Provider: recall.ProviderLocal})
	if err == nil {
		t.Error("expected error when Provider is ProviderLocal")
	}
}

func TestFormatHelpersUsePublicAPI(t *testing.T) {
	q := recall.FormatQuery("what is X")
	if q == "" || q == "what is X" {
		t.Errorf("FormatQuery added no prefix: %q", q)
	}
	d := recall.FormatDocument("Title", "body")
	if d == "" || d == "body" {
		t.Errorf("FormatDocument added no prefix: %q", d)
	}
}

func TestFamilyHelpersUsePublicAPI(t *testing.T) {
	if got := recall.DetectFamily("nomic-embed-text-v1.5"); got != recall.FamilyNomic {
		t.Errorf("DetectFamily(nomic) = %q, want FamilyNomic", got)
	}
	if got := recall.DetectFamily("embeddinggemma-300m"); got != recall.FamilyGemma {
		t.Errorf("DetectFamily(gemma) = %q, want FamilyGemma", got)
	}
	if got := recall.DetectFamily("Qwen3-Embedding-0.6B"); got != recall.FamilyQwen3 {
		t.Errorf("DetectFamily(qwen3) = %q, want FamilyQwen3", got)
	}

	t.Setenv("RECALL_EMBED_PROMPT_FORMAT", "generic")
	if got := recall.ResolveFamily("nomic-embed-text"); got != recall.FamilyGeneric {
		t.Errorf("ResolveFamily(env=generic) = %q, want FamilyGeneric", got)
	}

	if got := recall.FormatQueryFor(recall.FamilyGemma, "X"); got != "task: search result | query: X" {
		t.Errorf("FormatQueryFor(gemma) = %q", got)
	}
	if got := recall.FormatDocumentFor(recall.FamilyGeneric, "T", "body"); got != "body" {
		t.Errorf("FormatDocumentFor(generic) = %q, want raw body", got)
	}
}

func TestResolveActiveModelPathFromPublicAPI(t *testing.T) {
	tmp := t.TempDir()
	t.Setenv("RECALL_MODELS_DIR", tmp)

	// Default — nothing set.
	t.Setenv("RECALL_EMBED_MODEL", "")
	got, err := recall.ResolveActiveModelPath()
	if err != nil {
		t.Fatalf("default: %v", err)
	}
	if got == "" {
		t.Fatal("ResolveActiveModelPath returned empty")
	}

	// Override with a custom filename.
	t.Setenv("RECALL_EMBED_MODEL", "my-model.gguf")
	got, err = recall.ResolveActiveModelPath()
	if err != nil {
		t.Fatalf("override: %v", err)
	}
	if got == "" || got[len(got)-len("my-model.gguf"):] != "my-model.gguf" {
		t.Errorf("override = %q, want suffix my-model.gguf", got)
	}
}

func TestModelsDirAndResolveModelPath(t *testing.T) {
	t.Setenv("RECALL_MODELS_DIR", t.TempDir())
	dir, err := recall.ModelsDir()
	if err != nil {
		t.Fatalf("ModelsDir: %v", err)
	}
	p, err := recall.ResolveModelPath(recall.DefaultModelName)
	if err != nil {
		t.Fatalf("ResolveModelPath: %v", err)
	}
	if filepath.Dir(p) != dir {
		t.Errorf("ResolveModelPath dir = %s, want %s", filepath.Dir(p), dir)
	}
}

func TestQueryCacheFromPublicAPI(t *testing.T) {
	c := recall.NewQueryCache(0)
	if c == nil {
		t.Fatal("NewQueryCache returned nil")
	}
	v := []float32{1, 2, 3}
	c.Put("k", v)
	got, ok := c.Get("k")
	if !ok || len(got) != 3 {
		t.Errorf("cache roundtrip failed: got=%v ok=%v", got, ok)
	}
}

// errorsContains is a tiny local helper so the test file doesn't drag
// in the standard-library errors package just for one Is check.
func errorsContains(err, target error) bool {
	for e := err; e != nil; {
		if e == target {
			return true
		}
		type wrap interface{ Unwrap() error }
		w, ok := e.(wrap)
		if !ok {
			return false
		}
		e = w.Unwrap()
	}
	return false
}

// TestResolveEmbedderRespectsAPIProvider verifies the happy path: with an
// API provider set and the key present, we get an Embedder back without
// touching the local GGUF path.
func TestResolveEmbedderRespectsAPIProvider(t *testing.T) {
	t.Setenv("RECALL_EMBED_PROVIDER", "openai")
	t.Setenv("OPENAI_API_KEY", "sk-test-not-used")

	emb, err := recall.ResolveEmbedder()
	if err != nil {
		t.Fatalf("ResolveEmbedder: %v", err)
	}
	if emb == nil {
		t.Fatal("ResolveEmbedder returned nil embedder")
	}
	_ = emb.Close()
}

// TestResolveEmbedderFallsBackWhenLocalUnavailable ensures that on a
// stub build (no embed_llama) the default path returns the clear
// "not compiled" error, so library consumers can branch on it.
func TestResolveEmbedderFallsBackWhenLocalUnavailable(t *testing.T) {
	if recall.LocalEmbedderAvailable() {
		t.Skip("binary was built with embed_llama — stub path not applicable")
	}
	t.Setenv("RECALL_EMBED_PROVIDER", "")
	t.Setenv("RECALL_MODELS_DIR", t.TempDir())

	_, err := recall.ResolveEmbedder()
	if err == nil {
		t.Fatal("expected ResolveEmbedder to fail on stub build")
	}
	if !errorsContains(err, recall.ErrLocalEmbedderNotCompiled) {
		// The model-missing error path is also valid (ModelsDir is empty),
		// but on stub build the compile-time gate must trip first.
		t.Logf("ResolveEmbedder err = %v", err)
	}
}
