package embed

import (
	"math"
	"strings"
	"testing"
)

// TestMockEmbedder verifies the mock returns consistent dimensions and
// values, satisfying CLAUDE.md's testing rules.
func TestMockEmbedder(t *testing.T) {
	m := NewMockEmbedder(0)
	if m.Dimensions() != 768 {
		t.Fatalf("default dims = %d, want 768", m.Dimensions())
	}
	if m.ModelName() == "" {
		t.Error("ModelName empty")
	}

	a, err := m.EmbedSingle("hello world")
	if err != nil {
		t.Fatalf("EmbedSingle: %v", err)
	}
	if len(a) != 768 {
		t.Errorf("vector length = %d, want 768", len(a))
	}

	// Determinism.
	b, _ := m.EmbedSingle("hello world")
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("non-deterministic at index %d: %v vs %v", i, a[i], b[i])
		}
	}

	// Different input → different vector.
	c, _ := m.EmbedSingle("something else")
	identical := true
	for i := range a {
		if a[i] != c[i] {
			identical = false
			break
		}
	}
	if identical {
		t.Error("different inputs produced identical vectors")
	}

	// Vectors should be roughly unit length (we L2-normalise).
	var norm float64
	for _, v := range a {
		norm += float64(v) * float64(v)
	}
	if math.Abs(math.Sqrt(norm)-1) > 1e-3 {
		t.Errorf("vector not unit-normalised: |v| = %.6f", math.Sqrt(norm))
	}
}

// TestEmbedderInterface confirms MockEmbedder satisfies the contract; the
// real local backend's conformance is checked by its own _test.go inside
// the embed_llama build tag.
func TestEmbedderInterface(t *testing.T) {
	var _ Embedder = (*MockEmbedder)(nil)

	m := NewMockEmbedder(64)
	vecs, err := m.Embed([]string{"a", "b", "c"})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(vecs) != 3 {
		t.Fatalf("len(vecs) = %d", len(vecs))
	}
	for i, v := range vecs {
		if len(v) != 64 {
			t.Errorf("vec %d width = %d, want 64", i, len(v))
		}
	}

	if err := m.Close(); err != nil {
		t.Errorf("Close: %v", err)
	}
	if _, err := m.EmbedSingle("after close"); err == nil {
		t.Error("expected error after Close")
	}
}

func TestFormatQueryAndDocument(t *testing.T) {
	// nomic-embed-text-v1.5 task prefixes — see model card on HuggingFace.
	q := FormatQuery("how do I deploy")
	if !strings.HasPrefix(q, "search_query: ") {
		t.Errorf("FormatQuery: %q (expected nomic 'search_query: ' prefix)", q)
	}
	if !strings.Contains(q, "how do I deploy") {
		t.Errorf("FormatQuery missing query body: %q", q)
	}

	d := FormatDocument("Deployment", "step 1: ...")
	if !strings.HasPrefix(d, "search_document: ") {
		t.Errorf("FormatDocument: %q (expected nomic 'search_document: ' prefix)", d)
	}
	if !strings.Contains(d, "Deployment") {
		t.Errorf("FormatDocument missing title: %q", d)
	}
	if !strings.Contains(d, "step 1: ...") {
		t.Errorf("FormatDocument missing content: %q", d)
	}

	// title-less variant uses just the prefix
	dNoTitle := FormatDocument("", "body only")
	if dNoTitle != "search_document: body only" {
		t.Errorf("title-less FormatDocument: %q", dNoTitle)
	}
}

func TestDetectFamily(t *testing.T) {
	cases := []struct {
		in   string
		want PromptFamily
	}{
		{"nomic-embed-text-v1.5.Q8_0", FamilyNomic},
		{"NOMIC-Embed-Text-v2", FamilyNomic},
		{"nomic_embed_text_local", FamilyNomic},
		{"embeddinggemma-300M-Q8_0", FamilyGemma},
		{"embedding-gemma-300m", FamilyGemma},
		{"google_embedding_gemma_300m", FamilyGemma},
		{"Qwen3-Embedding-0.6B-Q8_0", FamilyQwen3},
		{"qwen3_embedding_0.6b", FamilyQwen3},
		{"Qwen3Embedding0.6B", FamilyQwen3},
		{"unknown-model.Q4", FamilyNomic}, // safe fallback
		{"", FamilyNomic},
	}
	for _, tc := range cases {
		t.Run(tc.in, func(t *testing.T) {
			if got := DetectFamily(tc.in); got != tc.want {
				t.Errorf("DetectFamily(%q) = %q, want %q", tc.in, got, tc.want)
			}
		})
	}
}

func TestResolveFamilyEnvOverride(t *testing.T) {
	cases := []struct {
		envVal, modelName string
		want              PromptFamily
	}{
		{"nomic", "Qwen3-Embedding-0.6B", FamilyNomic},
		{"gemma", "nomic-embed-text", FamilyGemma},
		{"embeddinggemma", "nomic-embed-text", FamilyGemma},
		{"qwen", "nomic-embed-text", FamilyQwen3},
		{"qwen3", "nomic-embed-text", FamilyQwen3},
		{"generic", "nomic-embed-text", FamilyGeneric},
		{"raw", "nomic-embed-text", FamilyGeneric},
		{"none", "nomic-embed-text", FamilyGeneric},
		{"  Gemma  ", "nomic-embed-text", FamilyGemma}, // case + trim insensitive
		{"", "nomic-embed-text", FamilyNomic},          // env unset → detection
		{"unknown-format", "nomic-embed-text", FamilyNomic}, // unknown env → detection
	}
	for _, tc := range cases {
		t.Run(tc.envVal+"/"+tc.modelName, func(t *testing.T) {
			t.Setenv("RECALL_EMBED_PROMPT_FORMAT", tc.envVal)
			if got := ResolveFamily(tc.modelName); got != tc.want {
				t.Errorf("ResolveFamily(env=%q, model=%q) = %q, want %q",
					tc.envVal, tc.modelName, got, tc.want)
			}
		})
	}
}

func TestFormatQueryForPerFamily(t *testing.T) {
	cases := []struct {
		family PromptFamily
		want   string
	}{
		{FamilyNomic, "search_query: how does X work"},
		{FamilyGemma, "task: search result | query: how does X work"},
		{FamilyQwen3, "Instruct: Given a query, retrieve relevant passages that answer the query\nQuery: how does X work"},
		{FamilyGeneric, "how does X work"},
	}
	for _, tc := range cases {
		t.Run(string(tc.family), func(t *testing.T) {
			if got := FormatQueryFor(tc.family, "how does X work"); got != tc.want {
				t.Errorf("FormatQueryFor(%q) = %q, want %q", tc.family, got, tc.want)
			}
		})
	}
}

func TestFormatDocumentForPerFamily(t *testing.T) {
	cases := []struct {
		family        PromptFamily
		title, body   string
		want          string
	}{
		{FamilyNomic, "Auth", "JWT details", "search_document: Auth — JWT details"},
		{FamilyNomic, "", "JWT details", "search_document: JWT details"},
		{FamilyGemma, "Auth", "JWT details", "title: Auth | text: JWT details"},
		{FamilyGemma, "", "JWT details", "title: none | text: JWT details"},
		{FamilyQwen3, "Auth", "JWT details", "JWT details"}, // qwen3 docs are raw
		{FamilyQwen3, "", "JWT details", "JWT details"},
		{FamilyGeneric, "Auth", "JWT details", "JWT details"},
	}
	for _, tc := range cases {
		t.Run(string(tc.family)+"/"+tc.title, func(t *testing.T) {
			if got := FormatDocumentFor(tc.family, tc.title, tc.body); got != tc.want {
				t.Errorf("FormatDocumentFor(%q, %q, %q) = %q, want %q",
					tc.family, tc.title, tc.body, got, tc.want)
			}
		})
	}
}

func TestResolveActiveModelPathHonoursEnv(t *testing.T) {
	tmp := t.TempDir()
	t.Setenv("RECALL_MODELS_DIR", tmp)

	// Default — no override.
	t.Setenv("RECALL_EMBED_MODEL", "")
	got, err := ResolveActiveModelPath()
	if err != nil {
		t.Fatalf("default: %v", err)
	}
	if !strings.HasSuffix(got, DefaultModelName) {
		t.Errorf("default = %q, want suffix %q", got, DefaultModelName)
	}

	// Bare filename — joined with ModelsDir.
	t.Setenv("RECALL_EMBED_MODEL", "my-embed.gguf")
	got, err = ResolveActiveModelPath()
	if err != nil {
		t.Fatalf("filename: %v", err)
	}
	if !strings.HasSuffix(got, "my-embed.gguf") {
		t.Errorf("filename = %q, want suffix my-embed.gguf", got)
	}
	if !strings.Contains(got, tmp) {
		t.Errorf("filename = %q, want to contain models dir %q", got, tmp)
	}

	// Absolute path — returned as-is.
	abs := "/opt/models/explicit-path.gguf"
	t.Setenv("RECALL_EMBED_MODEL", abs)
	got, err = ResolveActiveModelPath()
	if err != nil {
		t.Fatalf("absolute: %v", err)
	}
	if got != abs {
		t.Errorf("absolute = %q, want %q", got, abs)
	}
}
