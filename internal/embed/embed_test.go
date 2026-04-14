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
