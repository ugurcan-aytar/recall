package recall_test

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/ugurcan-aytar/recall/internal/embed"
	"github.com/ugurcan-aytar/recall/internal/store"
	"github.com/ugurcan-aytar/recall/pkg/recall"
)

func tempCollection(t *testing.T, files map[string]string) string {
	t.Helper()
	dir := t.TempDir()
	for name, content := range files {
		p := filepath.Join(dir, name)
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatalf("mkdir: %v", err)
		}
		if err := os.WriteFile(p, []byte(content), 0o644); err != nil {
			t.Fatalf("write: %v", err)
		}
	}
	return dir
}

func newEngine(t *testing.T) *recall.Engine {
	t.Helper()
	dir := t.TempDir()
	eng, err := recall.NewEngine(recall.WithDBPath(filepath.Join(dir, "idx.db")))
	if err != nil {
		t.Fatalf("NewEngine: %v", err)
	}
	t.Cleanup(func() { _ = eng.Close() })
	return eng
}

func TestEndToEndHybridSearch(t *testing.T) {
	eng := newEngine(t)

	dir := tempCollection(t, map[string]string{
		"auth.md":    "# Auth\nThe authentication flow handles JWT tokens.",
		"rate.md":    "# Rate Limiter\nDiscussion of rate limiting algorithms.",
		"weather.md": "# Misc\nUnrelated content about clouds and weather.",
	})
	if _, err := eng.AddCollection("notes", dir, "", "team notes"); err != nil {
		t.Fatal(err)
	}
	if _, err := eng.Index(); err != nil {
		t.Fatal(err)
	}

	emb := embed.NewMockEmbedder(store.EmbeddingDimensions)
	if _, err := eng.Embed(emb, false); err != nil {
		t.Fatalf("Embed: %v", err)
	}

	// Query close to "auth" content. The mock embedder is deterministic, so
	// embedding "authentication" for the query and the doc-formatted text
	// for the chunks won't yield identical vectors — but we mainly want to
	// exercise the full pipeline (BM25 + vector + RRF) without errors.
	results, err := eng.SearchHybrid(emb, "authentication", recall.WithLimit(5))
	if err != nil {
		t.Fatalf("SearchHybrid: %v", err)
	}
	if len(results) == 0 {
		t.Fatal("hybrid returned no results")
	}
	// At least one hit must reference the auth doc (BM25 will surface it).
	found := false
	for _, r := range results {
		if r.Path == "auth.md" {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("auth.md not in hybrid results: %+v", results)
	}
}

func TestEndToEndBM25Only(t *testing.T) {
	eng := newEngine(t)

	dir := tempCollection(t, map[string]string{
		"a.md": "# A\napples and oranges",
		"b.md": "# B\nzebras and lions",
	})
	if _, err := eng.AddCollection("zoo", dir, "", ""); err != nil {
		t.Fatal(err)
	}
	if _, err := eng.Index(); err != nil {
		t.Fatal(err)
	}

	// No Embed call → SearchHybrid must degrade to BM25 silently.
	results, err := eng.SearchHybrid(nil, "zebras", recall.WithLimit(5))
	if err != nil {
		t.Fatalf("SearchHybrid (BM25-only): %v", err)
	}
	if len(results) == 0 {
		t.Fatal("expected BM25 fallback hits")
	}
	if results[0].Path != "b.md" {
		t.Errorf("top = %s, want b.md", results[0].Path)
	}
}

func TestPublicAPILifecycle(t *testing.T) {
	eng := newEngine(t)

	dir := tempCollection(t, map[string]string{"x.md": "# X\nhello world"})
	c, err := eng.AddCollection("k", dir, "", "")
	if err != nil {
		t.Fatal(err)
	}
	if c.Name != "k" {
		t.Errorf("name = %q", c.Name)
	}

	cols, err := eng.ListCollections()
	if err != nil {
		t.Fatal(err)
	}
	if len(cols) != 1 {
		t.Fatalf("ListCollections = %d", len(cols))
	}

	idx, err := eng.Index()
	if err != nil {
		t.Fatal(err)
	}
	if idx.PerCollection["k"].Indexed != 1 {
		t.Errorf("indexed = %+v", idx.PerCollection["k"])
	}

	doc, err := eng.Get("k/x.md")
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if doc.Title != "X" {
		t.Errorf("title = %q", doc.Title)
	}

	emb := embed.NewMockEmbedder(store.EmbeddingDimensions)
	if _, err := eng.Embed(emb, false); err != nil {
		t.Fatal(err)
	}

	bm := eng.Store()
	if n, _ := bm.EmbeddingCount(); n == 0 {
		t.Error("no embeddings after Embed()")
	}

	if err := eng.AddContext("k", "/", "test context"); err != nil {
		t.Fatal(err)
	}
	ctxs, _ := eng.ListContexts()
	if len(ctxs) != 1 {
		t.Errorf("contexts = %+v", ctxs)
	}

	if err := eng.RemoveCollection("k"); err != nil {
		t.Fatal(err)
	}
}

func TestEngineEmbedRejectsMismatchedDims(t *testing.T) {
	eng := newEngine(t)

	dir := tempCollection(t, map[string]string{"a.md": "x"})
	if _, err := eng.AddCollection("c", dir, "", ""); err != nil {
		t.Fatal(err)
	}
	if _, err := eng.Index(); err != nil {
		t.Fatal(err)
	}

	bad := embed.NewMockEmbedder(64) // wrong width vs vec0(768)
	if _, err := eng.Embed(bad, false); err == nil {
		t.Error("expected dim-mismatch error")
	}
}
