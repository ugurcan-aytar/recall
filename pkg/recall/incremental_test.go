package recall_test

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/ugurcan-aytar/recall/internal/embed"
	"github.com/ugurcan-aytar/recall/internal/store"
	"github.com/ugurcan-aytar/recall/pkg/recall"
)

// TestIncrementalEmbedSkipsUnchangedChunks pins the V7 expectation:
// after a small edit to one document, re-embedding should only process
// chunks whose content actually changed — not the entire corpus.
func TestIncrementalEmbedSkipsUnchangedChunks(t *testing.T) {
	tmp := t.TempDir()

	// 5 standalone short notes — each becomes one chunk.
	notesDir := filepath.Join(tmp, "notes")
	if err := os.MkdirAll(notesDir, 0o755); err != nil {
		t.Fatal(err)
	}
	for i, body := range []string{
		"# Auth\nFirst note about auth.",
		"# Rate\nNotes on rate limiting.",
		"# Retry\nThoughts on retries.",
		"# Breaker\nCircuit breaker stuff.",
		"# Misc\nUnrelated weather notes.",
	} {
		name := []string{"a", "b", "c", "d", "e"}[i] + ".md"
		if err := os.WriteFile(filepath.Join(notesDir, name), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}

	eng, err := recall.NewEngine(recall.WithDBPath(filepath.Join(tmp, "i.db")))
	if err != nil {
		t.Fatal(err)
	}
	defer eng.Close()

	if _, err := eng.AddCollection("notes", notesDir, "", ""); err != nil {
		t.Fatal(err)
	}
	if _, err := eng.Index(); err != nil {
		t.Fatal(err)
	}

	mock := embed.NewMockEmbedder(store.EmbeddingDimensions)
	first, err := eng.Embed(mock, false)
	if err != nil {
		t.Fatal(err)
	}
	if first.Embedded != 5 {
		t.Fatalf("first embed should process 5 chunks, got %d", first.Embedded)
	}

	// Mutate one file. Re-index. Re-embed (force=false). Only the changed
	// document's chunks should need new vectors.
	if err := os.WriteFile(
		filepath.Join(notesDir, "c.md"),
		[]byte("# Retry\nSubstantially updated retry doc with new content."),
		0o644,
	); err != nil {
		t.Fatal(err)
	}
	if _, err := eng.Index(); err != nil {
		t.Fatal(err)
	}

	second, err := eng.Embed(mock, false)
	if err != nil {
		t.Fatal(err)
	}
	if second.Embedded == 0 {
		t.Errorf("expected to re-embed at least the changed doc, got 0")
	}
	if second.Embedded >= 5 {
		t.Errorf("expected partial re-embed, got full %d (incremental broken)", second.Embedded)
	}

	// Third pass with no changes: nothing to embed.
	third, err := eng.Embed(mock, false)
	if err != nil {
		t.Fatal(err)
	}
	if third.Embedded != 0 {
		t.Errorf("third pass with no changes embedded %d (should be 0)", third.Embedded)
	}
}

// TestForceEmbedReChunksAndReEmbeds pins that `Embed(emb, true)`
// (analog of `recall embed -f`) drops every vector and rebuilds from
// scratch.
func TestForceEmbedReChunksAndReEmbeds(t *testing.T) {
	tmp := t.TempDir()

	dir := filepath.Join(tmp, "n")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "a.md"), []byte("# a\nbody"), 0o644); err != nil {
		t.Fatal(err)
	}

	eng, err := recall.NewEngine(recall.WithDBPath(filepath.Join(tmp, "i.db")))
	if err != nil {
		t.Fatal(err)
	}
	defer eng.Close()
	if _, err := eng.AddCollection("n", dir, "", ""); err != nil {
		t.Fatal(err)
	}
	if _, err := eng.Index(); err != nil {
		t.Fatal(err)
	}

	mock := embed.NewMockEmbedder(store.EmbeddingDimensions)
	if _, err := eng.Embed(mock, false); err != nil {
		t.Fatal(err)
	}
	beforeCount, _ := eng.Store().EmbeddingCount()

	if _, err := eng.Embed(mock, true); err != nil {
		t.Fatal(err)
	}
	afterCount, _ := eng.Store().EmbeddingCount()

	// Same number of chunks → same number of vectors after force.
	if beforeCount != afterCount {
		t.Errorf("force-embed counts differ: before=%d after=%d", beforeCount, afterCount)
	}
}
