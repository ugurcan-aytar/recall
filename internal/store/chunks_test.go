package store

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/ugurcan-aytar/recall/internal/chunk"
)

func TestReplaceAndListChunks(t *testing.T) {
	s := openTestStore(t)
	dir := tempCollectionDir(t, map[string]string{"a.md": "# A\nbody"})
	c, err := s.AddCollection("n", dir, "", "")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := s.IndexCollection(c.ID); err != nil {
		t.Fatal(err)
	}

	doc, err := s.GetDocument("n/a.md")
	if err != nil {
		t.Fatal(err)
	}

	fresh := []chunk.Chunk{
		{Text: "hello", Seq: 0, StartPos: 0, EndPos: 5},
		{Text: "world", Seq: 1, StartPos: 6, EndPos: 11},
	}
	if err := s.ReplaceChunks(doc.ID, fresh); err != nil {
		t.Fatalf("ReplaceChunks: %v", err)
	}

	got, err := s.ListChunks(doc.ID)
	if err != nil {
		t.Fatalf("ListChunks: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("got %d chunks, want 2", len(got))
	}
	if got[0].Seq != 0 || got[0].Content != "hello" {
		t.Errorf("chunk 0 = %+v", got[0])
	}
	if got[1].ContentHash == "" {
		t.Error("content_hash not stored")
	}

	// Replace again → old rows must be gone.
	reduced := []chunk.Chunk{{Text: "solo", Seq: 0, StartPos: 0, EndPos: 4}}
	if err := s.ReplaceChunks(doc.ID, reduced); err != nil {
		t.Fatal(err)
	}
	got, _ = s.ListChunks(doc.ID)
	if len(got) != 1 || got[0].Content != "solo" {
		t.Fatalf("after replace: %+v", got)
	}
}

func TestChunkCount(t *testing.T) {
	s := openTestStore(t)
	n, err := s.ChunkCount()
	if err != nil {
		t.Fatal(err)
	}
	if n != 0 {
		t.Errorf("empty count = %d", n)
	}

	dir := tempCollectionDir(t, map[string]string{
		"a.md": "# a",
		"b.md": "# b",
	})
	c, _ := s.AddCollection("n", dir, "", "")
	if _, err := s.IndexCollection(c.ID); err != nil {
		t.Fatal(err)
	}

	n, err = s.ChunkCount()
	if err != nil {
		t.Fatal(err)
	}
	// Two short docs, each becomes a single chunk.
	if n != 2 {
		t.Errorf("after index: count = %d, want 2", n)
	}
}

func TestIndexCreatesChunks(t *testing.T) {
	s := openTestStore(t)
	dir := tempCollectionDir(t, map[string]string{
		"a.md": "# Alpha\n\nshort doc",
	})
	c, _ := s.AddCollection("n", dir, "", "")
	if _, err := s.IndexCollection(c.ID); err != nil {
		t.Fatal(err)
	}

	doc, err := s.GetDocument("n/a.md")
	if err != nil {
		t.Fatal(err)
	}
	chunks, err := s.ListChunks(doc.ID)
	if err != nil {
		t.Fatal(err)
	}
	if len(chunks) == 0 {
		t.Fatal("indexing should have produced at least one chunk")
	}
	if chunks[0].ContentHash == "" {
		t.Error("chunk stored without content_hash")
	}
}

func TestIndexUpdateReChunks(t *testing.T) {
	s := openTestStore(t)
	dir := tempCollectionDir(t, map[string]string{"a.md": "# A\nfirst"})
	c, _ := s.AddCollection("n", dir, "", "")
	if _, err := s.IndexCollection(c.ID); err != nil {
		t.Fatal(err)
	}

	doc, _ := s.GetDocument("n/a.md")
	before, _ := s.ListChunks(doc.ID)
	firstHash := before[0].ContentHash

	// Rewrite the file with different content.
	path := filepath.Join(dir, "a.md")
	if err := os.WriteFile(path, []byte("# A\ncompletely different body now"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := s.IndexCollection(c.ID); err != nil {
		t.Fatal(err)
	}

	doc2, _ := s.GetDocument("n/a.md")
	after, _ := s.ListChunks(doc2.ID)
	if len(after) == 0 {
		t.Fatal("no chunks after update")
	}
	if after[0].ContentHash == firstHash {
		t.Error("chunk content_hash did not change after update")
	}
}
