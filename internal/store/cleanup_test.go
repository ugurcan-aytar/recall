package store

import (
	"testing"

	"github.com/ugurcan-aytar/recall/internal/embed"
)

func TestCleanupRemovesStaleEmbeddings(t *testing.T) {
	s := openTestStore(t)
	mock := embed.NewMockEmbedder(EmbeddingDimensions)

	// Index a doc, embed it, then delete the chunks directly so the
	// embeddings become orphaned.
	dir := tempCollectionDir(t, map[string]string{"a.md": "# A\nbody"})
	c, _ := s.AddCollection("n", dir, "", "")
	if _, err := s.IndexCollection(c.ID); err != nil {
		t.Fatal(err)
	}
	chunks, _ := s.ChunksNeedingEmbed()
	for _, ch := range chunks {
		v, _ := mock.EmbedSingle(ch.Content)
		if err := s.UpsertEmbedding(ch.ID, v); err != nil {
			t.Fatal(err)
		}
	}
	pre, _ := s.EmbeddingCount()
	if pre == 0 {
		t.Fatal("expected at least one embedding before cleanup")
	}

	// Manually orphan the embeddings by wiping chunks.
	if _, err := s.DB().Exec(`DELETE FROM chunks`); err != nil {
		t.Fatal(err)
	}

	stats, err := s.Cleanup()
	if err != nil {
		t.Fatalf("Cleanup: %v", err)
	}
	if stats.StaleEmbeddings != pre {
		t.Errorf("StaleEmbeddings = %d, want %d", stats.StaleEmbeddings, pre)
	}
	post, _ := s.EmbeddingCount()
	if post != 0 {
		t.Errorf("embeddings after cleanup = %d, want 0", post)
	}
}

func TestCleanupVacuumShrinksFile(t *testing.T) {
	s := openTestStore(t)

	// Create a chunky doc, then drop it so VACUUM has space to reclaim.
	dir := tempCollectionDir(t, map[string]string{"big.md": ""})
	c, _ := s.AddCollection("n", dir, "", "")
	// Insert a sizeable document directly so we don't depend on the indexer.
	bigContent := make([]byte, 500_000)
	for i := range bigContent {
		bigContent[i] = 'x'
	}
	if _, err := s.DB().Exec(
		`INSERT INTO documents(collection_id, path, absolute_path, title, content, content_hash, doc_id) VALUES (?, ?, ?, ?, ?, ?, ?)`,
		c.ID, "big.md", dir+"/big.md", "big", string(bigContent), "h", "abcdef",
	); err != nil {
		t.Fatal(err)
	}
	// Force the page cache to disk.
	if _, err := s.DB().Exec(`PRAGMA wal_checkpoint(FULL)`); err != nil {
		t.Fatal(err)
	}
	// Drop the doc.
	if _, err := s.DB().Exec(`DELETE FROM documents`); err != nil {
		t.Fatal(err)
	}

	stats, err := s.Cleanup()
	if err != nil {
		t.Fatal(err)
	}
	if stats.BytesAfter > stats.BytesBefore {
		t.Errorf("VACUUM grew the file: before=%d after=%d", stats.BytesBefore, stats.BytesAfter)
	}
}

func TestCleanupNoopOnEmptyStore(t *testing.T) {
	s := openTestStore(t)
	stats, err := s.Cleanup()
	if err != nil {
		t.Fatalf("Cleanup on empty store: %v", err)
	}
	if stats.OrphanedChunks != 0 {
		t.Errorf("OrphanedChunks = %d, want 0", stats.OrphanedChunks)
	}
	if stats.StaleEmbeddings != 0 {
		t.Errorf("StaleEmbeddings = %d, want 0", stats.StaleEmbeddings)
	}
}

func TestCleanupOrphanedChunks(t *testing.T) {
	s := openTestStore(t)

	// Bypass FK cascade by inserting a chunk with a bogus document_id while
	// foreign keys are off. Confirms the orphan-chunks pass actually fires.
	if _, err := s.DB().Exec(`PRAGMA foreign_keys = OFF`); err != nil {
		t.Fatal(err)
	}
	if _, err := s.DB().Exec(
		`INSERT INTO chunks(document_id, seq, start_pos, end_pos, content, content_hash) VALUES (9999, 0, 0, 1, 'x', 'h')`,
	); err != nil {
		t.Fatal(err)
	}
	if _, err := s.DB().Exec(`PRAGMA foreign_keys = ON`); err != nil {
		t.Fatal(err)
	}

	stats, err := s.Cleanup()
	if err != nil {
		t.Fatalf("Cleanup: %v", err)
	}
	if stats.OrphanedChunks < 1 {
		t.Errorf("OrphanedChunks = %d, want ≥ 1", stats.OrphanedChunks)
	}
}
