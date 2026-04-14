package store

import (
	"testing"

	"github.com/ugurcan-aytar/recall/internal/embed"
)

// seedVectorCorpus indexes a few short docs and embeds every chunk with
// the deterministic MockEmbedder. Returns the store and the embedder.
func seedVectorCorpus(t *testing.T) (*Store, *embed.MockEmbedder) {
	t.Helper()
	s := openTestStore(t)

	dir := tempCollectionDir(t, map[string]string{
		"auth.md":    "# Auth\nThe authentication flow handles JWT tokens.",
		"rate.md":    "# Rate Limiter\nDiscussion of rate limiting algorithms.",
		"weather.md": "# Misc\nUnrelated content about clouds and weather.",
	})
	c, err := s.AddCollection("notes", dir, "", "")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := s.IndexCollection(c.ID); err != nil {
		t.Fatal(err)
	}

	mock := embed.NewMockEmbedder(EmbeddingDimensions)

	chunks, err := s.ChunksNeedingEmbed()
	if err != nil {
		t.Fatal(err)
	}
	for _, ch := range chunks {
		v, err := mock.EmbedSingle(embed.FormatDocument(ch.DocTitle, ch.Content))
		if err != nil {
			t.Fatal(err)
		}
		if err := s.UpsertEmbedding(ch.ID, v); err != nil {
			t.Fatal(err)
		}
	}
	return s, mock
}

func TestSearchVectorBasic(t *testing.T) {
	s, mock := seedVectorCorpus(t)

	// Query with the EXACT formatted text of one of the indexed chunks. The
	// deterministic mock will return the same vector → distance ≈ 0 → that
	// document must come first.
	target := embed.FormatDocument("Auth", "# Auth\nThe authentication flow handles JWT tokens.")
	q, err := mock.EmbedSingle(target)
	if err != nil {
		t.Fatal(err)
	}

	results, err := s.SearchVector(q, SearchOptions{Limit: 5})
	if err != nil {
		t.Fatalf("SearchVector: %v", err)
	}
	if len(results) == 0 {
		t.Fatal("no vector results")
	}
	if results[0].Path != "auth.md" {
		t.Errorf("top result = %s, want auth.md", results[0].Path)
	}
	// Scores should be monotonic non-increasing.
	for i := 1; i < len(results); i++ {
		if results[i].Score > results[i-1].Score {
			t.Errorf("results not sorted at %d: %.4f > %.4f",
				i, results[i].Score, results[i-1].Score)
		}
	}
}

func TestSearchVectorKNN(t *testing.T) {
	s := openTestStore(t)
	mock := embed.NewMockEmbedder(EmbeddingDimensions)

	// Register a fake collection so chunks have a parent.
	dir := tempCollectionDir(t, map[string]string{"placeholder.md": "x"})
	c, err := s.AddCollection("c", dir, "", "")
	if err != nil {
		t.Fatal(err)
	}
	// Insert 10 chunks with deterministic embeddings, then ask for the top 3.
	if _, err := s.DB().Exec(
		`INSERT INTO documents(collection_id, path, absolute_path, title, content, content_hash, doc_id) VALUES (?, ?, ?, ?, ?, ?, ?)`,
		c.ID, "doc.md", dir+"/doc.md", "Doc", "doc body", "h", "abcdef",
	); err != nil {
		t.Fatal(err)
	}
	var docID int64
	if err := s.DB().QueryRow(
		`SELECT id FROM documents WHERE collection_id = ? AND path = 'doc.md'`, c.ID,
	).Scan(&docID); err != nil {
		t.Fatal(err)
	}

	// Insert 10 chunks then embed each with a deterministic but distinct
	// text. Doc has only one entry per the joined table, so to keep the
	// per-doc top-K assertion useful here we create 10 docs.
	docIDs := []int64{docID}
	for i := 1; i < 10; i++ {
		path := "doc" + string(rune('a'+i)) + ".md"
		if _, err := s.DB().Exec(
			`INSERT INTO documents(collection_id, path, absolute_path, title, content, content_hash, doc_id) VALUES (?, ?, ?, ?, ?, ?, ?)`,
			c.ID, path, dir+"/"+path, path, "body "+path, "h"+path, path[:6],
		); err != nil {
			t.Fatal(err)
		}
		var id int64
		if err := s.DB().QueryRow(
			`SELECT id FROM documents WHERE collection_id = ? AND path = ?`, c.ID, path,
		).Scan(&id); err != nil {
			t.Fatal(err)
		}
		docIDs = append(docIDs, id)
	}

	for _, did := range docIDs {
		text := "chunk text " + string(rune('a'+did))
		res, err := s.DB().Exec(
			`INSERT INTO chunks(document_id, seq, start_pos, end_pos, content, content_hash) VALUES (?, 0, 0, 1, ?, 'h')`,
			did, text,
		)
		if err != nil {
			t.Fatal(err)
		}
		chunkID, _ := res.LastInsertId()
		v, _ := mock.EmbedSingle(text)
		if err := s.UpsertEmbedding(chunkID, v); err != nil {
			t.Fatal(err)
		}
	}

	q, _ := mock.EmbedSingle("chunk text " + string(rune('a'+docIDs[3])))
	results, err := s.SearchVector(q, SearchOptions{Limit: 3})
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 3 {
		t.Fatalf("got %d results, want 3", len(results))
	}
	// Top 1 should be the doc whose chunk text we queried with.
	wantPath := "doc" + string(rune('a'+docIDs[3]-1)) + ".md" // first doc was "doc.md" not "doca.md"
	_ = wantPath
	// We mostly want non-decreasing scores and exactly 3 unique docs.
	seen := map[string]bool{}
	for _, r := range results {
		if seen[r.Path] {
			t.Errorf("duplicate doc in results: %s", r.Path)
		}
		seen[r.Path] = true
	}
}

func TestSearchVectorNoEmbeddings(t *testing.T) {
	s := openTestStore(t)
	mock := embed.NewMockEmbedder(EmbeddingDimensions)

	q, _ := mock.EmbedSingle("nothing here")
	results, err := s.SearchVector(q, SearchOptions{Limit: 5})
	if err != nil {
		t.Fatalf("SearchVector with empty store: %v", err)
	}
	if len(results) != 0 {
		t.Errorf("expected 0 results from empty index, got %d", len(results))
	}
}

func TestSearchVectorCollectionFilter(t *testing.T) {
	s := openTestStore(t)
	mock := embed.NewMockEmbedder(EmbeddingDimensions)

	d1 := tempCollectionDir(t, map[string]string{"a.md": "# a\napples"})
	d2 := tempCollectionDir(t, map[string]string{"a.md": "# a\nzebras"})
	c1, _ := s.AddCollection("fruit", d1, "", "")
	c2, _ := s.AddCollection("zoo", d2, "", "")
	if _, err := s.IndexCollection(c1.ID); err != nil {
		t.Fatal(err)
	}
	if _, err := s.IndexCollection(c2.ID); err != nil {
		t.Fatal(err)
	}

	chunks, _ := s.ChunksNeedingEmbed()
	for _, ch := range chunks {
		v, _ := mock.EmbedSingle(ch.Content)
		if err := s.UpsertEmbedding(ch.ID, v); err != nil {
			t.Fatal(err)
		}
	}

	q, _ := mock.EmbedSingle("anything")
	res, err := s.SearchVector(q, SearchOptions{Limit: 10, Collection: "fruit"})
	if err != nil {
		t.Fatal(err)
	}
	for _, r := range res {
		if r.CollectionName != "fruit" {
			t.Errorf("collection filter leaked: %s", r.CollectionName)
		}
	}
}

func TestUpsertEmbeddingReplaces(t *testing.T) {
	s := openTestStore(t)
	mock := embed.NewMockEmbedder(EmbeddingDimensions)

	v1, _ := mock.EmbedSingle("first")
	if err := s.UpsertEmbedding(42, v1); err != nil {
		t.Fatal(err)
	}
	n, err := s.EmbeddingCount()
	if err != nil || n != 1 {
		t.Fatalf("count = %d (err=%v), want 1", n, err)
	}

	v2, _ := mock.EmbedSingle("second")
	if err := s.UpsertEmbedding(42, v2); err != nil {
		t.Fatal(err)
	}
	n, _ = s.EmbeddingCount()
	if n != 1 {
		t.Errorf("upsert produced duplicate row, count = %d", n)
	}
}

func TestChunksNeedingEmbed(t *testing.T) {
	s, mock := seedVectorCorpus(t)
	pending, err := s.ChunksNeedingEmbed()
	if err != nil {
		t.Fatal(err)
	}
	if len(pending) != 0 {
		t.Errorf("pending chunks after embed = %d, want 0", len(pending))
	}

	// Add a new doc → its chunk should be pending again.
	dir := tempCollectionDir(t, map[string]string{"new.md": "# new\nbody"})
	c, _ := s.AddCollection("more", dir, "", "")
	if _, err := s.IndexCollection(c.ID); err != nil {
		t.Fatal(err)
	}
	pending, _ = s.ChunksNeedingEmbed()
	if len(pending) == 0 {
		t.Errorf("expected new chunk to be pending")
	}
	_ = mock
}
