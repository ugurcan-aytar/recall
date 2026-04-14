package store

import (
	"testing"
)

func TestSerializeEmbedding(t *testing.T) {
	vec := make([]float32, EmbeddingDimensions)
	for i := range vec {
		vec[i] = float32(i) / 100.0
	}
	blob, err := SerializeEmbedding(vec)
	if err != nil {
		t.Fatalf("SerializeEmbedding: %v", err)
	}
	// sqlite-vec encodes float32 as 4 bytes each.
	if want := EmbeddingDimensions * 4; len(blob) != want {
		t.Errorf("blob length = %d, want %d", len(blob), want)
	}
}

func TestVec0TableRoundTrip(t *testing.T) {
	s := openTestStore(t)

	vec := make([]float32, EmbeddingDimensions)
	for i := range vec {
		vec[i] = 0.1
	}
	blob, err := SerializeEmbedding(vec)
	if err != nil {
		t.Fatal(err)
	}

	if _, err := s.DB().Exec(
		`INSERT INTO chunk_embeddings(chunk_id, embedding) VALUES (?, ?)`,
		1, blob,
	); err != nil {
		t.Fatalf("insert into vec0: %v", err)
	}

	var n int
	if err := s.DB().QueryRow(
		`SELECT COUNT(*) FROM chunk_embeddings WHERE chunk_id = ?`, 1,
	).Scan(&n); err != nil {
		t.Fatalf("count: %v", err)
	}
	if n != 1 {
		t.Errorf("row not persisted: count=%d", n)
	}
}
