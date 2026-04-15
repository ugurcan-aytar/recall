package store

import (
	"database/sql"
	"errors"
	"fmt"
)

// sqliteVecMaxK is the hard ceiling sqlite-vec enforces on its knn
// MATCH parameter (`AND k = ?`). Exceeding it fails the SQL with
// "k value in knn query too large, provided X and the limit is 4096".
// Kept as a named constant so future bumps (sqlite-vec ≥ ?) are a
// one-line edit.
const sqliteVecMaxK = 4096

// SearchVector runs a KNN cosine-distance query against chunk_embeddings
// and maps winning chunks back to their documents. One row per document
// (best chunk) with similarity = 1 / (1 + distance).
func (s *Store) SearchVector(queryVec []float32, opts SearchOptions) ([]SearchResult, error) {
	if len(queryVec) == 0 {
		return nil, errors.New("empty query vector")
	}
	if len(queryVec) != EmbeddingDimensions {
		return nil, fmt.Errorf(
			"query vector has %d dims, chunk_embeddings expects %d",
			len(queryVec), EmbeddingDimensions,
		)
	}

	limit := opts.Limit
	if opts.All {
		// sqlite-vec caps its knn `k` parameter at 4096 per query.
		// --all historically set limit to 1 million which 4×-over-
		// fetched to 4 million and blew up as
		//   "k value in knn query too large, provided 4000000 and the limit is 4096"
		// Cap here instead so "--all" means "as many as sqlite-vec
		// can return in a single knn pass". Callers that genuinely
		// need paginated full scans should switch to SearchBM25 +
		// Get, or drive vec0 rowids directly.
		limit = sqliteVecMaxK
	} else if limit <= 0 {
		limit = 5
	}

	blob, err := SerializeEmbedding(queryVec)
	if err != nil {
		return nil, fmt.Errorf("serialize query vector: %w", err)
	}

	// Over-fetch: one document may own many chunks, we want a
	// unique-doc top-N. 4× limit is a safe empirical margin —
	// but never past sqlite-vec's hard k cap.
	knnLimit := limit * 4
	if knnLimit < 50 {
		knnLimit = 50
	}
	if knnLimit > sqliteVecMaxK {
		knnLimit = sqliteVecMaxK
	}

	colls := splitCollections(opts.Collection)
	var rows *sql.Rows
	switch len(colls) {
	case 0:
		rows, err = s.DB().Query(`
			SELECT
				d.id, d.doc_id, COALESCE(d.title, ''), d.path, d.absolute_path, c.name,
				ce.distance, ch.content
			FROM chunk_embeddings ce
			JOIN chunks ch ON ch.id = ce.chunk_id
			JOIN documents d ON d.id = ch.document_id
			JOIN collections c ON c.id = d.collection_id
			WHERE ce.embedding MATCH ?
			  AND k = ?
			ORDER BY ce.distance ASC`,
			blob, knnLimit,
		)
	case 1:
		rows, err = s.DB().Query(`
			SELECT
				d.id, d.doc_id, COALESCE(d.title, ''), d.path, d.absolute_path, c.name,
				ce.distance, ch.content
			FROM chunk_embeddings ce
			JOIN chunks ch ON ch.id = ce.chunk_id
			JOIN documents d ON d.id = ch.document_id
			JOIN collections c ON c.id = d.collection_id
			WHERE ce.embedding MATCH ?
			  AND k = ?
			  AND c.name = ?
			ORDER BY ce.distance ASC`,
			blob, knnLimit, colls[0],
		)
	default:
		query := `
			SELECT
				d.id, d.doc_id, COALESCE(d.title, ''), d.path, d.absolute_path, c.name,
				ce.distance, ch.content
			FROM chunk_embeddings ce
			JOIN chunks ch ON ch.id = ce.chunk_id
			JOIN documents d ON d.id = ch.document_id
			JOIN collections c ON c.id = d.collection_id
			WHERE ce.embedding MATCH ?
			  AND k = ?
			  AND c.name IN (` + placeholders(len(colls)) + `)
			ORDER BY ce.distance ASC`
		args := make([]any, 0, 2+len(colls))
		args = append(args, blob, knnLimit)
		for _, c := range colls {
			args = append(args, c)
		}
		rows, err = s.DB().Query(query, args...)
	}
	if err != nil {
		return nil, fmt.Errorf("vector query: %w", err)
	}
	defer rows.Close()

	// Keep the best (lowest-distance) chunk per document.
	seen := map[int64]struct{}{}
	var out []SearchResult

	for rows.Next() {
		var (
			docID                                               int64
			docUID, title, path, abs, collName, chunkContent    string
			distance                                            float64
		)
		if err := rows.Scan(&docID, &docUID, &title, &path, &abs, &collName, &distance, &chunkContent); err != nil {
			return nil, fmt.Errorf("scan vector row: %w", err)
		}
		if _, dup := seen[docID]; dup {
			continue
		}
		seen[docID] = struct{}{}

		sim := 1.0 / (1.0 + distance)
		if opts.MinScore > 0 && sim < opts.MinScore {
			continue
		}
		out = append(out, SearchResult{
			DocID:          docUID,
			Title:          title,
			Path:           path,
			AbsolutePath:   abs,
			CollectionName: collName,
			Score:          sim,
			Snippet:        snippetFromChunk(chunkContent, 240),
		})
		if len(out) >= limit {
			break
		}
	}
	return out, rows.Err()
}

// snippetFromChunk returns a trimmed preview of a chunk's text so vector
// results have something readable to display. No FTS5 markers — the vector
// backend has no per-term match to highlight.
func snippetFromChunk(s string, max int) string {
	if max <= 0 || len(s) <= max {
		return s
	}
	return s[:max] + "..."
}

// UpsertEmbedding writes a single chunk's vector into chunk_embeddings,
// replacing any existing row for that chunk_id.
func (s *Store) UpsertEmbedding(chunkID int64, vec []float32) error {
	if len(vec) != EmbeddingDimensions {
		return fmt.Errorf(
			"vector has %d dims, chunk_embeddings expects %d",
			len(vec), EmbeddingDimensions,
		)
	}
	blob, err := SerializeEmbedding(vec)
	if err != nil {
		return fmt.Errorf("serialize vector: %w", err)
	}
	// vec0 tables do not support UPSERT; delete-then-insert inside a tx.
	tx, err := s.DB().Begin()
	if err != nil {
		return err
	}
	defer func() { _ = tx.Rollback() }()

	if _, err := tx.Exec(`DELETE FROM chunk_embeddings WHERE chunk_id = ?`, chunkID); err != nil {
		return fmt.Errorf("delete old embedding for chunk %d: %w", chunkID, err)
	}
	if _, err := tx.Exec(
		`INSERT INTO chunk_embeddings(chunk_id, embedding) VALUES (?, ?)`,
		chunkID, blob,
	); err != nil {
		return fmt.Errorf("insert embedding for chunk %d: %w", chunkID, err)
	}
	return tx.Commit()
}

// EmbeddedChunkIDs returns the set of chunk IDs that currently have an
// embedding row. Used by the incremental embedder to know what to skip.
func (s *Store) EmbeddedChunkIDs() (map[int64]struct{}, error) {
	rows, err := s.DB().Query(`SELECT chunk_id FROM chunk_embeddings`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := map[int64]struct{}{}
	for rows.Next() {
		var id int64
		if err := rows.Scan(&id); err != nil {
			return nil, err
		}
		out[id] = struct{}{}
	}
	return out, rows.Err()
}

// EmbeddingCount returns the total number of vectors currently stored.
func (s *Store) EmbeddingCount() (int, error) {
	var n int
	err := s.DB().QueryRow(`SELECT COUNT(*) FROM chunk_embeddings`).Scan(&n)
	return n, err
}

// DeleteEmbeddingsForDocument clears vectors whose chunks belong to the
// given document. Callers use this when re-embedding with a different
// model or when chunks were replaced.
func (s *Store) DeleteEmbeddingsForDocument(documentID int64) error {
	// Two-step: pick chunk ids, delete from vec0.
	rows, err := s.DB().Query(`SELECT id FROM chunks WHERE document_id = ?`, documentID)
	if err != nil {
		return err
	}
	var ids []int64
	for rows.Next() {
		var id int64
		if err := rows.Scan(&id); err != nil {
			rows.Close()
			return err
		}
		ids = append(ids, id)
	}
	rows.Close()

	for _, id := range ids {
		if _, err := s.DB().Exec(`DELETE FROM chunk_embeddings WHERE chunk_id = ?`, id); err != nil {
			return err
		}
	}
	return nil
}

// ChunksNeedingEmbed returns chunks whose content_hash is not already paired
// with an embedding row. Used by the incremental embedder.
type ChunkForEmbed struct {
	ID          int64
	DocumentID  int64
	DocTitle    string
	Content     string
	ContentHash string
}

func (s *Store) ChunksNeedingEmbed() ([]ChunkForEmbed, error) {
	rows, err := s.DB().Query(`
		SELECT ch.id, ch.document_id, COALESCE(d.title, ''), ch.content, ch.content_hash
		FROM chunks ch
		JOIN documents d ON d.id = ch.document_id
		LEFT JOIN chunk_embeddings ce ON ce.chunk_id = ch.id
		WHERE ce.chunk_id IS NULL
		ORDER BY ch.id`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []ChunkForEmbed
	for rows.Next() {
		var c ChunkForEmbed
		if err := rows.Scan(&c.ID, &c.DocumentID, &c.DocTitle, &c.Content, &c.ContentHash); err != nil {
			return nil, err
		}
		out = append(out, c)
	}
	return out, rows.Err()
}

// AllChunksForEmbed returns every chunk regardless of embedding state. Used
// by `recall embed -f` (force mode) to redo the entire corpus.
func (s *Store) AllChunksForEmbed() ([]ChunkForEmbed, error) {
	rows, err := s.DB().Query(`
		SELECT ch.id, ch.document_id, COALESCE(d.title, ''), ch.content, ch.content_hash
		FROM chunks ch
		JOIN documents d ON d.id = ch.document_id
		ORDER BY ch.id`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []ChunkForEmbed
	for rows.Next() {
		var c ChunkForEmbed
		if err := rows.Scan(&c.ID, &c.DocumentID, &c.DocTitle, &c.Content, &c.ContentHash); err != nil {
			return nil, err
		}
		out = append(out, c)
	}
	return out, rows.Err()
}

// DropAllEmbeddings wipes the whole chunk_embeddings table. Called when
// switching models.
func (s *Store) DropAllEmbeddings() error {
	_, err := s.DB().Exec(`DELETE FROM chunk_embeddings`)
	return err
}
