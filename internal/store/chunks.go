package store

import (
	"database/sql"
	"fmt"

	"github.com/ugurcan-aytar/recall/internal/chunk"
)

// StoredChunk mirrors a row from the chunks table.
type StoredChunk struct {
	ID          int64
	DocumentID  int64
	Seq         int
	StartPos    int
	EndPos      int
	Content     string
	ContentHash string
}

// ReplaceChunks writes a fresh set of chunks for a document, replacing any
// existing rows. The call is atomic — either every chunk lands or none does.
//
// R2.2 stores chunks synchronously on insert / update. R3 will consult the
// content_hash column to skip re-embedding chunks whose text did not change.
func (s *Store) ReplaceChunks(documentID int64, chunks []chunk.Chunk) error {
	tx, err := s.db.Begin()
	if err != nil {
		return fmt.Errorf("begin tx: %w", err)
	}
	defer func() { _ = tx.Rollback() }()

	if _, err := tx.Exec(`DELETE FROM chunks WHERE document_id = ?`, documentID); err != nil {
		return fmt.Errorf("clear chunks for doc %d: %w", documentID, err)
	}

	stmt, err := tx.Prepare(`
		INSERT INTO chunks(document_id, seq, start_pos, end_pos, content, content_hash)
		VALUES (?, ?, ?, ?, ?, ?)`)
	if err != nil {
		return fmt.Errorf("prepare chunk insert: %w", err)
	}
	defer stmt.Close()

	for _, c := range chunks {
		if _, err := stmt.Exec(
			documentID, c.Seq, c.StartPos, c.EndPos, c.Text, c.ContentHash(),
		); err != nil {
			return fmt.Errorf("insert chunk seq=%d: %w", c.Seq, err)
		}
	}

	if err := tx.Commit(); err != nil {
		return fmt.Errorf("commit chunks: %w", err)
	}
	return nil
}

// ListChunks returns every chunk belonging to a document, ordered by seq.
func (s *Store) ListChunks(documentID int64) ([]StoredChunk, error) {
	rows, err := s.db.Query(`
		SELECT id, document_id, seq, start_pos, end_pos, content, content_hash
		FROM chunks WHERE document_id = ? ORDER BY seq ASC`, documentID)
	if err != nil {
		return nil, fmt.Errorf("query chunks: %w", err)
	}
	defer rows.Close()

	var out []StoredChunk
	for rows.Next() {
		var c StoredChunk
		if err := rows.Scan(
			&c.ID, &c.DocumentID, &c.Seq, &c.StartPos, &c.EndPos,
			&c.Content, &c.ContentHash,
		); err != nil {
			return nil, err
		}
		out = append(out, c)
	}
	return out, rows.Err()
}

// ChunkCount returns the total number of chunk rows in the database.
func (s *Store) ChunkCount() (int, error) {
	var n int
	err := s.db.QueryRow(`SELECT COUNT(*) FROM chunks`).Scan(&n)
	if err != nil && err != sql.ErrNoRows {
		return 0, err
	}
	return n, nil
}
