package store

import (
	"fmt"
	"os"
)

// CleanupStats reports what Cleanup did.
type CleanupStats struct {
	OrphanedChunks      int   // chunks whose document is gone (sanity check; FK cascade should keep this 0)
	StaleEmbeddings     int   // chunk_embeddings whose chunk is gone (vec0 has no FK, so this is the real source of bloat)
	BytesBefore         int64 // DB file size before cleanup
	BytesAfter          int64 // DB file size after VACUUM
}

// Cleanup removes orphaned rows and runs VACUUM. Safe to run any time;
// idempotent.
//
// Source of orphans:
//
//   - chunks (cascaded from documents via FK, so usually empty — defensive)
//   - chunk_embeddings (vec0 virtual table, no FK enforcement; rows here
//     become stale every time a chunk is replaced or its document is
//     re-indexed)
func (s *Store) Cleanup() (CleanupStats, error) {
	var stats CleanupStats

	if info, err := os.Stat(s.path); err == nil {
		stats.BytesBefore = info.Size()
	}

	// 1) Orphaned chunks (defensive — should be 0 thanks to FK cascade).
	res, err := s.db.Exec(`
		DELETE FROM chunks
		WHERE document_id NOT IN (SELECT id FROM documents)`)
	if err != nil {
		return stats, fmt.Errorf("delete orphaned chunks: %w", err)
	}
	if n, err := res.RowsAffected(); err == nil {
		stats.OrphanedChunks = int(n)
	}

	// 2) Stale embeddings — vec0 has no FK so we walk the table and remove
	// any chunk_id no longer present in chunks.
	rows, err := s.db.Query(`
		SELECT chunk_id FROM chunk_embeddings
		WHERE chunk_id NOT IN (SELECT id FROM chunks)`)
	if err != nil {
		return stats, fmt.Errorf("scan stale embeddings: %w", err)
	}
	var stale []int64
	for rows.Next() {
		var id int64
		if err := rows.Scan(&id); err != nil {
			rows.Close()
			return stats, err
		}
		stale = append(stale, id)
	}
	rows.Close()
	if err := rows.Err(); err != nil {
		return stats, err
	}
	for _, id := range stale {
		if _, err := s.db.Exec(`DELETE FROM chunk_embeddings WHERE chunk_id = ?`, id); err != nil {
			return stats, fmt.Errorf("delete stale embedding %d: %w", id, err)
		}
	}
	stats.StaleEmbeddings = len(stale)

	// 3) VACUUM. Cannot run inside a transaction; run it last so all the
	// DELETEs above can actually free pages.
	if _, err := s.db.Exec(`VACUUM`); err != nil {
		return stats, fmt.Errorf("vacuum: %w", err)
	}

	if info, err := os.Stat(s.path); err == nil {
		stats.BytesAfter = info.Size()
	}
	return stats, nil
}
