package store

import (
	"database/sql"
	"fmt"
	"strings"
)

// splitCollections normalises an [SearchOptions.Collection] value into a
// slice. Empty input returns nil ("no filter"); single value returns one
// element; comma-separated values are trimmed and split. Used by every
// search backend to support `-c repo1,repo2,repo3` (Phase R2b.5).
func splitCollections(raw string) []string {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return nil
	}
	parts := strings.Split(raw, ",")
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		if p = strings.TrimSpace(p); p != "" {
			out = append(out, p)
		}
	}
	return out
}

// placeholders returns "?, ?, …" with n entries — for IN-clause binding.
func placeholders(n int) string {
	if n <= 0 {
		return ""
	}
	return strings.Repeat("?,", n-1) + "?"
}

// SearchOptions configures a BM25 query.
type SearchOptions struct {
	Query      string
	Limit      int    // 0 means "use default" (5)
	Collection string // "" matches all collections
	MinScore   float64
	All        bool // when true, ignore Limit
}

// SearchResult is one hit from a BM25 search, matching what the CLI prints.
type SearchResult struct {
	DocID          string
	Title          string
	Path           string
	AbsolutePath   string
	CollectionName string
	Score          float64 // absolute bm25 score (higher = better for display)
	Snippet        string
}

// snippetMarkStart / snippetMarkEnd are the tokens FTS5 wraps around matched
// terms inside snippet(). Consumers can look for these and swap in ANSI codes,
// HTML, markdown, etc. See ROADMAP.md Task R1.4 "colorized CLI output".
const (
	snippetMarkStart = "\x1b[1m"
	snippetMarkEnd   = "\x1b[0m"
)

// searchBM25SQL matches documents across all collections. Score is flipped
// positive (bm25 is negative, more negative = better) and rows are ordered
// best-first.
const searchBM25SQL = `
SELECT d.id, d.doc_id, COALESCE(d.title, ''), d.path, d.absolute_path, c.name,
       -bm25(documents_fts) AS score,
       snippet(documents_fts, 1, ?, ?, '...', 15) AS snippet
FROM documents_fts
JOIN documents d ON d.id = documents_fts.rowid
JOIN collections c ON c.id = d.collection_id
WHERE documents_fts MATCH ?
ORDER BY bm25(documents_fts)
LIMIT ?`

// searchBM25InCollectionSQL is the same query with a collection-name filter.
const searchBM25InCollectionSQL = `
SELECT d.id, d.doc_id, COALESCE(d.title, ''), d.path, d.absolute_path, c.name,
       -bm25(documents_fts) AS score,
       snippet(documents_fts, 1, ?, ?, '...', 15) AS snippet
FROM documents_fts
JOIN documents d ON d.id = documents_fts.rowid
JOIN collections c ON c.id = d.collection_id
WHERE documents_fts MATCH ? AND c.name = ?
ORDER BY bm25(documents_fts)
LIMIT ?`

// SearchBM25 runs an FTS5 MATCH query ranked by the SQLite bm25() function.
// The hot path uses prepared statements cached on the Store.
func (s *Store) SearchBM25(opts SearchOptions) ([]SearchResult, error) {
	q := strings.TrimSpace(opts.Query)
	if q == "" {
		return nil, fmt.Errorf("empty search query")
	}

	limit := opts.Limit
	if opts.All {
		limit = 1_000_000
	} else if limit <= 0 {
		limit = 5
	}

	colls := splitCollections(opts.Collection)
	var (
		rows *sql.Rows
		err  error
	)
	switch len(colls) {
	case 0:
		rows, err = s.stmtSearchBM25.Query(snippetMarkStart, snippetMarkEnd, q, limit)
	case 1:
		rows, err = s.stmtSearchBM25InColl.Query(snippetMarkStart, snippetMarkEnd, q, colls[0], limit)
	default:
		query := `
SELECT d.id, d.doc_id, COALESCE(d.title, ''), d.path, d.absolute_path, c.name,
       -bm25(documents_fts) AS score,
       snippet(documents_fts, 1, ?, ?, '...', 15) AS snippet
FROM documents_fts
JOIN documents d ON d.id = documents_fts.rowid
JOIN collections c ON c.id = d.collection_id
WHERE documents_fts MATCH ? AND c.name IN (` + placeholders(len(colls)) + `)
ORDER BY bm25(documents_fts)
LIMIT ?`
		args := make([]any, 0, 4+len(colls))
		args = append(args, snippetMarkStart, snippetMarkEnd, q)
		for _, c := range colls {
			args = append(args, c)
		}
		args = append(args, limit)
		rows, err = s.db.Query(query, args...)
	}
	if err != nil {
		return nil, fmt.Errorf("bm25 query: %w", err)
	}
	defer rows.Close()

	var out []SearchResult
	for rows.Next() {
		var (
			r   SearchResult
			id  int64
			tmp string
		)
		if err := rows.Scan(
			&id, &r.DocID, &r.Title, &r.Path, &r.AbsolutePath, &r.CollectionName,
			&r.Score, &tmp,
		); err != nil {
			return nil, fmt.Errorf("scan search row: %w", err)
		}
		r.Snippet = tmp
		if opts.MinScore > 0 && r.Score < opts.MinScore {
			continue
		}
		out = append(out, r)
	}
	return out, rows.Err()
}
