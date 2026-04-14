package store

import (
	"database/sql"
	"errors"
	"fmt"
	"strings"
)

// PathContext is a descriptive annotation attached to a collection, path, or
// the global tree. It is appended to search results to help LLMs and users
// disambiguate where a hit comes from.
type PathContext struct {
	ID         int64
	Collection string // "" means global
	Path       string // "/" means "the whole collection" (or global when Collection is "")
	Context    string
}

// ErrContextNotFound is returned by RemoveContext when nothing matches.
var ErrContextNotFound = errors.New("context not found")

// AddContext upserts a context row. An empty collection means the context is
// global. An empty path defaults to "/".
func (s *Store) AddContext(collection, path, context string) error {
	if strings.TrimSpace(context) == "" {
		return errors.New("context text cannot be empty")
	}
	if path == "" {
		path = "/"
	}
	_, err := s.db.Exec(`
		INSERT INTO path_contexts(collection, path, context)
		VALUES (?, ?, ?)
		ON CONFLICT(collection, path) DO UPDATE SET context = excluded.context`,
		nullableString(collection), path, context,
	)
	if err != nil {
		return fmt.Errorf("upsert context: %w", err)
	}
	return nil
}

// RemoveContext deletes a single path context.
func (s *Store) RemoveContext(collection, path string) error {
	if path == "" {
		path = "/"
	}
	res, err := s.db.Exec(
		`DELETE FROM path_contexts WHERE collection IS ? AND path = ?`,
		nullableString(collection), path,
	)
	if err != nil {
		return err
	}
	n, err := res.RowsAffected()
	if err != nil {
		return err
	}
	if n == 0 {
		return fmt.Errorf("%w: collection=%q path=%q", ErrContextNotFound, collection, path)
	}
	return nil
}

// ListContexts returns every registered context. The result merges two
// sources — collection-level descriptions set via `recall collection
// add --context …` and path-level annotations added via `recall context
// add …`. Both are surfaced because the user mental model is "any
// context I've attached anywhere", and the existing path_contexts-only
// listing left collection-level descriptions invisible (validation
// confirmed this confusion). Ordered by collection then path.
func (s *Store) ListContexts() ([]PathContext, error) {
	rows, err := s.db.Query(`
		SELECT id, COALESCE(collection, ''), path, context FROM path_contexts
		UNION ALL
		SELECT -id, name, '/', context FROM collections WHERE context IS NOT NULL AND context != ''
		ORDER BY 2, 3`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var out []PathContext
	for rows.Next() {
		var c PathContext
		if err := rows.Scan(&c.ID, &c.Collection, &c.Path, &c.Context); err != nil {
			return nil, err
		}
		out = append(out, c)
	}
	return out, rows.Err()
}

// GetContext looks up a specific (collection, path) pair. Returns
// sql.ErrNoRows when absent so callers can distinguish from real errors.
func (s *Store) GetContext(collection, path string) (*PathContext, error) {
	if path == "" {
		path = "/"
	}
	var c PathContext
	err := s.db.QueryRow(
		`SELECT id, COALESCE(collection, ''), path, context
		 FROM path_contexts WHERE collection IS ? AND path = ?`,
		nullableString(collection), path,
	).Scan(&c.ID, &c.Collection, &c.Path, &c.Context)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, err
	}
	if err != nil {
		return nil, err
	}
	return &c, nil
}

// CheckContexts returns the names of every collection that does NOT have at
// least one context row attached. It answers the question "what have I
// indexed without bothering to describe?".
func (s *Store) CheckContexts() ([]string, error) {
	rows, err := s.db.Query(`
		SELECT c.name FROM collections c
		WHERE NOT EXISTS (
			SELECT 1 FROM path_contexts pc
			WHERE pc.collection = c.name OR pc.collection IS NULL
		)
		ORDER BY c.name`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var out []string
	for rows.Next() {
		var n string
		if err := rows.Scan(&n); err != nil {
			return nil, err
		}
		out = append(out, n)
	}
	return out, rows.Err()
}
