package store

import (
	"database/sql"
	"errors"
	"fmt"
	"path/filepath"
	"strings"
	"time"
)

// DefaultGlobPattern is recall's out-of-the-box file mask. As of Phase R2b
// it covers markdown / plain text, mainstream programming languages (with
// AST chunking via tree-sitter for Go / Python / TS / JS / Java / Rust),
// and common configuration formats (chunked as markdown; usually small
// enough to fit one chunk). Users can still narrow the set with --mask.
const DefaultGlobPattern = "**/*.{txt,md,go,ts,tsx,js,jsx,py,java,rs,rb,php,c,cpp,h,hpp,cs,swift,kt,scala,sql,sh,bash,yaml,yml,toml,json,xml,html,css,scss,proto,graphql,tf,hcl,Dockerfile}"

// Collection is a user-registered folder that recall indexes.
type Collection struct {
	ID          int64
	Name        string
	Path        string
	GlobPattern string
	Context     string
	CreatedAt   time.Time
	UpdatedAt   time.Time

	// Populated by ListCollections; zero when absent.
	DocCount      int
	LastIndexedAt time.Time
}

// ErrCollectionExists is returned when adding a collection whose name is
// already registered.
var ErrCollectionExists = errors.New("collection already exists")

// ErrCollectionNotFound is returned for lookups and mutations against an
// unknown collection name.
var ErrCollectionNotFound = errors.New("collection not found")

// AddCollection registers a folder. name defaults to the folder's basename
// when empty, glob defaults to DefaultGlobPattern. context may be empty.
func (s *Store) AddCollection(name, path, glob, context string) (*Collection, error) {
	abs, err := filepath.Abs(path)
	if err != nil {
		return nil, fmt.Errorf("resolve absolute path for %s: %w", path, err)
	}

	if name == "" {
		name = filepath.Base(abs)
	}
	if glob == "" {
		glob = DefaultGlobPattern
	}

	res, err := s.db.Exec(
		`INSERT INTO collections(name, path, glob_pattern, context) VALUES (?, ?, ?, ?)`,
		name, abs, glob, nullableString(context),
	)
	if err != nil {
		if isUniqueViolation(err) {
			return nil, fmt.Errorf("%w: %s", ErrCollectionExists, name)
		}
		return nil, fmt.Errorf("insert collection: %w", err)
	}

	id, err := res.LastInsertId()
	if err != nil {
		return nil, fmt.Errorf("last insert id: %w", err)
	}

	return s.GetCollectionByID(id)
}

// RemoveCollection deletes a collection and (via ON DELETE CASCADE) all its
// documents and chunks.
func (s *Store) RemoveCollection(name string) error {
	res, err := s.db.Exec(`DELETE FROM collections WHERE name = ?`, name)
	if err != nil {
		return fmt.Errorf("delete collection %s: %w", name, err)
	}
	n, err := res.RowsAffected()
	if err != nil {
		return err
	}
	if n == 0 {
		return fmt.Errorf("%w: %s", ErrCollectionNotFound, name)
	}
	return nil
}

// RenameCollection changes a collection's display name. The underlying path
// is unaffected.
func (s *Store) RenameCollection(oldName, newName string) error {
	if strings.TrimSpace(newName) == "" {
		return errors.New("new name cannot be empty")
	}
	res, err := s.db.Exec(
		`UPDATE collections SET name = ?, updated_at = CURRENT_TIMESTAMP WHERE name = ?`,
		newName, oldName,
	)
	if err != nil {
		if isUniqueViolation(err) {
			return fmt.Errorf("%w: %s", ErrCollectionExists, newName)
		}
		return fmt.Errorf("rename collection: %w", err)
	}
	n, err := res.RowsAffected()
	if err != nil {
		return err
	}
	if n == 0 {
		return fmt.Errorf("%w: %s", ErrCollectionNotFound, oldName)
	}
	return nil
}

// ListCollections returns every registered collection, along with document
// counts and the most recent indexed-at time. Ordered by name.
func (s *Store) ListCollections() ([]Collection, error) {
	rows, err := s.db.Query(`
		SELECT c.id, c.name, c.path, c.glob_pattern, COALESCE(c.context, ''),
		       c.created_at, c.updated_at,
		       COUNT(d.id) AS doc_count,
		       COALESCE(MAX(d.updated_at), '') AS last_indexed
		FROM collections c
		LEFT JOIN documents d ON d.collection_id = c.id
		GROUP BY c.id
		ORDER BY c.name ASC
	`)
	if err != nil {
		return nil, fmt.Errorf("query collections: %w", err)
	}
	defer rows.Close()

	var out []Collection
	for rows.Next() {
		var c Collection
		var lastIndexedStr string
		if err := rows.Scan(
			&c.ID, &c.Name, &c.Path, &c.GlobPattern, &c.Context,
			&c.CreatedAt, &c.UpdatedAt,
			&c.DocCount, &lastIndexedStr,
		); err != nil {
			return nil, fmt.Errorf("scan collection row: %w", err)
		}
		if lastIndexedStr != "" {
			if t, err := parseSQLiteTime(lastIndexedStr); err == nil {
				c.LastIndexedAt = t
			}
		}
		out = append(out, c)
	}
	return out, rows.Err()
}

// GetCollectionByName looks up a single collection. Returns
// ErrCollectionNotFound when the name is unknown.
func (s *Store) GetCollectionByName(name string) (*Collection, error) {
	return s.getCollectionBy("name", name)
}

// GetCollectionByID fetches a collection by primary key.
func (s *Store) GetCollectionByID(id int64) (*Collection, error) {
	return s.getCollectionBy("id", id)
}

func (s *Store) getCollectionBy(column string, value any) (*Collection, error) {
	query := fmt.Sprintf(`
		SELECT id, name, path, glob_pattern, COALESCE(context, ''),
		       created_at, updated_at
		FROM collections WHERE %s = ?`, column)
	var c Collection
	err := s.db.QueryRow(query, value).Scan(
		&c.ID, &c.Name, &c.Path, &c.GlobPattern, &c.Context,
		&c.CreatedAt, &c.UpdatedAt,
	)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, fmt.Errorf("%w: %v", ErrCollectionNotFound, value)
	}
	if err != nil {
		return nil, err
	}
	return &c, nil
}

func nullableString(s string) any {
	if s == "" {
		return nil
	}
	return s
}

func isUniqueViolation(err error) bool {
	if err == nil {
		return false
	}
	// mattn/go-sqlite3 surfaces "UNIQUE constraint failed" in the message.
	return strings.Contains(err.Error(), "UNIQUE constraint failed")
}

func parseSQLiteTime(s string) (time.Time, error) {
	// SQLite default CURRENT_TIMESTAMP format: "2006-01-02 15:04:05"
	layouts := []string{
		"2006-01-02 15:04:05",
		"2006-01-02T15:04:05Z",
		time.RFC3339,
	}
	for _, l := range layouts {
		if t, err := time.Parse(l, s); err == nil {
			return t, nil
		}
	}
	return time.Time{}, fmt.Errorf("unrecognised time format: %q", s)
}
