// Package store owns recall's persistence: SQLite schema, migrations,
// collections, documents, BM25 search, and path contexts.
//
// All data access in recall goes through a *Store. The package never calls
// other recall packages; orchestration lives in pkg/recall or internal/commands.
package store

import (
	"database/sql"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	_ "github.com/mattn/go-sqlite3"
)

// DefaultDBPath is the filesystem location recall stores its index at when
// neither --db nor RECALL_DB_PATH is provided.
const DefaultDBPath = "~/.recall/index.db"

// SchemaVersion is written to the metadata table on first open. Bump when a
// migration changes the schema.
const SchemaVersion = "1"

// Store is a handle to recall's SQLite database. It is safe to use from
// multiple goroutines — sql.DB is internally pooled.
type Store struct {
	db   *sql.DB
	path string

	// Prepared statements for hot-path queries. See Design Decisions
	// ("Performance Optimizations") in ROADMAP.md: FTS5 search queries are
	// structurally identical, so preparing once avoids SQL parsing overhead on
	// every search call.
	stmtSearchBM25       *sql.Stmt
	stmtSearchBM25InColl *sql.Stmt
}

// Open opens (or creates) a recall database at dbPath. An empty dbPath falls
// back to $RECALL_DB_PATH, then to ~/.recall/index.db. The database directory
// is created if it does not exist. Migrations run automatically on every open.
func Open(dbPath string) (*Store, error) {
	resolved, err := ResolveDBPath(dbPath)
	if err != nil {
		return nil, fmt.Errorf("resolve db path: %w", err)
	}

	if err := os.MkdirAll(filepath.Dir(resolved), 0o755); err != nil {
		return nil, fmt.Errorf("create db dir %s: %w", filepath.Dir(resolved), err)
	}

	// Pragmas applied via DSN so every pooled connection inherits them.
	dsn := resolved +
		"?_journal_mode=WAL" +
		"&_busy_timeout=5000" +
		"&_synchronous=NORMAL" +
		"&_cache_size=-64000" +
		"&_foreign_keys=ON"

	db, err := sql.Open("sqlite3", dsn)
	if err != nil {
		return nil, fmt.Errorf("open sqlite at %s: %w", resolved, err)
	}

	// SQLite's default pool behaviour with WAL is fine, but cap it to avoid
	// connection storms on laptops under load.
	db.SetMaxOpenConns(8)

	s := &Store{db: db, path: resolved}

	if err := s.migrate(); err != nil {
		_ = db.Close()
		// Translate the most common opaque sqlite error ("no such module: fts5")
		// into a concrete instruction. Users hit this when they `go install`
		// or `go build` without the build tag — recall's full-text search
		// is gated behind it.
		if strings.Contains(err.Error(), "no such module: fts5") {
			return nil, fmt.Errorf(
				"this build of recall is missing FTS5 support. Rebuild with " +
					"`go build -tags sqlite_fts5 ./cmd/recall` " +
					"(or `make build` from a source checkout). " +
					"Pre-built release binaries already include the tag.")
		}
		return nil, fmt.Errorf("migrate: %w", err)
	}

	if err := s.prepare(); err != nil {
		_ = db.Close()
		return nil, fmt.Errorf("prepare statements: %w", err)
	}

	return s, nil
}

// Path reports the fully resolved filesystem path the store is backed by.
func (s *Store) Path() string { return s.path }

// DB exposes the underlying *sql.DB. It exists for tests and advanced callers;
// most code should go through Store's methods.
func (s *Store) DB() *sql.DB { return s.db }

// Close shuts the database down and releases prepared statements. It is safe
// to call multiple times — the second call returns sql.ErrConnDone-equivalent
// behaviour via the underlying driver.
func (s *Store) Close() error {
	var firstErr error
	for _, st := range []*sql.Stmt{s.stmtSearchBM25, s.stmtSearchBM25InColl} {
		if st == nil {
			continue
		}
		if err := st.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	if err := s.db.Close(); err != nil && firstErr == nil {
		firstErr = err
	}
	return firstErr
}

// ResolveDBPath expands and normalises a database path. Empty input consults
// $RECALL_DB_PATH, then falls back to DefaultDBPath. A leading "~/" is
// expanded against the current user's home directory.
//
// For the --index variant see [ResolveIndexPath] — it lives alongside this
// resolver because the commands layer chooses between them (and warns on
// conflicts) before calling Open.
func ResolveDBPath(explicit string) (string, error) {
	p := explicit
	if p == "" {
		p = os.Getenv("RECALL_DB_PATH")
	}
	if p == "" {
		p = DefaultDBPath
	}
	if strings.HasPrefix(p, "~/") || p == "~" {
		home, err := os.UserHomeDir()
		if err != nil {
			return "", fmt.Errorf("get home dir: %w", err)
		}
		if p == "~" {
			return home, nil
		}
		return filepath.Join(home, p[2:]), nil
	}
	return p, nil
}

// ResolveIndexPath maps a --index name to a concrete file path. qmd's
// equivalent stores everything flat under ~/.cache/qmd/<name>.sqlite;
// recall keeps the default at ~/.recall/index.db and nests named
// indexes one level deeper at ~/.recall/indexes/<name>.db so a user
// can't accidentally shadow the default with `--index index`.
//
// Name sanitisation: alphanumerics, dash, underscore. No path
// separators, no leading dot, no "..". Reserved names ("index" is
// fine — it lands at ~/.recall/indexes/index.db, distinct from the
// default ~/.recall/index.db).
func ResolveIndexPath(name string) (string, error) {
	if name == "" {
		return "", fmt.Errorf("index name is empty")
	}
	if !validIndexName(name) {
		return "", fmt.Errorf("invalid index name %q — use alphanumerics, dashes, and underscores only", name)
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("get home dir: %w", err)
	}
	return filepath.Join(home, ".recall", "indexes", name+".db"), nil
}

// validIndexName returns true when name is made up of ASCII letters,
// digits, dash, or underscore. Keeps index names portable across
// macOS / Linux / Windows filesystems and rules out "../escape".
func validIndexName(name string) bool {
	if name == "" {
		return false
	}
	for _, r := range name {
		switch {
		case r >= 'a' && r <= 'z':
		case r >= 'A' && r <= 'Z':
		case r >= '0' && r <= '9':
		case r == '-', r == '_':
		default:
			return false
		}
	}
	return true
}

// migrate applies the schema. The schema is idempotent (CREATE IF NOT EXISTS)
// so re-running is a no-op on an already-migrated database.
func (s *Store) migrate() error {
	if _, err := s.db.Exec(schemaSQL); err != nil {
		return fmt.Errorf("apply schema: %w", err)
	}
	if _, err := s.db.Exec(
		`INSERT OR IGNORE INTO metadata(key, value) VALUES('schema_version', ?)`,
		SchemaVersion,
	); err != nil {
		return fmt.Errorf("write schema_version: %w", err)
	}
	return nil
}

// prepare caches hot-path statements. Errors here are fatal to Open.
func (s *Store) prepare() error {
	var err error
	s.stmtSearchBM25, err = s.db.Prepare(searchBM25SQL)
	if err != nil {
		return fmt.Errorf("prepare searchBM25: %w", err)
	}
	s.stmtSearchBM25InColl, err = s.db.Prepare(searchBM25InCollectionSQL)
	if err != nil {
		return fmt.Errorf("prepare searchBM25InCollection: %w", err)
	}
	return nil
}

// GetMetadata returns a metadata value by key, or an empty string and false
// when the key is absent.
func (s *Store) GetMetadata(key string) (string, bool, error) {
	var v string
	err := s.db.QueryRow(`SELECT value FROM metadata WHERE key = ?`, key).Scan(&v)
	if errors.Is(err, sql.ErrNoRows) {
		return "", false, nil
	}
	if err != nil {
		return "", false, err
	}
	return v, true, nil
}

// SetMetadata upserts a metadata key/value pair.
func (s *Store) SetMetadata(key, value string) error {
	_, err := s.db.Exec(
		`INSERT INTO metadata(key, value) VALUES(?, ?)
		 ON CONFLICT(key) DO UPDATE SET value = excluded.value`,
		key, value,
	)
	return err
}

// schemaSQL is the full schema. Idempotent — every statement uses IF NOT
// EXISTS so re-running Open on an existing database is a no-op.
const schemaSQL = `
CREATE TABLE IF NOT EXISTS collections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    path TEXT NOT NULL,
    glob_pattern TEXT NOT NULL DEFAULT '**/*.{txt,md}',
    context TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    collection_id INTEGER NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
    path TEXT NOT NULL,
    absolute_path TEXT NOT NULL,
    title TEXT,
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    doc_id TEXT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(collection_id, path)
);

CREATE INDEX IF NOT EXISTS idx_documents_doc_id ON documents(doc_id);
CREATE INDEX IF NOT EXISTS idx_documents_collection ON documents(collection_id);

CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
    title,
    content,
    content=documents,
    content_rowid=id,
    tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS documents_ai AFTER INSERT ON documents BEGIN
    INSERT INTO documents_fts(rowid, title, content) VALUES (new.id, new.title, new.content);
END;
CREATE TRIGGER IF NOT EXISTS documents_ad AFTER DELETE ON documents BEGIN
    INSERT INTO documents_fts(documents_fts, rowid, title, content) VALUES('delete', old.id, old.title, old.content);
END;
CREATE TRIGGER IF NOT EXISTS documents_au AFTER UPDATE ON documents BEGIN
    INSERT INTO documents_fts(documents_fts, rowid, title, content) VALUES('delete', old.id, old.title, old.content);
    INSERT INTO documents_fts(rowid, title, content) VALUES (new.id, new.title, new.content);
END;

CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    seq INTEGER NOT NULL,
    start_pos INTEGER NOT NULL,
    end_pos INTEGER NOT NULL,
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    UNIQUE(document_id, seq)
);

CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_hash ON chunks(content_hash);

CREATE VIRTUAL TABLE IF NOT EXISTS chunk_embeddings USING vec0(
    chunk_id INTEGER PRIMARY KEY,
    embedding float[768]
);

CREATE TABLE IF NOT EXISTS path_contexts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    collection TEXT,
    path TEXT NOT NULL,
    context TEXT NOT NULL,
    UNIQUE(collection, path)
);

CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
`
