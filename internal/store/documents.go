package store

import (
	"bufio"
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"

	"github.com/bmatcuk/doublestar/v4"

	"github.com/ugurcan-aytar/recall/internal/chunk"
)

// Document is a single indexed file.
type Document struct {
	ID             int64
	CollectionID   int64
	CollectionName string // populated by GetDocument; empty after IndexCollection
	Path           string // relative to the collection root
	AbsolutePath   string
	Title          string
	Content        string
	ContentHash    string
	DocID          string // 6-char prefix of ContentHash
}

// IndexStats reports what changed during an IndexCollection call.
type IndexStats struct {
	Indexed   int // newly inserted
	Updated   int // content changed
	Unchanged int // still present, content identical
	Removed   int // file no longer on disk, row deleted
}

// DocIDLength is the number of hex characters used for the short, stable
// document identifier surfaced in search results (e.g. "#abc123").
const DocIDLength = 6

// File-type and language detectors live in [chunk.DetectFileType] and
// [chunk.DetectLanguage] (Phase R2b). They sit in the chunk package so
// the strategy selector there can call them without importing store
// (store already imports chunk).

// ErrDocumentNotFound indicates a Get / MultiGet lookup failed to match.
var ErrDocumentNotFound = errors.New("document not found")

// ComputeContentHash returns the SHA-256 hex digest of raw file content. It
// exists as a public helper so callers can pre-hash and skip IO.
func ComputeContentHash(content []byte) string {
	sum := sha256.Sum256(content)
	return hex.EncodeToString(sum[:])
}

// DocIDFromHash returns the short, stable identifier recall uses to cite
// documents in search output. It is the first DocIDLength characters of the
// content hash.
func DocIDFromHash(contentHash string) string {
	if len(contentHash) < DocIDLength {
		return contentHash
	}
	return contentHash[:DocIDLength]
}

// ExtractTitle returns a display title for a document. The rules are
// per-file-type so search results read naturally:
//
//   - Markdown / text: first H1 heading; otherwise the filename stem.
//   - Code: per-language symbol extraction (e.g. "package auth — Validate"
//     for Go). Falls back to the markdown rule when no symbol is found.
//
// Leading YAML / TOML frontmatter is stripped before any markdown scan.
func ExtractTitle(content, filename string) string {
	if title := extractCodeTitle(content, filename); title != "" {
		return title
	}
	body := stripFrontmatter(content)
	scanner := bufio.NewScanner(strings.NewReader(body))
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if strings.HasPrefix(line, "# ") {
			return strings.TrimSpace(strings.TrimPrefix(line, "# "))
		}
	}
	base := filepath.Base(filename)
	return strings.TrimSuffix(base, filepath.Ext(base))
}

// extractCodeTitle returns a per-language title for code files, or ""
// when the file isn't code or no symbol is found. Pure regex parsing —
// we don't pay a tree-sitter parse for every doc just to label it.
func extractCodeTitle(content, filename string) string {
	switch strings.ToLower(filepath.Ext(filename)) {
	case ".go":
		return extractGoTitle(content)
	case ".py":
		return extractPythonTitle(content)
	case ".ts", ".tsx", ".js", ".jsx":
		return extractTSTitle(content)
	case ".java":
		return extractJavaTitle(content)
	case ".rs":
		return extractRustTitle(content)
	}
	return ""
}

var (
	goPackageRE = regexp.MustCompile(`^\s*package\s+([A-Za-z_][A-Za-z0-9_]*)`)
	goSymbolRE  = regexp.MustCompile(`^\s*func\s+(?:\([^)]*\)\s+)?([A-Z][A-Za-z0-9_]*)\s*\(|^\s*type\s+([A-Z][A-Za-z0-9_]*)`)

	pythonSymbolRE = regexp.MustCompile(`^\s*(?:async\s+)?(?:class|def)\s+([A-Za-z_][A-Za-z0-9_]*)`)

	tsExportRE = regexp.MustCompile(
		`^\s*export\s+(?:default\s+)?` +
			`(?:async\s+)?(?:function|class|interface|type|const|let|var)\s+([A-Za-z_$][A-Za-z0-9_$]*)`,
	)
	tsTopLevelRE = regexp.MustCompile(
		`^\s*(?:async\s+)?(?:function|class|interface)\s+([A-Za-z_$][A-Za-z0-9_$]*)`,
	)

	javaClassRE = regexp.MustCompile(`(?m)^\s*public\s+(?:final\s+|abstract\s+)?(?:class|interface|enum)\s+([A-Za-z_][A-Za-z0-9_]*)`)

	rustPubRE = regexp.MustCompile(`^\s*pub\s+(?:fn|struct|enum|trait|mod)\s+([A-Za-z_][A-Za-z0-9_]*)`)
)

func extractGoTitle(content string) string {
	pkg := ""
	sym := ""
	scanner := bufio.NewScanner(strings.NewReader(content))
	for scanner.Scan() {
		line := scanner.Text()
		if pkg == "" {
			if m := goPackageRE.FindStringSubmatch(line); m != nil {
				pkg = m[1]
			}
		}
		if sym == "" {
			if m := goSymbolRE.FindStringSubmatch(line); m != nil {
				if m[1] != "" {
					sym = m[1]
				} else {
					sym = m[2]
				}
			}
		}
		if pkg != "" && sym != "" {
			break
		}
	}
	switch {
	case pkg != "" && sym != "":
		return "package " + pkg + " — " + sym
	case pkg != "":
		return "package " + pkg
	case sym != "":
		return sym
	}
	return ""
}

func extractPythonTitle(content string) string {
	scanner := bufio.NewScanner(strings.NewReader(content))
	for scanner.Scan() {
		if m := pythonSymbolRE.FindStringSubmatch(scanner.Text()); m != nil {
			return m[1]
		}
	}
	return ""
}

func extractTSTitle(content string) string {
	scanner := bufio.NewScanner(strings.NewReader(content))
	var fallback string
	for scanner.Scan() {
		line := scanner.Text()
		if m := tsExportRE.FindStringSubmatch(line); m != nil {
			return m[1]
		}
		if fallback == "" {
			if m := tsTopLevelRE.FindStringSubmatch(line); m != nil {
				fallback = m[1]
			}
		}
	}
	return fallback
}

func extractJavaTitle(content string) string {
	if m := javaClassRE.FindStringSubmatch(content); m != nil {
		return m[1]
	}
	return ""
}

func extractRustTitle(content string) string {
	scanner := bufio.NewScanner(strings.NewReader(content))
	for scanner.Scan() {
		if m := rustPubRE.FindStringSubmatch(scanner.Text()); m != nil {
			return m[1]
		}
	}
	return ""
}

// IndexCollection re-scans a collection's directory and reconciles the index
// against what is on disk. It is idempotent: files that haven't changed are
// left alone.
func (s *Store) IndexCollection(collectionID int64) (IndexStats, error) {
	var stats IndexStats

	c, err := s.GetCollectionByID(collectionID)
	if err != nil {
		return stats, err
	}

	info, err := os.Stat(c.Path)
	if err != nil {
		return stats, fmt.Errorf("stat collection root %s: %w", c.Path, err)
	}
	if !info.IsDir() {
		return stats, fmt.Errorf("collection root %s is not a directory", c.Path)
	}

	seen := map[string]struct{}{}

	err = doublestar.GlobWalk(
		os.DirFS(c.Path),
		c.GlobPattern,
		func(rel string, d fs.DirEntry) error {
			if d.IsDir() {
				return nil
			}
			abs := filepath.Join(c.Path, rel)
			content, err := os.ReadFile(abs)
			if err != nil {
				// Skip unreadable files rather than aborting the whole index.
				fmt.Fprintf(os.Stderr, "warning: skip %s: %v\n", abs, err)
				return nil
			}
			seen[rel] = struct{}{}

			hash := ComputeContentHash(content)
			docID := DocIDFromHash(hash)
			title := ExtractTitle(string(content), rel)

			existingHash, existed, err := s.getDocumentHash(collectionID, rel)
			if err != nil {
				return fmt.Errorf("lookup existing doc %s: %w", rel, err)
			}

			switch {
			case !existed:
				res, err := s.db.Exec(
					`INSERT INTO documents(collection_id, path, absolute_path, title,
					                        content, content_hash, doc_id)
					 VALUES (?, ?, ?, ?, ?, ?, ?)`,
					collectionID, rel, abs, title, string(content), hash, docID,
				)
				if err != nil {
					return fmt.Errorf("insert %s: %w", rel, err)
				}
				docDBID, err := res.LastInsertId()
				if err != nil {
					return fmt.Errorf("last insert id %s: %w", rel, err)
				}
				if err := s.ReplaceChunks(docDBID, chunk.ChunkFile(string(content), rel, chunk.StrategyAuto, 0, 0)); err != nil {
					return fmt.Errorf("chunk %s: %w", rel, err)
				}
				stats.Indexed++
			case existingHash == hash:
				stats.Unchanged++
			default:
				if _, err := s.db.Exec(
					`UPDATE documents
					 SET absolute_path = ?, title = ?, content = ?, content_hash = ?,
					     doc_id = ?, updated_at = CURRENT_TIMESTAMP
					 WHERE collection_id = ? AND path = ?`,
					abs, title, string(content), hash, docID, collectionID, rel,
				); err != nil {
					return fmt.Errorf("update %s: %w", rel, err)
				}
				var docDBID int64
				if err := s.db.QueryRow(
					`SELECT id FROM documents WHERE collection_id = ? AND path = ?`,
					collectionID, rel,
				).Scan(&docDBID); err != nil {
					return fmt.Errorf("lookup doc id after update %s: %w", rel, err)
				}
				if err := s.ReplaceChunks(docDBID, chunk.ChunkFile(string(content), rel, chunk.StrategyAuto, 0, 0)); err != nil {
					return fmt.Errorf("chunk %s: %w", rel, err)
				}
				stats.Updated++
			}
			return nil
		},
		doublestar.WithFilesOnly(),
	)
	if err != nil {
		return stats, fmt.Errorf("walk collection %s: %w", c.Name, err)
	}

	removed, err := s.deleteMissing(collectionID, seen)
	if err != nil {
		return stats, err
	}
	stats.Removed = removed

	return stats, nil
}

func (s *Store) getDocumentHash(collectionID int64, path string) (string, bool, error) {
	var hash string
	err := s.db.QueryRow(
		`SELECT content_hash FROM documents WHERE collection_id = ? AND path = ?`,
		collectionID, path,
	).Scan(&hash)
	if errors.Is(err, sql.ErrNoRows) {
		return "", false, nil
	}
	if err != nil {
		return "", false, err
	}
	return hash, true, nil
}

func (s *Store) deleteMissing(collectionID int64, seen map[string]struct{}) (int, error) {
	rows, err := s.db.Query(
		`SELECT path FROM documents WHERE collection_id = ?`, collectionID,
	)
	if err != nil {
		return 0, fmt.Errorf("list docs for deletion: %w", err)
	}
	var gone []string
	for rows.Next() {
		var p string
		if err := rows.Scan(&p); err != nil {
			rows.Close()
			return 0, err
		}
		if _, ok := seen[p]; !ok {
			gone = append(gone, p)
		}
	}
	rows.Close()
	if err := rows.Err(); err != nil {
		return 0, err
	}

	for _, p := range gone {
		if _, err := s.db.Exec(
			`DELETE FROM documents WHERE collection_id = ? AND path = ?`,
			collectionID, p,
		); err != nil {
			return 0, fmt.Errorf("delete %s: %w", p, err)
		}
	}
	return len(gone), nil
}

// GetDocument resolves either a relative path (optionally prefixed with a
// collection name, e.g. "notes/meeting.md") or a "#docid" citation.
// Returns ErrDocumentNotFound when nothing matches.
func (s *Store) GetDocument(spec string) (*Document, error) {
	spec = strings.TrimSpace(spec)
	if spec == "" {
		return nil, errors.New("empty document spec")
	}

	if strings.HasPrefix(spec, "#") {
		return s.getByDocID(strings.TrimPrefix(spec, "#"))
	}
	return s.getByPath(spec)
}

func (s *Store) getByDocID(docID string) (*Document, error) {
	row := s.db.QueryRow(`
		SELECT d.id, d.collection_id, c.name, d.path, d.absolute_path, d.title,
		       d.content, d.content_hash, d.doc_id
		FROM documents d
		JOIN collections c ON c.id = d.collection_id
		WHERE d.doc_id = ?
		LIMIT 1`, docID)
	return scanDocument(row, docID)
}

func (s *Store) getByPath(spec string) (*Document, error) {
	// Accept "collection/rel/path" or bare "rel/path".
	parts := strings.SplitN(spec, "/", 2)
	if len(parts) == 2 {
		row := s.db.QueryRow(`
			SELECT d.id, d.collection_id, c.name, d.path, d.absolute_path, d.title,
			       d.content, d.content_hash, d.doc_id
			FROM documents d
			JOIN collections c ON c.id = d.collection_id
			WHERE c.name = ? AND d.path = ?
			LIMIT 1`, parts[0], parts[1])
		if doc, err := scanDocument(row, spec); err == nil {
			return doc, nil
		} else if !errors.Is(err, ErrDocumentNotFound) {
			return nil, err
		}
	}
	// Fall back: try matching the whole spec as a path in any collection.
	row := s.db.QueryRow(`
		SELECT d.id, d.collection_id, c.name, d.path, d.absolute_path, d.title,
		       d.content, d.content_hash, d.doc_id
		FROM documents d
		JOIN collections c ON c.id = d.collection_id
		WHERE d.path = ?
		LIMIT 1`, spec)
	return scanDocument(row, spec)
}

func scanDocument(row *sql.Row, spec string) (*Document, error) {
	var d Document
	err := row.Scan(
		&d.ID, &d.CollectionID, &d.CollectionName, &d.Path, &d.AbsolutePath,
		&d.Title, &d.Content, &d.ContentHash, &d.DocID,
	)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, fmt.Errorf("%w: %s", ErrDocumentNotFound, spec)
	}
	if err != nil {
		return nil, err
	}
	return &d, nil
}

// ListDocumentPaths returns the relative paths of every document in a
// collection, sorted. It powers `recall ls`.
func (s *Store) ListDocumentPaths(collectionName, subPath string) ([]string, error) {
	c, err := s.GetCollectionByName(collectionName)
	if err != nil {
		return nil, err
	}
	rows, err := s.db.Query(
		`SELECT path FROM documents WHERE collection_id = ? ORDER BY path`,
		c.ID,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var paths []string
	for rows.Next() {
		var p string
		if err := rows.Scan(&p); err != nil {
			return nil, err
		}
		if subPath == "" || strings.HasPrefix(p, subPath) {
			paths = append(paths, p)
		}
	}
	sort.Strings(paths)
	return paths, rows.Err()
}

// DocumentCount returns the number of documents in a collection. Convenient
// for status reporting.
func (s *Store) DocumentCount(collectionID int64) (int, error) {
	var n int
	err := s.db.QueryRow(
		`SELECT COUNT(*) FROM documents WHERE collection_id = ?`, collectionID,
	).Scan(&n)
	return n, err
}

// TotalDocumentCount returns the total number of documents across all
// collections.
func (s *Store) TotalDocumentCount() (int, error) {
	var n int
	err := s.db.QueryRow(`SELECT COUNT(*) FROM documents`).Scan(&n)
	return n, err
}

// AllDocuments returns every document in the database with its full
// content. Used by tools that need to walk the corpus (e.g. re-chunking
// after a strategy switch).
func (s *Store) AllDocuments() ([]Document, error) {
	rows, err := s.db.Query(`
		SELECT d.id, d.collection_id, c.name, d.path, d.absolute_path, d.title,
		       d.content, d.content_hash, d.doc_id
		FROM documents d
		JOIN collections c ON c.id = d.collection_id
		ORDER BY d.id`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []Document
	for rows.Next() {
		var d Document
		if err := rows.Scan(
			&d.ID, &d.CollectionID, &d.CollectionName, &d.Path, &d.AbsolutePath,
			&d.Title, &d.Content, &d.ContentHash, &d.DocID,
		); err != nil {
			return nil, err
		}
		out = append(out, d)
	}
	return out, rows.Err()
}

// MultiGetGlob returns every document whose path matches the given glob
// pattern. The pattern is matched against `collection_name/relative_path`,
// so `notes/*.md` and `*.md` both work.
func (s *Store) MultiGetGlob(pattern string) ([]Document, error) {
	rows, err := s.db.Query(`
		SELECT d.id, d.collection_id, c.name, d.path, d.absolute_path, d.title,
		       d.content, d.content_hash, d.doc_id
		FROM documents d
		JOIN collections c ON c.id = d.collection_id`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var out []Document
	for rows.Next() {
		var d Document
		if err := rows.Scan(
			&d.ID, &d.CollectionID, &d.CollectionName, &d.Path, &d.AbsolutePath,
			&d.Title, &d.Content, &d.ContentHash, &d.DocID,
		); err != nil {
			return nil, err
		}
		key := d.CollectionName + "/" + d.Path
		if ok, _ := doublestar.Match(pattern, key); ok {
			out = append(out, d)
			continue
		}
		if ok, _ := doublestar.Match(pattern, d.Path); ok {
			out = append(out, d)
		}
	}
	return out, rows.Err()
}

// stripFrontmatter removes leading YAML (--- … ---) or TOML (+++ … +++)
// frontmatter from a string, returning the body. When no frontmatter is
// present the input is returned unchanged.
func stripFrontmatter(s string) string {
	trimmed := strings.TrimLeft(s, " \t\r\n")
	for _, delim := range []string{"---", "+++"} {
		if !strings.HasPrefix(trimmed, delim+"\n") && !strings.HasPrefix(trimmed, delim+"\r\n") {
			continue
		}
		rest := trimmed[len(delim):]
		rest = strings.TrimLeft(rest, "\r\n")
		end := strings.Index(rest, "\n"+delim)
		if end < 0 {
			return s
		}
		body := rest[end+len("\n"+delim):]
		return strings.TrimLeft(body, "\r\n")
	}
	return s
}
