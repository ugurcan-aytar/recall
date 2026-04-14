package store

import (
	"database/sql"
	"fmt"
	"strings"
	"unicode"
)

// sanitizeFTSQuery rewrites a user-typed query into an FTS5-safe
// MATCH expression. Raw user input has two classes of problem:
//
//  1. Punctuation FTS5 reads as operators ("?", ":", "*", apostrophes,
//     parens, uppercase AND / OR / NOT / NEAR) — passing through
//     verbatim produces cryptic "fts5: syntax error" failures.
//  2. Natural-language noise ("what is", "how can I", "the", "a")
//     that FTS5's implicit-AND then demands every doc must contain.
//     A chat-style question like "what's the circuit breaker pattern"
//     fans out to seven required terms — no single chunk contains all
//     seven, so retrieval returns zero hits even though the corpus
//     has an obvious match.
//
// The sanitiser handles (1) by replacing every non-word rune with a
// space (FTS5 reads space-separated barewords as implicit AND) and
// lowercasing surviving AND/OR/NOT/NEAR. It handles (2) by dropping
// common stopwords in English + Turkish (the two languages recall
// actively supports) once the query has more than three tokens —
// short queries are treated as precise keyword searches and pass
// through untouched.
//
// If stopword filtering would leave the query empty (e.g. the user
// literally typed "the"), we fall back to the unfiltered token list
// so BM25 still runs rather than returning an "empty query" error.
// Unicode letters / digits pass through unchanged so non-ASCII
// corpora keep working.
func sanitizeFTSQuery(q string) string {
	var b strings.Builder
	b.Grow(len(q))
	for _, r := range q {
		switch {
		case unicode.IsLetter(r), unicode.IsDigit(r), r == '_':
			b.WriteRune(r)
		default:
			b.WriteByte(' ')
		}
	}
	fields := strings.Fields(b.String())
	for i, f := range fields {
		switch strings.ToUpper(f) {
		case "AND", "OR", "NOT", "NEAR":
			fields[i] = strings.ToLower(f)
		}
	}
	// Only filter stopwords on longer queries. "rate limit" and
	// "auth flow" are precise 2-token searches; we don't want the
	// sanitiser second-guessing those. Natural-language questions
	// start to benefit at around 4+ tokens.
	if len(fields) <= 3 {
		return strings.Join(fields, " ")
	}
	filtered := fields[:0:len(fields)]
	for _, f := range fields {
		if _, isStop := ftsStopwords[strings.ToLower(f)]; isStop {
			continue
		}
		filtered = append(filtered, f)
	}
	if len(filtered) == 0 {
		return strings.Join(fields, " ")
	}
	return strings.Join(filtered, " ")
}

// ftsStopwords is the union of English + Turkish fillers we drop from
// 4-plus-token queries before handing them to FTS5. Keeps the list
// tight — content words (nouns, verbs, adjectives) and
// domain-specific jargon stay. Extending this set is a semver-patch
// event; removing an entry is semver-minor because it changes recall
// behaviour on existing queries.
var ftsStopwords = map[string]struct{}{
	// English fillers.
	"a": {}, "an": {}, "and": {}, "are": {}, "as": {}, "at": {}, "be": {}, "but": {},
	"by": {}, "can": {}, "could": {}, "did": {}, "do": {}, "does": {}, "for": {},
	"from": {}, "had": {}, "has": {}, "have": {}, "he": {}, "her": {}, "here": {},
	"his": {}, "how": {}, "i": {}, "if": {}, "in": {}, "into": {}, "is": {}, "it": {},
	"its": {}, "me": {}, "my": {}, "not": {}, "now": {}, "of": {}, "on": {}, "or": {},
	"our": {}, "out": {}, "she": {}, "so": {}, "than": {}, "that": {},
	"the": {}, "their": {}, "them": {}, "then": {}, "there": {}, "these": {},
	"they": {}, "this": {}, "those": {}, "to": {}, "too": {}, "up": {}, "us": {},
	"was": {}, "we": {}, "were": {}, "what": {}, "when": {}, "where": {}, "which": {},
	"while": {}, "who": {}, "why": {}, "will": {}, "with": {}, "would": {}, "you": {},
	"your": {},
	// Contraction leftovers after the non-word strip — "don't" becomes
	// "don t", "what's" becomes "what s", "we'll" becomes "we ll", etc.
	// These roots and enclitics are almost never genuine search terms
	// (a programmer grepping for "don" is vanishingly rare vs "don't"
	// noise), so we drop them.
	"ain": {}, "aren": {}, "couldn": {}, "didn": {}, "doesn": {}, "don": {},
	"hadn": {}, "hasn": {}, "haven": {}, "isn": {}, "shouldn": {}, "wasn": {},
	"weren": {}, "won": {}, "wouldn": {},
	"d": {}, "ll": {}, "m": {}, "re": {}, "s": {}, "t": {}, // "ve" handled in the Turkish section below (same literal covers both languages).
	// Turkish fillers. Kept minimal; brain's classifier already lowercases
	// queries before reaching here, so case folding isn't needed.
	"bir": {}, "bu": {}, "çok": {}, "da": {}, "daha": {}, "de": {}, "en": {},
	"için": {}, "ile": {}, "ki": {}, "kim": {}, "mi": {}, "mı": {}, "mu": {},
	"mü": {}, "ne": {}, "ni": {}, "nin": {}, "o": {}, "şu": {}, "ve": {},
	"veya": {}, "ya": {},
}

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
	q := sanitizeFTSQuery(opts.Query)
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
