package store

import (
	"database/sql"
	"fmt"
	"strings"
	"unicode"
)

// sanitizeFTSQuery rewrites a user-typed query into an FTS5-safe
// MATCH expression. The sanitiser has three jobs:
//
//  1. Translate qmd-style user syntax into FTS5 syntax:
//
//       "exact phrase"         → FTS5 "phrase"    (quotes preserved)
//       -term                  → FTS5 NOT term    (positive … NOT term)
//       -"exact phrase"        → FTS5 NOT "phrase"
//       word1 word2            → FTS5 word1 word2 (implicit AND)
//
//  2. Strip punctuation FTS5 reads as operators ("?", ":", "*",
//     apostrophes, parens, stray dashes) from barewords — without
//     this, natural-language input produces cryptic "fts5: syntax
//     error" failures.
//
//  3. Drop common English + Turkish stopwords from un-quoted,
//     un-negated barewords once the query has more than three
//     effective terms. Quoted phrases and negated tokens are
//     preserved verbatim — when a user explicitly quotes or excludes
//     something, they meant it.
//
// If filtering would leave no positive terms (e.g. the user typed
// only "the" or only a bare "-redis"), the sanitiser returns an
// empty string so the caller can produce an "empty search query"
// error rather than handing FTS5 something meaningless.
func sanitizeFTSQuery(q string) string {
	tokens := parseQueryTokens(q)
	if len(tokens) == 0 {
		return ""
	}

	// Apply stopword filtering only to un-negated, un-quoted barewords
	// once we have more than three effective terms. Phrases and
	// negations bypass the filter — the user asked for them
	// explicitly.
	bareCount := 0
	for _, t := range tokens {
		if !t.phrase && !t.negated {
			bareCount++
		}
	}
	filterStopwords := bareCount > 3

	var positives, negatives []queryToken
	for _, t := range tokens {
		if filterStopwords && !t.phrase && !t.negated {
			if _, isStop := ftsStopwords[strings.ToLower(t.text)]; isStop {
				continue
			}
		}
		if t.negated {
			negatives = append(negatives, t)
		} else {
			positives = append(positives, t)
		}
	}

	// FTS5's NOT is binary: "A NOT B". An all-negations query
	// ("-redis") or a query where stopword filtering stripped every
	// positive term leaves nothing for NOT to bind to. Return empty
	// so SearchBM25 raises "empty search query".
	if len(positives) == 0 {
		return ""
	}

	var b strings.Builder
	for i, t := range positives {
		if i > 0 {
			b.WriteByte(' ')
		}
		b.WriteString(t.ftsForm())
	}
	for _, t := range negatives {
		b.WriteString(" NOT ")
		b.WriteString(t.ftsForm())
	}
	return b.String()
}

// queryToken is one unit out of parseQueryTokens. text holds the
// cleaned content (no quotes, no leading '-'). phrase means the user
// wrapped it in double quotes. negated means the user prefixed it
// with '-'.
type queryToken struct {
	text    string
	phrase  bool
	negated bool
}

// ftsForm renders a token into its FTS5 literal — phrases keep their
// quotes and lowercase any operator keywords inside, barewords drop
// through strings.Fields cleanup.
func (t queryToken) ftsForm() string {
	if t.phrase {
		return `"` + t.text + `"`
	}
	return t.text
}

// parseQueryTokens walks the raw query and splits it into
// queryTokens. It honours simple shell-style quoting ("…") and a
// single leading '-' on each un-quoted word or on a quoted phrase
// (-"…"). Mid-word dashes ("rate-limit") are NOT treated as
// negation — they just split into two barewords, matching what
// FTS5 would see anyway.
func parseQueryTokens(q string) []queryToken {
	var out []queryToken
	i := 0
	for i < len(q) {
		// Skip leading whitespace.
		for i < len(q) && isSpace(q[i]) {
			i++
		}
		if i >= len(q) {
			break
		}

		negated := false
		// Leading '-' becomes a negation prefix. Require a non-space
		// non-dash to follow (so stray "-" or "--" in the input don't
		// create empty tokens).
		if q[i] == '-' && i+1 < len(q) && !isSpace(q[i+1]) && q[i+1] != '-' {
			negated = true
			i++
		}

		if i < len(q) && q[i] == '"' {
			// Consume a quoted phrase. Missing close-quote = phrase
			// runs to end of input.
			i++
			start := i
			for i < len(q) && q[i] != '"' {
				i++
			}
			phraseText := sanitizePhraseBody(q[start:i])
			if i < len(q) {
				i++ // consume closing '"'
			}
			if phraseText != "" {
				out = append(out, queryToken{text: phraseText, phrase: true, negated: negated})
			}
			continue
		}

		// Unquoted bareword — consume until whitespace or a quote.
		start := i
		for i < len(q) && !isSpace(q[i]) && q[i] != '"' {
			i++
		}
		raw := q[start:i]
		cleaned := sanitizeBareword(raw)
		if cleaned == "" {
			continue
		}
		// If the token splits on internal punctuation (e.g.
		// "rate-limit" → "rate limit"), emit each piece as its own
		// token carrying the same negation flag. This matches what
		// FTS5 would do with a raw bareword anyway.
		for _, piece := range strings.Fields(cleaned) {
			lower := strings.ToLower(piece)
			switch lower {
			case "and", "or", "not", "near":
				// Operator keywords — lowercase so FTS5 sees them as
				// literal terms, not control syntax. (Users never
				// intend FTS5 AND/OR semantics; recall's contract is
				// implicit-AND with phrase + negation extensions.)
				piece = lower
			}
			out = append(out, queryToken{text: piece, negated: negated})
		}
	}
	return out
}

// sanitizeBareword strips characters FTS5 would treat as operators
// ("?", ":", "*", parens, apostrophes, etc.) from a bareword. Unicode
// letters, digits, and underscores survive. Internal dashes become
// spaces so "rate-limit" → "rate limit".
func sanitizeBareword(s string) string {
	var b strings.Builder
	b.Grow(len(s))
	for _, r := range s {
		switch {
		case unicode.IsLetter(r), unicode.IsDigit(r), r == '_':
			b.WriteRune(r)
		default:
			b.WriteByte(' ')
		}
	}
	return b.String()
}

// sanitizePhraseBody cleans the content of a quoted phrase. FTS5's
// phrase syntax is strict: the only meaningful characters inside
// quotes are tokenised by the unicode61 tokenizer, and embedded
// double quotes need doubling ("" for a literal quote). We
// conservatively strip anything that isn't a word character, then
// collapse whitespace.
func sanitizePhraseBody(s string) string {
	var b strings.Builder
	b.Grow(len(s))
	for _, r := range s {
		switch {
		case unicode.IsLetter(r), unicode.IsDigit(r), r == '_', r == ' ':
			b.WriteRune(r)
		default:
			b.WriteByte(' ')
		}
	}
	return strings.TrimSpace(strings.Join(strings.Fields(b.String()), " "))
}

func isSpace(b byte) bool {
	return b == ' ' || b == '\t' || b == '\n' || b == '\r'
}

// ftsStopwords is the union of English + Turkish fillers we drop from
// 4-plus-token queries before handing them to FTS5. Keeps the list
// tight — content words (nouns, verbs, adjectives) and
// domain-specific jargon stay. Extending this set is a semver-patch
// event; removing an entry is semver-minor because it changes recall
// behaviour on existing queries.
var ftsStopwords = map[string]struct{}{
	// English fillers — articles, prepositions, auxiliaries, common
	// question words, and the most frequent query-time fillers. Content
	// words (nouns, verbs, adjectives) stay.
	"a": {}, "about": {}, "after": {}, "all": {}, "an": {}, "and": {}, "any": {},
	"are": {}, "as": {}, "at": {}, "be": {}, "been": {}, "before": {}, "being": {},
	"both": {}, "but": {}, "by": {}, "can": {}, "could": {}, "did": {}, "do": {},
	"does": {}, "during": {}, "each": {}, "for": {}, "from": {}, "had": {}, "has": {},
	"have": {}, "having": {}, "he": {}, "her": {}, "here": {}, "his": {}, "how": {},
	"i": {}, "if": {}, "in": {}, "into": {}, "is": {}, "it": {}, "its": {}, "just": {},
	"me": {}, "more": {}, "most": {}, "my": {}, "no": {}, "nor": {}, "not": {},
	"now": {}, "of": {}, "off": {}, "on": {}, "once": {}, "only": {}, "or": {},
	"other": {}, "our": {}, "out": {}, "over": {}, "own": {}, "same": {}, "she": {},
	"should": {}, "so": {}, "some": {}, "such": {}, "than": {}, "that": {}, "the": {},
	"their": {}, "them": {}, "then": {}, "there": {}, "these": {}, "they": {},
	"this": {}, "those": {}, "through": {}, "to": {}, "too": {}, "under": {},
	"until": {}, "up": {}, "us": {}, "very": {}, "was": {}, "we": {}, "were": {},
	"what": {}, "when": {}, "where": {}, "which": {}, "while": {}, "who": {},
	"why": {}, "will": {}, "with": {}, "would": {}, "you": {}, "your": {},
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
