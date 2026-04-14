package store

import (
	"strings"
	"testing"
)

func seedSearchCorpus(t *testing.T) *Store {
	t.Helper()
	s := openTestStore(t)

	d1 := tempCollectionDir(t, map[string]string{
		"auth.md":    "# Auth\nThe authentication flow handles JWT tokens.",
		"rate.md":    "# Rate Limiter\nDiscussion of rate limiting algorithms.",
		"random.md":  "# Misc\nSome unrelated content about weather.",
	})
	c1, err := s.AddCollection("notes", d1, "", "")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := s.IndexCollection(c1.ID); err != nil {
		t.Fatal(err)
	}

	d2 := tempCollectionDir(t, map[string]string{
		"api.md": "# API\nWe follow consistent authentication across endpoints.",
	})
	c2, err := s.AddCollection("docs", d2, "", "")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := s.IndexCollection(c2.ID); err != nil {
		t.Fatal(err)
	}

	return s
}

func TestSearchBM25Basic(t *testing.T) {
	s := seedSearchCorpus(t)

	results, err := s.SearchBM25(SearchOptions{Query: "authentication", Limit: 5})
	if err != nil {
		t.Fatalf("SearchBM25: %v", err)
	}
	if len(results) < 2 {
		t.Fatalf("got %d hits, want >= 2", len(results))
	}
	// Scores should be monotonic non-increasing (best first).
	for i := 1; i < len(results); i++ {
		if results[i].Score > results[i-1].Score {
			t.Errorf("results not sorted at %d: %.4f > %.4f",
				i, results[i].Score, results[i-1].Score)
		}
	}
	// Every result should have a non-empty docid and snippet.
	for _, r := range results {
		if r.DocID == "" {
			t.Error("empty docid")
		}
		if r.Snippet == "" {
			t.Errorf("empty snippet for %s", r.Path)
		}
	}
}

func TestSearchBM25NoResults(t *testing.T) {
	s := seedSearchCorpus(t)
	results, err := s.SearchBM25(SearchOptions{Query: "zzqqnonexistent", Limit: 5})
	if err != nil {
		t.Fatalf("SearchBM25: %v", err)
	}
	if len(results) != 0 {
		t.Fatalf("expected 0 results, got %d: %+v", len(results), results)
	}
}

func TestSearchBM25CollectionFilter(t *testing.T) {
	s := seedSearchCorpus(t)

	results, err := s.SearchBM25(SearchOptions{
		Query: "authentication", Collection: "docs", Limit: 10,
	})
	if err != nil {
		t.Fatalf("SearchBM25: %v", err)
	}
	if len(results) == 0 {
		t.Fatal("expected hits in docs collection")
	}
	for _, r := range results {
		if r.CollectionName != "docs" {
			t.Errorf("collection filter leaked: got %s", r.CollectionName)
		}
	}
}

func TestSearchBM25MinScore(t *testing.T) {
	s := seedSearchCorpus(t)

	// First get the top score without filtering.
	top, err := s.SearchBM25(SearchOptions{Query: "authentication", Limit: 1})
	if err != nil || len(top) == 0 {
		t.Fatalf("warmup: %v, len=%d", err, len(top))
	}

	// Now filter above it — expect empty.
	filtered, err := s.SearchBM25(SearchOptions{
		Query: "authentication", Limit: 10, MinScore: top[0].Score + 1000,
	})
	if err != nil {
		t.Fatalf("filtered: %v", err)
	}
	if len(filtered) != 0 {
		t.Fatalf("min-score filter failed, got %d results", len(filtered))
	}
}

func TestSearchBM25Snippet(t *testing.T) {
	s := seedSearchCorpus(t)

	results, err := s.SearchBM25(SearchOptions{Query: "JWT", Limit: 5})
	if err != nil {
		t.Fatal(err)
	}
	if len(results) == 0 {
		t.Fatal("expected at least one hit for JWT")
	}
	// Snippet should contain either the literal term (optionally wrapped in
	// our ANSI markers) or an ellipsis indicating FTS5 took over.
	found := false
	for _, r := range results {
		lower := strings.ToLower(r.Snippet)
		if strings.Contains(lower, "jwt") {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("no snippet contained match token; snippets: %+v", results)
	}
}

func TestSplitCollections(t *testing.T) {
	cases := []struct {
		in   string
		want []string
	}{
		{"", nil},
		{"   ", nil},
		{"notes", []string{"notes"}},
		{"notes,docs", []string{"notes", "docs"}},
		{" notes , docs , api ", []string{"notes", "docs", "api"}},
		{"notes,,docs", []string{"notes", "docs"}},
	}
	for _, c := range cases {
		got := splitCollections(c.in)
		if len(got) != len(c.want) {
			t.Errorf("splitCollections(%q) len = %d, want %d", c.in, len(got), len(c.want))
			continue
		}
		for i := range got {
			if got[i] != c.want[i] {
				t.Errorf("splitCollections(%q)[%d] = %q, want %q", c.in, i, got[i], c.want[i])
			}
		}
	}
}

func TestSearchBM25MultiCollectionFilter(t *testing.T) {
	s := openTestStore(t)

	d1 := tempCollectionDir(t, map[string]string{"a.md": "# a\nauthentication flows"})
	d2 := tempCollectionDir(t, map[string]string{"b.md": "# b\nauthentication tokens"})
	d3 := tempCollectionDir(t, map[string]string{"c.md": "# c\nauthentication users"})
	c1, _ := s.AddCollection("repo1", d1, "", "")
	c2, _ := s.AddCollection("repo2", d2, "", "")
	c3, _ := s.AddCollection("repo3", d3, "", "")
	for _, id := range []int64{c1.ID, c2.ID, c3.ID} {
		if _, err := s.IndexCollection(id); err != nil {
			t.Fatal(err)
		}
	}

	// Filter to only repo1+repo2 — repo3 must be excluded.
	out, err := s.SearchBM25(SearchOptions{
		Query: "authentication", Collection: "repo1,repo2", Limit: 10,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(out) == 0 {
		t.Fatal("expected hits")
	}
	for _, r := range out {
		if r.CollectionName != "repo1" && r.CollectionName != "repo2" {
			t.Errorf("multi-collection filter leaked: got %s", r.CollectionName)
		}
	}
}

func TestSearchBM25EmptyQuery(t *testing.T) {
	s := openTestStore(t)
	if _, err := s.SearchBM25(SearchOptions{Query: "   "}); err == nil {
		t.Error("expected error on empty query")
	}
}

// Natural-language questions often contain characters ("?", ":", "*")
// that FTS5 treats as operators. Raw passthrough produces cryptic
// "fts5: syntax error" failures. SearchBM25 must sanitise the query
// first. Regression guard for the v0.1.2 fix.
func TestSearchBM25SanitizesUserQuestions(t *testing.T) {
	s := seedSearchCorpus(t)

	cases := []string{
		"What did the team decide about authentication?",
		"auth*",
		"what is the rate-limit:",
		"(broken paren",
		"AND",
		"auth OR rate",
		"auth? NOT rate.",
	}
	for _, q := range cases {
		q := q
		t.Run(q, func(t *testing.T) {
			out, err := s.SearchBM25(SearchOptions{Query: q, Limit: 5})
			if err != nil {
				t.Fatalf("SearchBM25(%q) errored: %v", q, err)
			}
			_ = out // presence / absence of hits is corpus-dependent; we're just guarding against syntax errors.
		})
	}
}

func TestSanitizeFTSQuery(t *testing.T) {
	cases := []struct {
		in, want string
	}{
		{"auth", "auth"},
		{"What did the team decide about authentication?", "What did the team decide about authentication"},
		{"auth*", "auth"},
		{"auth OR rate", "auth or rate"},
		{"AND", "and"},
		{"NEAR foo", "near foo"},
		{"  \t\n  ", ""},
		{"hello world!", "hello world"},
		{"rate-limit", "rate limit"},
		{"naïve café", "naïve café"}, // unicode letters preserved
	}
	for _, tc := range cases {
		got := sanitizeFTSQuery(tc.in)
		if got != tc.want {
			t.Errorf("sanitizeFTSQuery(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
}
