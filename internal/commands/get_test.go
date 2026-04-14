package commands

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestParseGetSpec(t *testing.T) {
	cases := []struct {
		in    string
		spec  string
		line  int
	}{
		{"a.md", "a.md", 0},
		{"a.md:42", "a.md", 42},
		{"notes/a.md:10", "notes/a.md", 10},
		{"#abc123", "#abc123", 0},
		{"a.md:oops", "a.md:oops", 0}, // non-numeric after colon is left intact
	}
	for _, c := range cases {
		spec, line := parseGetSpec(c.in)
		if spec != c.spec || line != c.line {
			t.Errorf("parseGetSpec(%q) = (%q,%d); want (%q,%d)",
				c.in, spec, line, c.spec, c.line)
		}
	}
}

func TestSliceLines(t *testing.T) {
	content := "line1\nline2\nline3\nline4\nline5"

	if got := sliceLines(content, 0, 0); got != content {
		t.Errorf("no limits mutated content")
	}
	if got := sliceLines(content, 2, 2); got != "line2\nline3" {
		t.Errorf("from=2 max=2 = %q", got)
	}
	if got := sliceLines(content, 4, 10); got != "line4\nline5" {
		t.Errorf("from=4: %q", got)
	}
	if got := sliceLines(content, 100, 1); got != "" {
		t.Errorf("from past end: %q", got)
	}
}

func TestGetCmdFlags(t *testing.T) {
	for _, name := range []string{"from", "lines", "json"} {
		if f := getCmd.Flags().Lookup(name); f == nil {
			t.Errorf("get missing --%s", name)
		}
	}
}

// TestSuggestFallbackRanksByBasename pins the V5 fix: the fuzzy
// suggestion path ranks substring matches across collections (the old
// implementation used a doublestar `*X*` glob that couldn't cross `/`,
// so it never produced suggestions).
func TestSuggestFallbackRanksByBasename(t *testing.T) {
	dir := t.TempDir()
	dbPath = filepath.Join(dir, "i.db")
	t.Cleanup(func() { dbPath = "" })

	s, err := openStore()
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()

	cdir := t.TempDir()
	for _, p := range []string{"meeting.md", "ideas.md", "notes/meeting-2026.md"} {
		full := filepath.Join(cdir, p)
		if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(full, []byte("# x"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	c, _ := s.AddCollection("notes", cdir, "", "")
	if _, err := s.IndexCollection(c.ID); err != nil {
		t.Fatal(err)
	}

	err = suggestFallback(s, "meeting")
	if err == nil {
		t.Fatal("expected error with suggestions")
	}
	msg := err.Error()
	if !strings.Contains(msg, "did you mean") {
		t.Errorf("missing 'did you mean' header: %q", msg)
	}
	if !strings.Contains(msg, "meeting.md") {
		t.Errorf("expected meeting.md in suggestions, got: %q", msg)
	}
}
