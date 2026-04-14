package commands

import (
	"strings"
	"testing"

	"github.com/ugurcan-aytar/recall/internal/store"
)

func TestDefaultLimitForFormat(t *testing.T) {
	if n := defaultLimitForFormat(searchFlags{}); n != 5 {
		t.Errorf("human default = %d, want 5", n)
	}
	for _, f := range []searchFlags{
		{JSON: true}, {CSV: true}, {XML: true}, {Files: true},
	} {
		if n := defaultLimitForFormat(f); n != 20 {
			t.Errorf("machine format default = %d, want 20 (flags=%+v)", n, f)
		}
	}
}

func TestStripANSI(t *testing.T) {
	in := "\x1b[1mhit\x1b[0m rest"
	if got := stripANSI(in); got != "hit rest" {
		t.Errorf("stripANSI = %q", got)
	}
}

func TestSearchFlagsRegistered(t *testing.T) {
	for _, name := range []string{
		"limit", "collection", "all", "min-score", "full", "line-numbers",
		"explain", "json", "csv", "md", "xml", "files",
	} {
		if f := searchCmd.Flags().Lookup(name); f == nil {
			t.Errorf("search missing --%s", name)
		}
	}
}

func TestToJSONResultsStripsANSI(t *testing.T) {
	out := toJSONResults([]store.SearchResult{
		{DocID: "a", Snippet: "\x1b[1mx\x1b[0m"},
	})
	if strings.Contains(out[0].Snippet, "\x1b[") {
		t.Errorf("json snippet kept ANSI: %q", out[0].Snippet)
	}
}
