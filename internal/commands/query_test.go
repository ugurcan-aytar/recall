package commands

import "testing"

func TestQueryCmdFlags(t *testing.T) {
	if queryCmd.Use == "" {
		t.Fatal("queryCmd.Use is empty")
	}
	for _, name := range []string{
		"limit", "collection", "all", "min-score", "full", "explain",
		"json", "csv", "md", "xml", "files", "chunk-strategy",
	} {
		if f := queryCmd.Flags().Lookup(name); f == nil {
			t.Errorf("query missing --%s", name)
		}
	}
}
