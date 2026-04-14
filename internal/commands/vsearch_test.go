package commands

import "testing"

func TestVsearchCmdFlags(t *testing.T) {
	if vsearchCmd.Use == "" {
		t.Fatal("vsearchCmd.Use is empty")
	}
	for _, name := range []string{
		"limit", "collection", "all", "min-score", "full", "explain",
		"json", "csv", "md", "xml", "files",
	} {
		if f := vsearchCmd.Flags().Lookup(name); f == nil {
			t.Errorf("vsearch missing --%s", name)
		}
	}
}
