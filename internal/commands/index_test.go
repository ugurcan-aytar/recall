package commands

import "testing"

func TestIndexCmdDefined(t *testing.T) {
	if indexCmd.Use != "index" {
		t.Fatalf("Use = %q", indexCmd.Use)
	}
	if f := indexCmd.Flags().Lookup("pull"); f == nil {
		t.Error("index missing --pull flag")
	}
}
