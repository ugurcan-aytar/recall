package commands

import (
	"testing"
)

func TestModelsSubcommandsRegistered(t *testing.T) {
	want := []string{"list", "download", "path"}
	got := map[string]bool{}
	for _, c := range modelsCmd.Commands() {
		got[c.Name()] = true
	}
	for _, n := range want {
		if !got[n] {
			t.Errorf("models missing subcommand %q", n)
		}
	}
}

func TestModelsDownloadFlags(t *testing.T) {
	for _, name := range []string{"url", "sha256", "dest"} {
		if f := modelsDownloadCmd.Flags().Lookup(name); f == nil {
			t.Errorf("models download missing --%s", name)
		}
	}
}
