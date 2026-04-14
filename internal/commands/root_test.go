package commands

import "testing"

// TestRootHasAllSubcommands asserts every subcommand we intend to expose is
// registered on rootCmd. This catches the "file exists but forgot to
// AddCommand" regression.
func TestRootHasAllSubcommands(t *testing.T) {
	want := []string{
		"collection", "search", "vsearch", "query",
		"index", "embed", "get", "multi-get",
		"context", "status", "doctor", "models",
		"ls", "cleanup", "version",
	}
	got := map[string]bool{}
	for _, c := range rootCmd.Commands() {
		got[c.Name()] = true
	}
	for _, name := range want {
		if !got[name] {
			t.Errorf("rootCmd missing subcommand %q", name)
		}
	}
}

func TestRootPersistentFlags(t *testing.T) {
	for _, name := range []string{"db", "no-color"} {
		if f := rootCmd.PersistentFlags().Lookup(name); f == nil {
			t.Errorf("persistent flag %q missing", name)
		}
	}
}

func TestColorsEnabledRespectsNoColor(t *testing.T) {
	t.Setenv("NO_COLOR", "")
	noColor = false
	if !colorsEnabled() {
		t.Error("default: colors should be enabled")
	}
	noColor = true
	if colorsEnabled() {
		t.Error("--no-color flag: should disable")
	}
	noColor = false
	t.Setenv("NO_COLOR", "1")
	if colorsEnabled() {
		t.Error("NO_COLOR env: should disable")
	}
}
