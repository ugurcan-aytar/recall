package commands

import "testing"

func TestContainsGlobMeta(t *testing.T) {
	for _, s := range []string{"*.md", "a?.md", "[ab].md", "docs/{a,b}.md"} {
		if !containsGlobMeta(s) {
			t.Errorf("containsGlobMeta(%q) = false", s)
		}
	}
	for _, s := range []string{"plain.md", "notes/meeting.md", "#abc123"} {
		if containsGlobMeta(s) {
			t.Errorf("containsGlobMeta(%q) = true", s)
		}
	}
}

func TestMultiGetCmdFlags(t *testing.T) {
	for _, name := range []string{"max-bytes", "json"} {
		if f := multiGetCmd.Flags().Lookup(name); f == nil {
			t.Errorf("multi-get missing --%s", name)
		}
	}
}
