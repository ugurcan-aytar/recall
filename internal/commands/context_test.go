package commands

import "testing"

func TestContextSubcommandsRegistered(t *testing.T) {
	want := []string{"add", "list", "rm", "check"}
	got := map[string]bool{}
	for _, c := range contextCmd.Commands() {
		got[c.Name()] = true
	}
	for _, n := range want {
		if !got[n] {
			t.Errorf("context missing subcommand %q", n)
		}
	}
}

func TestSplitContextSpec(t *testing.T) {
	cases := []struct {
		in   string
		coll string
		path string
	}{
		{"/", "", "/"},
		{"", "", "/"},
		{"notes", "notes", "/"},
		{"notes/", "notes", "/"},
		{"notes/work", "notes", "work"},
		{"notes/deep/path", "notes", "deep/path"},
		{"recall://notes/ideas", "notes", "ideas"},
		{"qmd://notes", "notes", "/"},
	}
	for _, c := range cases {
		coll, path := splitContextSpec(c.in)
		if coll != c.coll || path != c.path {
			t.Errorf("splitContextSpec(%q) = (%q,%q); want (%q,%q)",
				c.in, coll, path, c.coll, c.path)
		}
	}
}

func TestFormatContextRef(t *testing.T) {
	cases := []struct {
		coll, path, want string
	}{
		{"", "/", "(global)"},
		{"notes", "/", "notes"},
		{"notes", "work", "notes/work"},
	}
	for _, c := range cases {
		if got := formatContextRef(c.coll, c.path); got != c.want {
			t.Errorf("formatContextRef(%q,%q) = %q, want %q",
				c.coll, c.path, got, c.want)
		}
	}
}
