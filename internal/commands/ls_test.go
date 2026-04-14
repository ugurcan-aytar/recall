package commands

import "testing"

func TestSplitCollectionPath(t *testing.T) {
	cases := []struct {
		in, coll, rest string
	}{
		{"notes", "notes", ""},
		{"notes/work", "notes", "work"},
		{"notes/deep/nested", "notes", "deep/nested"},
	}
	for _, c := range cases {
		coll, rest := splitCollectionPath(c.in)
		if coll != c.coll || rest != c.rest {
			t.Errorf("splitCollectionPath(%q) = (%q,%q); want (%q,%q)",
				c.in, coll, rest, c.coll, c.rest)
		}
	}
}

func TestLsCmdDefined(t *testing.T) {
	if lsCmd.Use == "" {
		t.Fatal("lsCmd.Use is empty")
	}
}
