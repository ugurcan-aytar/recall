package commands

import "testing"

func TestCollectionSubcommandsRegistered(t *testing.T) {
	want := []string{"add", "remove", "list", "rename"}
	got := map[string]bool{}
	for _, c := range collectionCmd.Commands() {
		got[c.Name()] = true
	}
	for _, n := range want {
		if !got[n] {
			t.Errorf("collection missing subcommand %q", n)
		}
	}
}

func TestCollectionAddFlags(t *testing.T) {
	for _, name := range []string{"name", "mask", "context"} {
		if f := collectionAddCmd.Flags().Lookup(name); f == nil {
			t.Errorf("collection add missing --%s", name)
		}
	}
}
