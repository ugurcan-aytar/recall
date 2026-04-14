package store

import (
	"errors"
	"testing"
)

func TestAddContext(t *testing.T) {
	s := openTestStore(t)

	if err := s.AddContext("notes", "work", "Work notes folder"); err != nil {
		t.Fatalf("AddContext: %v", err)
	}
	got, err := s.GetContext("notes", "work")
	if err != nil {
		t.Fatalf("GetContext: %v", err)
	}
	if got.Context != "Work notes folder" {
		t.Errorf("text = %q", got.Context)
	}

	// Upsert: calling again with new text replaces.
	if err := s.AddContext("notes", "work", "Updated"); err != nil {
		t.Fatal(err)
	}
	got, _ = s.GetContext("notes", "work")
	if got.Context != "Updated" {
		t.Errorf("upsert failed: %q", got.Context)
	}

	// Empty context should error.
	if err := s.AddContext("notes", "work", "  "); err == nil {
		t.Error("expected error on empty context")
	}
}

func TestAddGlobalContext(t *testing.T) {
	s := openTestStore(t)

	if err := s.AddContext("", "/", "All notes for my projects"); err != nil {
		t.Fatalf("AddContext global: %v", err)
	}
	got, err := s.GetContext("", "/")
	if err != nil {
		t.Fatalf("GetContext: %v", err)
	}
	if got.Collection != "" {
		t.Errorf("collection = %q, want empty for global", got.Collection)
	}
	if got.Path != "/" {
		t.Errorf("path = %q, want /", got.Path)
	}
}

func TestRemoveContext(t *testing.T) {
	s := openTestStore(t)

	_ = s.AddContext("notes", "meetings", "Meeting notes")
	if err := s.RemoveContext("notes", "meetings"); err != nil {
		t.Fatalf("RemoveContext: %v", err)
	}
	if err := s.RemoveContext("notes", "meetings"); !errors.Is(err, ErrContextNotFound) {
		t.Errorf("second remove: want ErrContextNotFound, got %v", err)
	}
}

func TestContextCheck(t *testing.T) {
	s := openTestStore(t)

	d1 := tempCollectionDir(t, map[string]string{"a.md": "a"})
	d2 := tempCollectionDir(t, map[string]string{"b.md": "b"})
	if _, err := s.AddCollection("contextful", d1, "", ""); err != nil {
		t.Fatal(err)
	}
	if _, err := s.AddCollection("contextless", d2, "", ""); err != nil {
		t.Fatal(err)
	}
	if err := s.AddContext("contextful", "/", "described"); err != nil {
		t.Fatal(err)
	}

	missing, err := s.CheckContexts()
	if err != nil {
		t.Fatalf("CheckContexts: %v", err)
	}
	if len(missing) != 1 || missing[0] != "contextless" {
		t.Errorf("got %v, want [contextless]", missing)
	}
}

func TestListContextsEmpty(t *testing.T) {
	s := openTestStore(t)
	cs, err := s.ListContexts()
	if err != nil {
		t.Fatalf("list: %v", err)
	}
	if len(cs) != 0 {
		t.Fatalf("want empty, got %+v", cs)
	}
}

// TestListContextsIncludesCollectionLevel pins the validation-time fix:
// `--context "Sample notes…"` set on `collection add` must show up in
// `recall context list`, otherwise users have no way to see what they
// just wrote.
func TestListContextsIncludesCollectionLevel(t *testing.T) {
	s := openTestStore(t)
	dir := tempCollectionDir(t, map[string]string{"a.md": "x"})

	if _, err := s.AddCollection("notes", dir, "", "Personal notes"); err != nil {
		t.Fatal(err)
	}
	// Also add a plain path-context so we can confirm both layers appear.
	if err := s.AddContext("notes", "work", "Work-related sub-folder"); err != nil {
		t.Fatal(err)
	}

	cs, err := s.ListContexts()
	if err != nil {
		t.Fatal(err)
	}
	var sawCollection, sawPath bool
	for _, c := range cs {
		if c.Collection == "notes" && c.Path == "/" && c.Context == "Personal notes" {
			sawCollection = true
		}
		if c.Collection == "notes" && c.Path == "work" && c.Context == "Work-related sub-folder" {
			sawPath = true
		}
	}
	if !sawCollection {
		t.Errorf("collection-level context missing from ListContexts: %+v", cs)
	}
	if !sawPath {
		t.Errorf("path-level context missing: %+v", cs)
	}
}
