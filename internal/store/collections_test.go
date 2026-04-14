package store

import (
	"errors"
	"os"
	"path/filepath"
	"testing"
)

func tempCollectionDir(t *testing.T, files map[string]string) string {
	t.Helper()
	dir := t.TempDir()
	for name, content := range files {
		p := filepath.Join(dir, name)
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatalf("mkdir: %v", err)
		}
		if err := os.WriteFile(p, []byte(content), 0o644); err != nil {
			t.Fatalf("write: %v", err)
		}
	}
	return dir
}

func TestAddCollection(t *testing.T) {
	s := openTestStore(t)
	dir := tempCollectionDir(t, map[string]string{"a.md": "# Hello"})

	c, err := s.AddCollection("notes", dir, "", "my notes")
	if err != nil {
		t.Fatalf("AddCollection: %v", err)
	}
	if c.Name != "notes" || c.Path != dir || c.Context != "my notes" {
		t.Fatalf("got %+v", c)
	}
	if c.GlobPattern != DefaultGlobPattern {
		t.Errorf("glob default = %q, want %q", c.GlobPattern, DefaultGlobPattern)
	}

	cols, err := s.ListCollections()
	if err != nil {
		t.Fatalf("ListCollections: %v", err)
	}
	if len(cols) != 1 || cols[0].Name != "notes" {
		t.Fatalf("list = %+v", cols)
	}
}

func TestAddDuplicateCollection(t *testing.T) {
	s := openTestStore(t)
	dir := tempCollectionDir(t, map[string]string{"a.md": "x"})

	if _, err := s.AddCollection("n", dir, "", ""); err != nil {
		t.Fatalf("first add: %v", err)
	}
	_, err := s.AddCollection("n", dir, "", "")
	if !errors.Is(err, ErrCollectionExists) {
		t.Fatalf("want ErrCollectionExists, got %v", err)
	}
}

func TestRemoveCollection(t *testing.T) {
	s := openTestStore(t)
	dir := tempCollectionDir(t, map[string]string{"a.md": "# Hi"})

	c, err := s.AddCollection("n", dir, "", "")
	if err != nil {
		t.Fatalf("AddCollection: %v", err)
	}
	if _, err := s.IndexCollection(c.ID); err != nil {
		t.Fatalf("IndexCollection: %v", err)
	}

	if err := s.RemoveCollection("n"); err != nil {
		t.Fatalf("RemoveCollection: %v", err)
	}

	// Cascade: documents for this collection should be gone.
	var count int
	if err := s.DB().QueryRow(`SELECT COUNT(*) FROM documents`).Scan(&count); err != nil {
		t.Fatal(err)
	}
	if count != 0 {
		t.Errorf("documents not cascaded, count=%d", count)
	}

	err = s.RemoveCollection("n")
	if !errors.Is(err, ErrCollectionNotFound) {
		t.Errorf("remove missing: want ErrCollectionNotFound, got %v", err)
	}
}

func TestListCollections(t *testing.T) {
	s := openTestStore(t)

	cols, err := s.ListCollections()
	if err != nil {
		t.Fatalf("empty list: %v", err)
	}
	if len(cols) != 0 {
		t.Fatalf("want empty, got %+v", cols)
	}

	dir1 := tempCollectionDir(t, map[string]string{"x.md": "x"})
	dir2 := tempCollectionDir(t, map[string]string{"y.md": "y"})

	if _, err := s.AddCollection("a-notes", dir1, "", ""); err != nil {
		t.Fatalf("add a: %v", err)
	}
	if _, err := s.AddCollection("b-notes", dir2, "", ""); err != nil {
		t.Fatalf("add b: %v", err)
	}

	cols, err = s.ListCollections()
	if err != nil {
		t.Fatalf("list: %v", err)
	}
	if len(cols) != 2 {
		t.Fatalf("want 2, got %d", len(cols))
	}
	// Sorted alphabetically by name.
	if cols[0].Name != "a-notes" || cols[1].Name != "b-notes" {
		t.Errorf("order = %s,%s", cols[0].Name, cols[1].Name)
	}
}

func TestRenameCollection(t *testing.T) {
	s := openTestStore(t)
	dir := tempCollectionDir(t, map[string]string{"a.md": "# a"})

	if _, err := s.AddCollection("old", dir, "", ""); err != nil {
		t.Fatalf("add: %v", err)
	}
	if err := s.RenameCollection("old", "new"); err != nil {
		t.Fatalf("rename: %v", err)
	}
	if _, err := s.GetCollectionByName("old"); !errors.Is(err, ErrCollectionNotFound) {
		t.Errorf("old still present: %v", err)
	}
	if _, err := s.GetCollectionByName("new"); err != nil {
		t.Errorf("new missing: %v", err)
	}

	// rename missing → not found
	if err := s.RenameCollection("ghost", "x"); !errors.Is(err, ErrCollectionNotFound) {
		t.Errorf("rename missing: want ErrCollectionNotFound, got %v", err)
	}

	// rename to existing name → exists
	dir2 := tempCollectionDir(t, map[string]string{"b.md": "b"})
	if _, err := s.AddCollection("other", dir2, "", ""); err != nil {
		t.Fatal(err)
	}
	if err := s.RenameCollection("new", "other"); !errors.Is(err, ErrCollectionExists) {
		t.Errorf("rename to existing: want ErrCollectionExists, got %v", err)
	}
}
