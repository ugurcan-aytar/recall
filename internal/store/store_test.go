package store

import (
	"os"
	"path/filepath"
	"testing"
)

func openTestStore(t *testing.T) *Store {
	t.Helper()
	dir := t.TempDir()
	s, err := Open(filepath.Join(dir, "index.db"))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	t.Cleanup(func() { _ = s.Close() })
	return s
}

func TestOpen(t *testing.T) {
	s := openTestStore(t)

	for _, table := range []string{"collections", "documents", "documents_fts", "path_contexts", "metadata"} {
		var name string
		err := s.DB().QueryRow(
			`SELECT name FROM sqlite_master WHERE name = ?`, table,
		).Scan(&name)
		if err != nil {
			t.Fatalf("table %s not created: %v", table, err)
		}
	}

	v, present, err := s.GetMetadata("schema_version")
	if err != nil {
		t.Fatalf("GetMetadata: %v", err)
	}
	if !present || v != SchemaVersion {
		t.Fatalf("schema_version: got (%q, %v), want (%q, true)", v, present, SchemaVersion)
	}
}

func TestOpenExisting(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "idx.db")

	s, err := Open(path)
	if err != nil {
		t.Fatalf("first Open: %v", err)
	}
	if _, err := s.AddCollection("notes", dir, "", ""); err != nil {
		t.Fatalf("AddCollection: %v", err)
	}
	if err := s.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	s2, err := Open(path)
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	defer s2.Close()

	cols, err := s2.ListCollections()
	if err != nil {
		t.Fatalf("ListCollections: %v", err)
	}
	if len(cols) != 1 || cols[0].Name != "notes" {
		t.Fatalf("data did not survive reopen: %+v", cols)
	}
}

func TestClose(t *testing.T) {
	dir := t.TempDir()
	s, err := Open(filepath.Join(dir, "x.db"))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	if err := s.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if _, err := s.DB().Exec(`SELECT 1`); err == nil {
		t.Fatal("expected error after Close, got nil")
	}
}

func TestResolveDBPath(t *testing.T) {
	t.Setenv("RECALL_DB_PATH", "")

	home, err := os.UserHomeDir()
	if err != nil {
		t.Fatalf("UserHomeDir: %v", err)
	}

	cases := []struct {
		in   string
		want string
		env  string
	}{
		{"/tmp/foo.db", "/tmp/foo.db", ""},
		{"~/custom.db", filepath.Join(home, "custom.db"), ""},
		{"", filepath.Join(home, ".recall", "index.db"), ""},
		{"", "/env/path.db", "/env/path.db"},
	}
	for _, c := range cases {
		t.Setenv("RECALL_DB_PATH", c.env)
		got, err := ResolveDBPath(c.in)
		if err != nil {
			t.Fatalf("ResolveDBPath(%q, env=%q): %v", c.in, c.env, err)
		}
		if got != c.want {
			t.Errorf("ResolveDBPath(%q, env=%q) = %q, want %q", c.in, c.env, got, c.want)
		}
	}
}

func TestMetadataUpsert(t *testing.T) {
	s := openTestStore(t)

	if err := s.SetMetadata("foo", "bar"); err != nil {
		t.Fatalf("SetMetadata: %v", err)
	}
	v, ok, err := s.GetMetadata("foo")
	if err != nil || !ok || v != "bar" {
		t.Fatalf("got (%q, %v, %v), want (bar, true, nil)", v, ok, err)
	}
	if err := s.SetMetadata("foo", "baz"); err != nil {
		t.Fatalf("update: %v", err)
	}
	v, _, _ = s.GetMetadata("foo")
	if v != "baz" {
		t.Fatalf("upsert did not overwrite: %q", v)
	}
}
