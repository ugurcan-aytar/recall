package store

import (
	"errors"
	"os"
	"path/filepath"
	"testing"
)

func mustIndex(t *testing.T, s *Store, id int64) IndexStats {
	t.Helper()
	stats, err := s.IndexCollection(id)
	if err != nil {
		t.Fatalf("IndexCollection: %v", err)
	}
	return stats
}

func TestScanAndIndex(t *testing.T) {
	s := openTestStore(t)
	dir := tempCollectionDir(t, map[string]string{
		"intro.md":       "# Intro\n\nthis is the intro",
		"notes/ideas.md": "# Ideas\n\nbig thoughts",
		"ignored.log":    "not indexed",
	})

	c, err := s.AddCollection("n", dir, "", "")
	if err != nil {
		t.Fatal(err)
	}
	stats := mustIndex(t, s, c.ID)
	if stats.Indexed != 2 {
		t.Fatalf("indexed=%d, want 2 (.log should be skipped by glob)", stats.Indexed)
	}
	if stats.Updated != 0 || stats.Removed != 0 {
		t.Errorf("unexpected mutation counts: %+v", stats)
	}
	n, err := s.TotalDocumentCount()
	if err != nil {
		t.Fatal(err)
	}
	if n != 2 {
		t.Fatalf("total docs = %d, want 2", n)
	}
}

func TestScanUpdate(t *testing.T) {
	s := openTestStore(t)
	dir := tempCollectionDir(t, map[string]string{"a.md": "# A\nfirst"})

	c, _ := s.AddCollection("n", dir, "", "")
	_ = mustIndex(t, s, c.ID)

	// Modify on disk.
	if err := os.WriteFile(filepath.Join(dir, "a.md"), []byte("# A\nsecond"), 0o644); err != nil {
		t.Fatal(err)
	}

	stats := mustIndex(t, s, c.ID)
	if stats.Updated != 1 {
		t.Fatalf("updated=%d, want 1", stats.Updated)
	}
	if stats.Indexed != 0 {
		t.Errorf("indexed=%d on update run", stats.Indexed)
	}

	// Hash should reflect new content.
	doc, err := s.GetDocument("n/a.md")
	if err != nil {
		t.Fatal(err)
	}
	if doc.Content != "# A\nsecond" {
		t.Errorf("content = %q", doc.Content)
	}
	if doc.ContentHash != ComputeContentHash([]byte("# A\nsecond")) {
		t.Errorf("hash did not update")
	}
}

func TestScanDelete(t *testing.T) {
	s := openTestStore(t)
	dir := tempCollectionDir(t, map[string]string{
		"keep.md": "# keep",
		"gone.md": "# gone",
	})

	c, _ := s.AddCollection("n", dir, "", "")
	_ = mustIndex(t, s, c.ID)

	if err := os.Remove(filepath.Join(dir, "gone.md")); err != nil {
		t.Fatal(err)
	}

	stats := mustIndex(t, s, c.ID)
	if stats.Removed != 1 {
		t.Fatalf("removed=%d, want 1", stats.Removed)
	}
	if _, err := s.GetDocument("n/gone.md"); !errors.Is(err, ErrDocumentNotFound) {
		t.Errorf("gone.md still present: %v", err)
	}
}

func TestScanUnchanged(t *testing.T) {
	s := openTestStore(t)
	dir := tempCollectionDir(t, map[string]string{"a.md": "x"})
	c, _ := s.AddCollection("n", dir, "", "")

	_ = mustIndex(t, s, c.ID)
	stats := mustIndex(t, s, c.ID)
	if stats.Unchanged != 1 || stats.Indexed != 0 || stats.Updated != 0 {
		t.Errorf("second pass stats = %+v", stats)
	}
}

func TestTitleExtraction(t *testing.T) {
	cases := []struct {
		name     string
		content  string
		filename string
		want     string
	}{
		{"h1 heading", "# My Title\n\nbody", "a.md", "My Title"},
		{"no heading", "just text\nmore", "notes.md", "notes"},
		{"yaml frontmatter then h1", "---\ntitle: foo\n---\n# Real Heading\nbody", "x.md", "Real Heading"},
		{"h1 not first line", "some preamble\n# Buried Heading\n", "x.md", "Buried Heading"},
		{"nested path filename fallback", "plain body", "docs/deep/file.md", "file"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := ExtractTitle(c.content, c.filename)
			if got != c.want {
				t.Errorf("ExtractTitle = %q, want %q", got, c.want)
			}
		})
	}
}

func TestDocIdGeneration(t *testing.T) {
	h1 := ComputeContentHash([]byte("hello"))
	h2 := ComputeContentHash([]byte("hello"))
	if h1 != h2 {
		t.Fatal("hash not deterministic")
	}
	if DocIDFromHash(h1) != h1[:DocIDLength] {
		t.Fatalf("DocIDFromHash = %q", DocIDFromHash(h1))
	}
	if DocIDFromHash("abc") != "abc" {
		t.Fatalf("short hash passthrough broken: %q", DocIDFromHash("abc"))
	}
	// Different content → different docid.
	h3 := ComputeContentHash([]byte("world"))
	if DocIDFromHash(h1) == DocIDFromHash(h3) {
		t.Fatalf("collision on trivial inputs: %q vs %q", DocIDFromHash(h1), DocIDFromHash(h3))
	}
}

func TestMultiGetGlob(t *testing.T) {
	s := openTestStore(t)
	dir := tempCollectionDir(t, map[string]string{
		"a.md":         "# A",
		"sub/b.md":     "# B",
		"notes/ops.md": "# Ops",
	})
	c, _ := s.AddCollection("n", dir, "", "")
	_ = mustIndex(t, s, c.ID)

	docs, err := s.MultiGetGlob("n/*.md")
	if err != nil {
		t.Fatal(err)
	}
	if len(docs) != 1 || docs[0].Path != "a.md" {
		t.Errorf("got %d docs, want 1 (a.md)", len(docs))
	}

	all, err := s.MultiGetGlob("n/**/*.md")
	if err != nil {
		t.Fatal(err)
	}
	if len(all) != 3 {
		t.Errorf("recursive glob returned %d docs, want 3", len(all))
	}
}

func TestAllDocuments(t *testing.T) {
	s := openTestStore(t)
	dir := tempCollectionDir(t, map[string]string{
		"a.md": "# a\nbody",
		"b.md": "# b\nbody",
	})
	c, _ := s.AddCollection("n", dir, "", "")
	if _, err := s.IndexCollection(c.ID); err != nil {
		t.Fatal(err)
	}
	docs, err := s.AllDocuments()
	if err != nil {
		t.Fatal(err)
	}
	if len(docs) != 2 {
		t.Errorf("got %d docs, want 2", len(docs))
	}
	for _, d := range docs {
		if d.CollectionName != "n" {
			t.Errorf("CollectionName = %q", d.CollectionName)
		}
		if d.Content == "" {
			t.Errorf("empty content for %s", d.Path)
		}
	}
}

func TestExtractCodeTitle(t *testing.T) {
	cases := []struct {
		name, file, body, want string
	}{
		{
			"go pkg + symbol",
			"main.go",
			"package auth\n\nimport \"fmt\"\n\nfunc ValidateToken(t string) error {\n  return nil\n}\n",
			"package auth — ValidateToken",
		},
		{
			"go pkg + type",
			"x.go",
			"package gateway\n\ntype Server struct { Port int }\n",
			"package gateway — Server",
		},
		{
			"go pkg only (private fn)",
			"x.go",
			"package noop\n\nfunc internal() {}\n",
			"package noop",
		},
		{
			"python class",
			"app.py",
			"import os\n\nclass Greeter:\n    pass\n",
			"Greeter",
		},
		{
			"python def",
			"app.py",
			"def hello():\n    return 1\n",
			"hello",
		},
		{
			"ts export class",
			"svc.ts",
			"import x from 'y';\nexport class TokenService {}\n",
			"TokenService",
		},
		{
			"ts export default function",
			"svc.ts",
			"export default function handler() {}\n",
			"handler",
		},
		{
			"ts top-level fallback",
			"x.ts",
			"function helper() { return 1; }\n",
			"helper",
		},
		{
			"java public class",
			"Foo.java",
			"package x;\npublic class Foo {}\n",
			"Foo",
		},
		{
			"rust pub struct",
			"lib.rs",
			"use std::io;\npub struct Point { x: i32 }\n",
			"Point",
		},
		{
			"non-code falls through to filename",
			"notes/ideas.md",
			"plain body, no heading",
			"ideas",
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := ExtractTitle(c.body, c.file)
			if got != c.want {
				t.Errorf("ExtractTitle(%q) = %q, want %q", c.file, got, c.want)
			}
		})
	}
}

func TestStripFrontmatter(t *testing.T) {
	in := "---\ntitle: X\ntags: [a, b]\n---\n\n# Body\n"
	got := stripFrontmatter(in)
	want := "# Body\n"
	if got != want {
		t.Errorf("stripFrontmatter = %q, want %q", got, want)
	}

	noFM := "# Body only"
	if stripFrontmatter(noFM) != noFM {
		t.Error("stripFrontmatter mutated input without frontmatter")
	}

	tomlFM := "+++\nkey = 1\n+++\nbody\n"
	if stripFrontmatter(tomlFM) != "body\n" {
		t.Errorf("toml frontmatter: %q", stripFrontmatter(tomlFM))
	}
}
