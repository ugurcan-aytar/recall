package chunk

import (
	"strings"
	"testing"
)

// containsAll returns true when every needle appears in haystack.
func containsAll(haystack string, needles ...string) bool {
	for _, n := range needles {
		if !strings.Contains(haystack, n) {
			return false
		}
	}
	return true
}

// chunkContaining returns the index of the first chunk whose Text contains
// every needle, or -1.
func chunkContaining(chunks []Chunk, needles ...string) int {
	for i, c := range chunks {
		if containsAll(c.Text, needles...) {
			return i
		}
	}
	return -1
}

// ---------------------------------------------------------------------------
// Per-language R2b.6 tests
// ---------------------------------------------------------------------------

func TestChunkGoFunction(t *testing.T) {
	body := `package main

import "fmt"

func Alpha() error {
	fmt.Println("alpha")
	return nil
}

func Beta() error {
	fmt.Println("beta")
	return nil
}

func Gamma() error {
	fmt.Println("gamma")
	return nil
}
`
	chunks := ChunkCode(body, "go", 0, 0)
	if len(chunks) == 0 {
		t.Fatal("no chunks for Go file")
	}

	// Each function must end up intact in some chunk (its full body is
	// present, not split). With these tiny functions they may all land
	// in one chunk under the default target — the assertion is that no
	// function's body is severed.
	for _, fn := range []string{"func Alpha", "func Beta", "func Gamma"} {
		idx := chunkContaining(chunks, fn, "return nil\n}")
		if idx < 0 {
			t.Errorf("function %q split across chunks", fn)
		}
	}
}

func TestChunkGoLargeFunction(t *testing.T) {
	// Build one Go function whose token count blows past the test's
	// explicit 900-token target, forcing the AST chunker to fall back
	// to internal blank-line splits.
	var b strings.Builder
	b.WriteString("package main\n\nfunc Big() {\n")
	for i := 0; i < 1500; i++ {
		b.WriteString("\tx := \"line " + strings.Repeat("y", 30) + "\"\n")
		if i%20 == 0 {
			b.WriteString("\n")
		}
		b.WriteString("\t_ = x\n")
	}
	b.WriteString("}\n")

	chunks := ChunkCode(b.String(), "go", 900, 0.15)
	if len(chunks) < 2 {
		t.Fatalf("oversize function should produce >= 2 chunks, got %d", len(chunks))
	}
	// Concatenated chunk text must reconstruct the body modulo overlap.
	rejoined := chunks[0].Text
	for i := 1; i < len(chunks); i++ {
		overlap := chunks[i-1].EndPos - chunks[i].StartPos
		if overlap < 0 {
			overlap = 0
		}
		rejoined += chunks[i].Text[overlap:]
	}
	if !strings.HasPrefix(rejoined, "package main") {
		t.Error("first chunk lost the package clause")
	}
	if !strings.HasSuffix(strings.TrimRight(rejoined, "\n"), "}") {
		t.Error("last chunk lost the closing brace")
	}
}

func TestChunkPythonClass(t *testing.T) {
	body := `import os

class Greeter:
    def __init__(self, name):
        self.name = name

    def greet(self):
        return f"hello {self.name}"

class Other:
    pass
`
	chunks := ChunkCode(body, "python", 0, 0)
	if len(chunks) == 0 {
		t.Fatal("no chunks for python")
	}
	idx := chunkContaining(chunks, "class Greeter", "def greet")
	if idx < 0 {
		t.Errorf("class Greeter split — chunks were:\n%+v", chunkSummary(chunks))
	}
}

func TestChunkTypeScriptExports(t *testing.T) {
	body := `import { sign } from "jsonwebtoken";

export class TokenService {
  issue(claims: object): string {
    return sign(claims, "secret");
  }
}

export function verify(token: string): boolean {
  return token.length > 0;
}

export interface Claims {
  sub: string;
}
`
	chunks := ChunkCode(body, "typescript", 0, 0)
	if len(chunks) == 0 {
		t.Fatal("no chunks for ts")
	}
	for _, sym := range []string{"class TokenService", "function verify", "interface Claims"} {
		if chunkContaining(chunks, sym) < 0 {
			t.Errorf("%q missing from any chunk", sym)
		}
	}
}

func TestChunkJavaClass(t *testing.T) {
	body := `package com.example;

import java.util.List;

public class Foo {
    private int x;

    public int getX() {
        return x;
    }

    public void setX(int v) {
        this.x = v;
    }
}

class Helper {
    static String name() { return "h"; }
}
`
	chunks := ChunkCode(body, "java", 0, 0)
	if len(chunks) == 0 {
		t.Fatal("no chunks for java")
	}
	if chunkContaining(chunks, "public class Foo", "getX", "setX") < 0 {
		t.Errorf("Foo class lost a method:\n%+v", chunkSummary(chunks))
	}
}

func TestChunkRustImpl(t *testing.T) {
	body := `use std::fmt;

pub struct Point {
    pub x: i32,
    pub y: i32,
}

impl Point {
    pub fn new(x: i32, y: i32) -> Self {
        Point { x, y }
    }

    pub fn shift(&mut self, dx: i32, dy: i32) {
        self.x += dx;
        self.y += dy;
    }
}
`
	chunks := ChunkCode(body, "rust", 0, 0)
	if len(chunks) == 0 {
		t.Fatal("no chunks for rust")
	}
	if chunkContaining(chunks, "impl Point", "fn new", "fn shift") < 0 {
		t.Errorf("impl Point lost a method:\n%+v", chunkSummary(chunks))
	}
}

func TestChunkImportBlock(t *testing.T) {
	body := `package main

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"time"
)

func main() {
	_ = context.Background
	_ = fmt.Println
	_ = http.NewRequest
	_ = os.Stat
	_ = time.Now
}
`
	chunks := ChunkCode(body, "go", 0, 0)
	if len(chunks) == 0 {
		t.Fatal("no chunks")
	}
	// All five imports should sit in the same chunk.
	idx := chunkContaining(chunks, `"context"`, `"fmt"`, `"net/http"`, `"os"`, `"time"`)
	if idx < 0 {
		t.Errorf("imports split across chunks; summary:\n%+v", chunkSummary(chunks))
	}
}

func TestChunkUnsupportedLanguage(t *testing.T) {
	body := "# Doc\n\n" + strings.Repeat("filler word ", 1000)
	// Pass a clearly unsupported language label.
	chunks := ChunkCode(body, "haskell", 800, 0.15)
	if len(chunks) == 0 {
		t.Fatal("unsupported lang should still produce chunks via fallback")
	}
	// Should behave identically to the markdown chunker.
	mdChunks := Split(body, 800, 0.15)
	if len(chunks) != len(mdChunks) {
		t.Errorf("fallback chunk count = %d, markdown = %d", len(chunks), len(mdChunks))
	}
}

func TestChunkCodeEmptyContent(t *testing.T) {
	if c := ChunkCode("", "go", 0, 0); c != nil {
		t.Errorf("empty content should yield nil, got %+v", c)
	}
}

func TestLanguageSupported(t *testing.T) {
	for _, lang := range []string{"go", "python", "typescript", "javascript", "java", "rust"} {
		if !LanguageSupported(lang) {
			t.Errorf("%s should be supported", lang)
		}
	}
	if LanguageSupported("haskell") {
		t.Error("haskell should not be supported")
	}
	if LanguageSupported("") {
		t.Error("empty lang should not be supported")
	}
}

func TestAstBreakPointsMatchROADMAP(t *testing.T) {
	// Pin a few critical entries so accidental edits to AstBreakPoints
	// surface as test failures.
	cases := []struct {
		lang, node string
		want       int
	}{
		{"go", "function_declaration", 100},
		{"go", "method_declaration", 100},
		{"go", "import_declaration", 80},
		{"python", "class_definition", 100},
		{"python", "import_from_statement", 80},
		{"typescript", "interface_declaration", 90},
		{"java", "method_declaration", 100},
		{"rust", "impl_item", 100},
	}
	for _, c := range cases {
		if got := AstBreakPoints[c.lang][c.node]; got != c.want {
			t.Errorf("AstBreakPoints[%s][%s] = %d, want %d", c.lang, c.node, got, c.want)
		}
	}
}

// chunkSummary is a debug helper that summarises chunks for failure
// messages.
func chunkSummary(chunks []Chunk) string {
	var sb strings.Builder
	for i, c := range chunks {
		head := c.Text
		if len(head) > 60 {
			head = head[:60]
		}
		head = strings.ReplaceAll(head, "\n", " ⏎ ")
		sb.WriteString("  [")
		sb.WriteString(itoa(i))
		sb.WriteString("] ")
		sb.WriteString(head)
		sb.WriteString("\n")
	}
	return sb.String()
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var b [20]byte
	pos := len(b)
	for n > 0 {
		pos--
		b[pos] = byte('0' + n%10)
		n /= 10
	}
	return string(b[pos:])
}
