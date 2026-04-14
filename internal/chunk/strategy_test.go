package chunk

import (
	"strings"
	"testing"
)

func TestStrategyAutoMarkdown(t *testing.T) {
	body := "# Title\n\n" + strings.Repeat("filler word ", 1000)
	auto := ChunkFile(body, "notes.md", StrategyAuto, 800, 0.15)
	regex := Split(body, 800, 0.15)
	if len(auto) != len(regex) {
		t.Errorf("auto on .md should equal regex output: %d vs %d", len(auto), len(regex))
	}
}

func TestStrategyAutoGo(t *testing.T) {
	body := `package main

func A() {}
func B() {}
`
	auto := ChunkFile(body, "main.go", StrategyAuto, 0, 0)
	if len(auto) == 0 {
		t.Fatal("auto on .go produced no chunks")
	}
	// AST chunker must keep both functions intact in some chunk.
	rejoined := joinTexts(auto)
	if !strings.Contains(rejoined, "func A() {}") || !strings.Contains(rejoined, "func B() {}") {
		t.Errorf("auto on .go lost a function; rejoined =\n%s", rejoined)
	}
}

func TestStrategyAutoYaml(t *testing.T) {
	body := "key: value\nlist:\n  - one\n  - two\n"
	auto := ChunkFile(body, "config.yaml", StrategyAuto, 0, 0)
	regex := Split(body, 0, 0)
	if len(auto) != len(regex) {
		t.Errorf("auto on .yaml should fall back to regex: auto=%d regex=%d", len(auto), len(regex))
	}
}

func TestStrategyForceRegex(t *testing.T) {
	body := `package main

func A() {}
func B() {}
`
	out := ChunkFile(body, "main.go", StrategyRegex, 0, 0)
	regex := Split(body, 0, 0)
	if len(out) != len(regex) {
		t.Errorf("regex strategy on .go should match Split: %d vs %d", len(out), len(regex))
	}
}

func TestStrategyForceAST(t *testing.T) {
	body := `def hello():
    return "hi"

def world():
    return "world"
`
	out := ChunkFile(body, "x.py", StrategyAST, 0, 0)
	if len(out) == 0 {
		t.Fatal("ast strategy on .py produced no chunks")
	}
	if !strings.Contains(joinTexts(out), "def hello") || !strings.Contains(joinTexts(out), "def world") {
		t.Error("ast strategy lost a function")
	}
}

func TestStrategyForceASTOnUnsupportedFallsBack(t *testing.T) {
	body := "key: value\nfoo: bar\n"
	out := ChunkFile(body, "x.haskell", StrategyAST, 0, 0)
	// Unknown language under AST strategy → falls back to markdown.
	if len(out) == 0 {
		t.Fatal("AST on unsupported language should still produce chunks")
	}
}

func TestStrategyUnknownValueBehavesAsAuto(t *testing.T) {
	body := "# Hi\nbody"
	auto := ChunkFile(body, "x.md", StrategyAuto, 0, 0)
	weird := ChunkFile(body, "x.md", ChunkStrategy("nonsense"), 0, 0)
	if len(auto) != len(weird) {
		t.Errorf("unknown strategy should behave like auto: %d vs %d", len(auto), len(weird))
	}
}

func TestDetectFileType(t *testing.T) {
	cases := []struct {
		path string
		want FileType
	}{
		{"notes.md", FileTypeMarkdown},
		{"NOTES.MD", FileTypeMarkdown},
		{"readme.txt", FileTypeMarkdown},
		{"docs.rst", FileTypeMarkdown},
		{"main.go", FileTypeCode},
		{"app.ts", FileTypeCode},
		{"page.tsx", FileTypeCode},
		{"server.js", FileTypeCode},
		{"foo.py", FileTypeCode},
		{"Foo.java", FileTypeCode},
		{"lib.rs", FileTypeCode},
		{"config.yaml", FileTypeConfig},
		{"data.json", FileTypeConfig},
		{"build.toml", FileTypeConfig},
		{"page.html", FileTypeConfig},
		{"unknown.xyz", FileTypeText},
		{"noext", FileTypeText},
	}
	for _, c := range cases {
		if got := DetectFileType(c.path); got != c.want {
			t.Errorf("DetectFileType(%q) = %v, want %v", c.path, got, c.want)
		}
	}
}

func TestDetectLanguage(t *testing.T) {
	cases := []struct {
		path string
		want string
	}{
		{"main.go", "go"},
		{"app.py", "python"},
		{"app.ts", "typescript"},
		{"page.tsx", "typescript"},
		{"server.js", "javascript"},
		{"page.jsx", "javascript"},
		{"Foo.java", "java"},
		{"lib.rs", "rust"},
		{"notes.md", ""},
		{"data.json", ""},
		{"unknown.xyz", ""},
	}
	for _, c := range cases {
		if got := DetectLanguage(c.path); got != c.want {
			t.Errorf("DetectLanguage(%q) = %q, want %q", c.path, got, c.want)
		}
	}
}

// joinTexts concatenates chunk texts in seq order. Chunks may overlap, so
// the result is a strict superset of the original — fine for "did this
// substring survive?" assertions.
func joinTexts(chunks []Chunk) string {
	var b strings.Builder
	for _, c := range chunks {
		b.WriteString(c.Text)
		b.WriteByte('\n')
	}
	return b.String()
}
