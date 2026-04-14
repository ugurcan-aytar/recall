package chunk

import (
	"path/filepath"
	"strings"
)

// ChunkStrategy controls how [ChunkFile] picks between markdown
// break-point scoring and tree-sitter AST chunking.
type ChunkStrategy string

const (
	// StrategyAuto inspects the file extension: code files with a
	// supported tree-sitter grammar use AST chunking; everything else
	// (markdown, plain text, config) uses the markdown chunker.
	StrategyAuto ChunkStrategy = "auto"

	// StrategyRegex forces the markdown break-point chunker for every
	// file. Useful for benchmarking or when the AST chunker is suspected
	// of bad cuts on a specific corpus.
	StrategyRegex ChunkStrategy = "regex"

	// StrategyAST forces the tree-sitter AST chunker for every file.
	// Falls back to the markdown chunker for unsupported languages so
	// the call never fails.
	StrategyAST ChunkStrategy = "ast"
)

// FileType is a coarse extension-based classification.
type FileType int

const (
	FileTypeMarkdown FileType = iota // .md / .txt / .rst — markdown chunker
	FileTypeCode                     // .go / .py / … — AST chunker when supported
	FileTypeConfig                   // .yaml / .json / … — markdown chunker (usually one chunk)
	FileTypeText                     // anything else — markdown chunker
)

// DetectFileType returns the [FileType] for path based on its extension.
// Detection is purely lexical; we do not sniff content.
func DetectFileType(path string) FileType {
	switch strings.ToLower(filepath.Ext(path)) {
	case ".md", ".txt", ".rst":
		return FileTypeMarkdown
	case ".go", ".ts", ".tsx", ".js", ".jsx", ".py", ".java", ".rs",
		".rb", ".php", ".c", ".cpp", ".h", ".hpp", ".cs", ".swift",
		".kt", ".scala":
		return FileTypeCode
	case ".yaml", ".yml", ".toml", ".json", ".xml", ".html", ".css",
		".scss", ".proto", ".graphql", ".sql":
		return FileTypeConfig
	default:
		return FileTypeText
	}
}

// DetectLanguage returns the tree-sitter language identifier for path,
// or "" when the language has no AST chunker. The returned strings are
// the keys of [AstBreakPoints].
func DetectLanguage(path string) string {
	switch strings.ToLower(filepath.Ext(path)) {
	case ".go":
		return "go"
	case ".py":
		return "python"
	case ".ts", ".tsx":
		return "typescript"
	case ".js", ".jsx":
		return "javascript"
	case ".java":
		return "java"
	case ".rs":
		return "rust"
	default:
		return ""
	}
}

// ChunkFile is the strategy-aware entry point. It dispatches between
// [Split] (markdown) and [ChunkCode] (AST) according to strategy and the
// file's extension. Pass 0 for default targetTokens / overlapPct.
//
//	auto  : .go/.py/.ts/etc. with a supported grammar → AST; else markdown
//	regex : always markdown
//	ast   : always AST (with internal fallback for unsupported langs)
//
// Unsupported languages under StrategyAuto fall back silently to the
// markdown chunker — by design. recall never errors on a file just
// because it can't AST-parse it.
func ChunkFile(content, path string, strategy ChunkStrategy, targetTokens int, overlapPct float64) []Chunk {
	switch strategy {
	case StrategyRegex:
		return Split(content, targetTokens, overlapPct)
	case StrategyAST:
		lang := DetectLanguage(path)
		return ChunkCode(content, lang, targetTokens, overlapPct)
	case StrategyAuto, "":
		if DetectFileType(path) == FileTypeCode {
			lang := DetectLanguage(path)
			if LanguageSupported(lang) {
				return ChunkCode(content, lang, targetTokens, overlapPct)
			}
		}
		return Split(content, targetTokens, overlapPct)
	default:
		// Unknown strategy values behave like auto so a typo doesn't
		// hard-fail an indexing run.
		return ChunkFile(content, path, StrategyAuto, targetTokens, overlapPct)
	}
}
