package chunk

import (
	"context"

	sitter "github.com/smacker/go-tree-sitter"
	"github.com/smacker/go-tree-sitter/golang"
	"github.com/smacker/go-tree-sitter/java"
	"github.com/smacker/go-tree-sitter/javascript"
	"github.com/smacker/go-tree-sitter/python"
	"github.com/smacker/go-tree-sitter/rust"
	tsts "github.com/smacker/go-tree-sitter/typescript/typescript"
)

// AstBreakPoints maps each supported language to its AST node-type → score
// table. The scoring scheme matches the markdown chunker: higher = more
// natural cut point. Functions and classes are 100 (always good cut),
// imports are 80, lighter declarations 60, comments 40.
//
// Lookup is by tree-sitter node Type() string, which is the canonical
// node name from the language's grammar definition. Adding a new
// language is one entry in this map plus one case in [parserFor].
var AstBreakPoints = map[string]map[string]int{
	"go": {
		"function_declaration": 100,
		"method_declaration":   100,
		"type_declaration":     90, // struct, interface
		"import_declaration":   80,
		"package_clause":       80,
		"const_declaration":    60,
		"var_declaration":      60,
		"comment":              40,
	},
	"python": {
		"function_definition":   100,
		"class_definition":      100,
		"decorated_definition":  100, // @decorator + def/class
		"import_statement":      80,
		"import_from_statement": 80,
		"comment":               40,
	},
	"typescript": {
		"function_declaration":   100,
		"class_declaration":      100,
		"interface_declaration":  90,
		"type_alias_declaration": 80,
		"import_statement":       80,
		"export_statement":       80,
		"lexical_declaration":    60, // const / let
		"comment":                40,
	},
	"javascript": {
		"function_declaration":   100,
		"class_declaration":      100,
		"import_statement":       80,
		"export_statement":       80,
		"lexical_declaration":    60,
		"variable_declaration":   60,
		"comment":                40,
	},
	"java": {
		"method_declaration":    100,
		"class_declaration":     100,
		"interface_declaration": 90,
		"import_declaration":    80,
		"field_declaration":     60,
		"comment":               40,
	},
	"rust": {
		"function_item":   100,
		"impl_item":       100,
		"struct_item":     90,
		"enum_item":       90,
		"trait_item":      90,
		"use_declaration": 80,
		"mod_item":        80,
		"const_item":      60,
		"line_comment":    40,
		"block_comment":   40,
	},
}

// languageFor returns the tree-sitter language for the named language, or
// nil when unsupported.
func languageFor(lang string) *sitter.Language {
	switch lang {
	case "go":
		return golang.GetLanguage()
	case "python":
		return python.GetLanguage()
	case "typescript":
		return tsts.GetLanguage()
	case "javascript":
		return javascript.GetLanguage()
	case "java":
		return java.GetLanguage()
	case "rust":
		return rust.GetLanguage()
	default:
		return nil
	}
}

// LanguageSupported reports whether ChunkCode has an AST chunker for lang.
// Callers (in particular the strategy selector) use this to decide
// whether to attempt AST chunking or fall back to the markdown chunker.
func LanguageSupported(lang string) bool {
	_, ok := AstBreakPoints[lang]
	return ok && languageFor(lang) != nil
}

// ChunkCode AST-chunks source code in the given language. Falls back to
// [Split] when the language is unsupported or when the parser bails.
//
//   - targetTokens: 0 ⇒ [DefaultTargetTokens]
//   - overlapPct:   0 ⇒ [DefaultOverlapPct]
func ChunkCode(content, lang string, targetTokens int, overlapPct float64) []Chunk {
	if content == "" {
		return nil
	}
	if targetTokens <= 0 {
		targetTokens = DefaultTargetTokens
	}
	if overlapPct <= 0 {
		overlapPct = DefaultOverlapPct
	}
	if overlapPct > 0.5 {
		overlapPct = 0.5
	}

	if !LanguageSupported(lang) {
		return Split(content, targetTokens, overlapPct)
	}

	tree, err := parseSource(content, lang)
	if err != nil || tree == nil {
		return Split(content, targetTokens, overlapPct)
	}
	defer tree.Close()

	root := tree.RootNode()
	if root == nil || root.ChildCount() == 0 {
		return Split(content, targetTokens, overlapPct)
	}

	scores := AstBreakPoints[lang]
	source := []byte(content)

	// Walk top-level nodes, accumulate until we hit the token target,
	// then emit a chunk that ends at the previous node's boundary.
	var chunks []Chunk
	seq := 0
	chunkStart := 0
	chunkTokens := 0

	for i := 0; i < int(root.ChildCount()); i++ {
		node := root.Child(i)
		if node == nil {
			continue
		}
		nodeStart := int(node.StartByte())
		nodeEnd := int(node.EndByte())
		if nodeStart < chunkStart {
			// Should not happen; defensive against grammar quirks.
			continue
		}
		nodeText := string(source[nodeStart:nodeEnd])
		nodeTokens := EstimateTokens(nodeText)
		_, isBoundary := scores[node.Type()]

		// Case 1: a single node already exceeds the target. Flush whatever
		// we've accumulated so far, then chunk the oversize node by itself
		// (recursively, via blank-line breaks).
		if nodeTokens > targetTokens {
			if chunkStart < nodeStart {
				chunks = appendChunk(chunks, source, chunkStart, nodeStart, &seq)
			}
			subChunks := splitOversizeNode(source, nodeStart, nodeEnd, targetTokens, overlapPct)
			for _, c := range subChunks {
				c.Seq = seq
				seq++
				chunks = append(chunks, c)
			}
			chunkStart = nodeEnd
			chunkTokens = 0
			continue
		}

		// Case 2: adding this node would push us past the target. Cut
		// before it (only if we're sitting on a boundary node, otherwise
		// cut at the previous node's end).
		if chunkTokens > 0 && chunkTokens+nodeTokens > targetTokens && isBoundary {
			chunks = appendChunk(chunks, source, chunkStart, nodeStart, &seq)
			chunkStart = nodeStart
			chunkTokens = 0
		}

		chunkTokens += nodeTokens
	}

	// Flush the tail.
	if chunkStart < len(source) {
		chunks = appendChunk(chunks, source, chunkStart, len(source), &seq)
	}

	if len(chunks) == 0 {
		// Belt-and-braces: should never happen, but if the AST somehow
		// produced no boundaries, fall back so the caller still gets
		// something searchable.
		return Split(content, targetTokens, overlapPct)
	}
	return chunks
}

// appendChunk records bytes [start, end) as the next chunk and advances
// *seq. Empty / whitespace-only ranges are skipped.
func appendChunk(chunks []Chunk, source []byte, start, end int, seq *int) []Chunk {
	if start >= end {
		return chunks
	}
	text := string(source[start:end])
	if EstimateTokens(text) == 0 {
		return chunks
	}
	c := Chunk{
		Text:     text,
		Seq:      *seq,
		StartPos: start,
		EndPos:   end,
	}
	*seq++
	return append(chunks, c)
}

// splitOversizeNode handles a single AST node whose text exceeds
// targetTokens. It runs the markdown chunker (which uses blank-line
// scoring) over just the node's body so the result still respects token
// budgets without splitting mid-statement when avoidable. Positions are
// rebased back into the original source.
func splitOversizeNode(source []byte, start, end, targetTokens int, overlapPct float64) []Chunk {
	body := string(source[start:end])
	pieces := Split(body, targetTokens, overlapPct)
	out := make([]Chunk, len(pieces))
	for i, p := range pieces {
		out[i] = Chunk{
			Text:     p.Text,
			Seq:      i, // caller rewrites Seq after merging
			StartPos: start + p.StartPos,
			EndPos:   start + p.EndPos,
		}
	}
	return out
}

// parseSource is a thin wrapper around tree-sitter that returns a parsed
// tree (caller must Close()) or an error. Tracks the package-level cancel
// context so a future "stop indexing" hook can interrupt long parses.
func parseSource(content, lang string) (*sitter.Tree, error) {
	parser := sitter.NewParser()
	parser.SetLanguage(languageFor(lang))
	defer parser.Close()
	return parser.ParseCtx(context.Background(), nil, []byte(content))
}
