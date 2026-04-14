package chunk

import (
	"math"
	"strings"
	"testing"
)

// generateLorem returns approxWords words of filler text with realistic
// spacing so EstimateTokens produces ~approxWords × 1.3 tokens.
func generateLorem(approxWords int) string {
	word := "lorem ipsum dolor sit amet consectetur adipiscing elit sed do"
	parts := strings.Fields(word)
	var b strings.Builder
	for i := 0; i < approxWords; i++ {
		if i > 0 {
			if i%12 == 0 {
				b.WriteByte('\n')
			} else {
				b.WriteByte(' ')
			}
		}
		b.WriteString(parts[i%len(parts)])
	}
	return b.String()
}

// almostEqual compares floats within a small epsilon.
func almostEqual(a, b, eps float64) bool {
	return math.Abs(a-b) < eps
}

// ---------------------------------------------------------------------------
// R2.3 required tests
// ---------------------------------------------------------------------------

func TestChunkBasic(t *testing.T) {
	// ~1800 words ⇒ ~2340 tokens ⇒ must produce multiple chunks.
	body := generateLorem(1800)
	chunks := Split(body, 0, 0)
	if len(chunks) < 2 {
		t.Fatalf("got %d chunks, want >= 2", len(chunks))
	}
	for _, c := range chunks {
		if EstimateTokens(c.Text) == 0 {
			t.Errorf("empty chunk: %+v", c)
		}
	}
}

func TestChunkShortDoc(t *testing.T) {
	// 100 words ⇒ ~130 tokens ⇒ single chunk.
	body := "# Title\n\n" + generateLorem(100)
	chunks := Split(body, 0, 0)
	if len(chunks) != 1 {
		t.Fatalf("short doc produced %d chunks, want 1", len(chunks))
	}
	if chunks[0].Text != body {
		t.Errorf("chunk text mismatch")
	}
	if chunks[0].StartPos != 0 || chunks[0].EndPos != len(body) {
		t.Errorf("positions = [%d,%d), want [0,%d)", chunks[0].StartPos, chunks[0].EndPos, len(body))
	}
}

func TestChunkHeadingPriority(t *testing.T) {
	// Blank line at distance 0, H1 at distance ~window/2. H1's base 100 + small
	// penalty still beats blank's 20 + zero penalty.
	windowScore := ScoreBreakPoint(scoreH1, float64(windowTokens)/2, float64(windowTokens))
	blankScore := ScoreBreakPoint(scoreBlankLine, 0, float64(windowTokens))
	if windowScore <= blankScore {
		t.Fatalf("H1 at half window (%.2f) should outrank blank at 0 (%.2f)", windowScore, blankScore)
	}
}

func TestChunkH2vsBlankLine(t *testing.T) {
	// H2 at distance 100 (from cutoff) vs blank at 50 — H2's larger base
	// score should still win after the distance penalty is applied.
	// H2 adjusted: 90 × (1 − (100/200)² × 0.7) = 90 × 0.825 = 74.25
	// Blank adjusted: 20 × (1 − (50/200)² × 0.7) = 20 × (1 − 0.04375) = 20 × 0.95625 = 19.125
	h2 := ScoreBreakPoint(scoreH2, 100, 200)
	blank := ScoreBreakPoint(scoreBlankLine, 50, 200)
	if h2 <= blank {
		t.Fatalf("H2 (%.2f) should outrank blank (%.2f)", h2, blank)
	}
}

func TestChunkCodeFenceProtection(t *testing.T) {
	// Build a doc where the natural cutoff lands mid-code-fence.
	// Prefix (small) + long code fence (> target) + tail.
	prefix := generateLorem(400)
	var code strings.Builder
	code.WriteString("```go\n")
	for i := 0; i < 200; i++ {
		code.WriteString("func f()" + strings.Repeat("x", 5) + " { return nil }\n")
	}
	code.WriteString("```\n")
	tail := "\n\n" + generateLorem(100)
	body := "# Doc\n\n" + prefix + "\n\n" + code.String() + tail

	chunks := Split(body, 800, 0.15)
	if len(chunks) < 1 {
		t.Fatalf("no chunks")
	}
	// Every chunk must contain a balanced number of fence markers.
	for i, c := range chunks {
		count := strings.Count(c.Text, "```")
		if count%2 != 0 {
			t.Errorf("chunk %d splits through a code fence (%d ``` markers)", i, count)
		}
	}
}

func TestChunkOverlap(t *testing.T) {
	body := generateLorem(2000)
	chunks := Split(body, 800, 0.15)
	if len(chunks) < 2 {
		t.Fatalf("need >= 2 chunks for overlap check, got %d", len(chunks))
	}
	for i := 1; i < len(chunks); i++ {
		prev := chunks[i-1]
		curr := chunks[i]
		if curr.StartPos >= prev.EndPos {
			t.Errorf("chunk %d does not overlap prev: prev=[%d,%d) curr=[%d,%d)",
				i, prev.StartPos, prev.EndPos, curr.StartPos, curr.EndPos)
			continue
		}
		// The overlap region from prev should appear verbatim at the start
		// of curr.
		overlapLen := prev.EndPos - curr.StartPos
		if overlapLen <= 0 {
			t.Errorf("non-positive overlap len on chunk %d", i)
			continue
		}
		prevTail := prev.Text[len(prev.Text)-overlapLen:]
		currHead := curr.Text[:overlapLen]
		if prevTail != currHead {
			t.Errorf("chunk %d overlap text mismatch", i)
		}
		// Overlap should be roughly 15% of prev's tokens.
		prevTokens := EstimateTokens(prev.Text)
		overlapTokens := EstimateTokens(prevTail)
		// Generous bounds: between 8% and 25%.
		ratio := float64(overlapTokens) / float64(prevTokens)
		if ratio < 0.08 || ratio > 0.25 {
			t.Errorf("chunk %d overlap ratio = %.2f (want ~0.15)", i, ratio)
		}
	}
}

func TestChunkNoBreakPoints(t *testing.T) {
	// A giant single blob with no headings, no blank lines, no lists.
	// Built as one long run-on "line" split only by single spaces.
	body := strings.Repeat("word ", 1500) // ~1500 words → ~1950 tokens
	chunks := Split(body, 900, 0.15)
	if len(chunks) == 0 {
		t.Fatalf("zero chunks for large input")
	}
	// All content must be covered.
	var total strings.Builder
	for i, c := range chunks {
		if i == 0 {
			total.WriteString(c.Text)
			continue
		}
		// subsequent chunks may start with overlap that we already have
		prev := chunks[i-1]
		overlapLen := prev.EndPos - c.StartPos
		if overlapLen < 0 {
			overlapLen = 0
		}
		total.WriteString(c.Text[overlapLen:])
	}
	if total.String() != body {
		t.Errorf("concatenated chunks do not reconstruct body")
	}
}

func TestChunkGiantCodeBlock(t *testing.T) {
	// One code block of ~1500 tokens, wrapped in short preamble / tail.
	preamble := "# Doc\n\nSome intro paragraph.\n\n"
	var code strings.Builder
	code.WriteString("```go\n")
	for i := 0; i < 1300; i++ {
		code.WriteString("v" + strings.Repeat("o", 3) + " := 1\n")
	}
	code.WriteString("```\n")
	body := preamble + code.String() + "\nAfter code.\n"

	chunks := Split(body, 900, 0.15)
	if len(chunks) == 0 {
		t.Fatal("no chunks")
	}
	// Exactly one chunk should contain the full fence pair unbroken.
	hasWholeFence := 0
	for _, c := range chunks {
		if strings.Contains(c.Text, "```go\n") && strings.Contains(c.Text[strings.Index(c.Text, "```go\n")+1:], "```") {
			hasWholeFence++
		}
	}
	if hasWholeFence < 1 {
		t.Errorf("no chunk contains the whole code block intact")
	}
	for i, c := range chunks {
		if strings.Count(c.Text, "```")%2 != 0 {
			t.Errorf("chunk %d has unbalanced fence markers", i)
		}
	}
}

func TestChunkScoreFormula(t *testing.T) {
	// finalScore = baseScore × (1 − (distance/window)² × 0.7)
	cases := []struct {
		base, dist, window, want float64
	}{
		{100, 0, 200, 100},                    // no penalty
		{100, 200, 200, 30},                   // full penalty: 1 − 0.7 = 0.3
		{100, 100, 200, 82.5},                 // 1 − 0.25 × 0.7 = 0.825
		{90, 50, 200, 86.0625},                // 1 − 0.0625 × 0.7 = 0.95625
		{80, 150, 200, 80 * (1 - 0.5625*0.7)}, // (150/200)² = 0.5625
		{20, 500, 200, 20 * 0.3},              // distance beyond window clamps at 30%
		{60, 0, 0, 60},                        // window 0 ⇒ no penalty
	}
	for _, c := range cases {
		got := ScoreBreakPoint(c.base, c.dist, c.window)
		if !almostEqual(got, c.want, 1e-9) {
			t.Errorf("ScoreBreakPoint(%.1f, %.1f, %.1f) = %.6f, want %.6f",
				c.base, c.dist, c.window, got, c.want)
		}
	}
}

func TestChunkSequenceNumbers(t *testing.T) {
	body := generateLorem(3000) // comfortably more than 2 chunks
	chunks := Split(body, 700, 0.15)
	if len(chunks) < 3 {
		t.Fatalf("want >= 3 chunks for seq test, got %d", len(chunks))
	}
	for i, c := range chunks {
		if c.Seq != i {
			t.Errorf("chunk %d has Seq=%d, want %d", i, c.Seq, i)
		}
	}
}

func TestChunkPositions(t *testing.T) {
	body := generateLorem(2500)
	chunks := Split(body, 700, 0.15)
	for i, c := range chunks {
		if c.StartPos < 0 || c.EndPos > len(body) || c.StartPos >= c.EndPos {
			t.Errorf("chunk %d has invalid range [%d,%d)", i, c.StartPos, c.EndPos)
			continue
		}
		if body[c.StartPos:c.EndPos] != c.Text {
			t.Errorf("chunk %d text does not match its declared byte range", i)
		}
	}
}

// ---------------------------------------------------------------------------
// Auxiliary helpers — small coverage on internal pieces we rely on.
// ---------------------------------------------------------------------------

func TestEstimateTokens(t *testing.T) {
	cases := []struct {
		in   string
		want int
	}{
		{"", 0},
		{"   ", 0},
		{"hello", 1},                                      // 1 word × 1.3 → 1 (rounded)
		{"hello world", 3},                                // 2 × 1.3 = 2.6 → 3
		{"one two three four five six seven eight", 10},   // 8 × 1.3 = 10.4 → 10
	}
	for _, c := range cases {
		got := EstimateTokens(c.in)
		if got != c.want {
			t.Errorf("EstimateTokens(%q) = %d, want %d", c.in, got, c.want)
		}
	}
}

func TestClassifyLine(t *testing.T) {
	cases := []struct {
		in       string
		idx      int
		wantMin  int
		wantMax  int // same as wantMin when exact
	}{
		{"# Title", 0, scoreH1, scoreH1},
		{"## Sub", 0, scoreH2, scoreH2},
		{"### H3", 0, scoreH3, scoreH3},
		{"#### H4", 0, scoreH4, scoreH4},
		{"##### H5", 0, scoreH5, scoreH5},
		{"###### H6", 0, scoreH6, scoreH6},
		{"---", 0, scoreHrule, scoreHrule},
		{"***", 0, scoreHrule, scoreHrule},
		{"", 0, scoreBlankLine, scoreBlankLine},
		{"- item", 0, scoreListItem, scoreListItem},
		{"* item", 0, scoreListItem, scoreListItem},
		{"+ item", 0, scoreListItem, scoreListItem},
		{"1. numbered", 0, scoreListItem, scoreListItem},
		{"regular text", 0, 0, 0},         // first line, not a break point
		{"regular text", 5, scoreLineBreak, scoreLineBreak},
	}
	for _, c := range cases {
		got := classifyLine(c.in, c.idx)
		if got < c.wantMin || got > c.wantMax {
			t.Errorf("classifyLine(%q, idx=%d) = %d, want in [%d,%d]",
				c.in, c.idx, got, c.wantMin, c.wantMax)
		}
	}
}

func TestContentHashDeterministic(t *testing.T) {
	a := Chunk{Text: "abc"}
	b := Chunk{Text: "abc"}
	c := Chunk{Text: "abd"}
	if a.ContentHash() != b.ContentHash() {
		t.Error("identical text produced different hashes")
	}
	if a.ContentHash() == c.ContentHash() {
		t.Error("different text produced identical hashes")
	}
}

func TestChunkEmptyContent(t *testing.T) {
	if chunks := Split("", 0, 0); chunks != nil {
		t.Errorf("empty content should return nil, got %+v", chunks)
	}
}

func TestDetectFencesUnclosed(t *testing.T) {
	// An unclosed fence should still be detected and kept as one range so
	// the chunker doesn't try to split inside it.
	body := "intro\n```go\nfn a() {}\nfn b() {}\n"
	lines := indexLines(body)
	fences := detectFences(lines, body)
	if len(fences) != 1 {
		t.Fatalf("got %d fences, want 1", len(fences))
	}
	if fences[0].end != len(body) {
		t.Errorf("unclosed fence should run to EOF")
	}
}
