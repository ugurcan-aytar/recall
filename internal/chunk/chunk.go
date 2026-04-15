// Package chunk splits markdown / plain-text documents into overlapping
// chunks suitable for embedding.
//
// The algorithm is the one qmd uses: scan for structural break points
// (headings, code fences, blank lines, list items), accumulate ~384 tokens
// of content from the current position, then look for the best-scoring
// break point inside a 200-token window just before the hard cutoff. A
// distance-penalty formula biases cuts toward natural boundaries. 15% of
// each chunk's tail is echoed into the next chunk so context spans the seam.
//
// Code fences are always preserved: the chunker never emits a cutoff that
// lands in the middle of a ``` … ``` (or ~~~) block.
package chunk

import (
	"crypto/sha256"
	"encoding/hex"
	"regexp"
	"strings"
)

// Chunk is a slice of a document produced by the chunking algorithm.
type Chunk struct {
	Text     string
	Seq      int
	StartPos int // byte offset in original content
	EndPos   int // byte offset (exclusive) in original content
}

// ContentHash is the SHA-256 hex digest of Text. The store compares this
// against the previously stored hash so that only chunks whose content
// actually changed get re-embedded on subsequent indexing runs.
func (c Chunk) ContentHash() string {
	sum := sha256.Sum256([]byte(c.Text))
	return hex.EncodeToString(sum[:])
}

// Tuning knobs.
//
// DefaultTargetTokens history: 900 (qmd R2) → 384 (R3+, gollama's
// effective 512 n_ubatch cap) → 768 (v0.2.3, after llama-server
// subprocess removed the 512 cap) → 900 (v0.2.7, matches qmd's
// original target). A/B on a 3968-doc / 116k-chunk corpus at 900
// vs 768: chunk count dropped 13.6% (57,928 → 50,036), zero
// truncation events against nomic's 2048-token ceiling (the
// maxInputChars=4000 guard bounds us well below it), and
// retrieval-quality parity-or-slight-win across five semantic
// queries (pricing surfaces Madhavan Ramanujam, retention
// surfaces RetentionAndEngagementBonus; the rest shuffle within
// the same score cluster). Smaller index, faster embed, no
// regression — 900 shipped on that evidence.
const (
	DefaultTargetTokens = 900
	DefaultOverlapPct   = 0.15
	windowTokens        = 200
	tokensPerWord       = 1.3
)

// Base scores for each break-point type.
const (
	scoreH1        = 100
	scoreH2        = 90
	scoreH3        = 80
	scoreH4        = 70
	scoreH5        = 60
	scoreH6        = 50
	scoreCodeFence = 80
	scoreHrule     = 60
	scoreBlankLine = 20
	scoreListItem  = 5
	scoreLineBreak = 1
)

// EstimateTokens approximates token count as words × 1.3, matching qmd's
// heuristic. Empty and whitespace-only strings return 0.
func EstimateTokens(s string) int {
	words := len(strings.Fields(s))
	if words == 0 {
		return 0
	}
	// Round half up.
	return int(float64(words)*tokensPerWord + 0.5)
}

// ScoreBreakPoint applies the distance-penalty formula
//
//	finalScore = baseScore × (1 − (distance / window)² × 0.7)
//
// The ratio is clamped to [0, 1], so distances beyond the window floor at
// 30% of baseScore.
func ScoreBreakPoint(baseScore, distance, window float64) float64 {
	if window <= 0 {
		return baseScore
	}
	ratio := distance / window
	if ratio < 0 {
		ratio = 0
	}
	if ratio > 1 {
		ratio = 1
	}
	return baseScore * (1 - ratio*ratio*0.7)
}

// ---- internal machinery ----------------------------------------------------

type lineInfo struct {
	start, end int // byte range; end is exclusive, includes trailing \n when present
	tokens     int
}

type breakPoint struct {
	pos   int // byte offset of the start of the break-point line
	score int
}

type fenceRange struct {
	start, end int // byte positions: start = opening fence line start, end = closing fence line end
}

// Split slices content into overlapping [Chunk]s using the qmd-inspired
// break-point-scoring algorithm. Pass 0 for defaults.
//
//   - targetTokens: approximate upper bound per chunk (default 384)
//   - overlapPct:   fraction of each chunk's tail echoed into the next (default 0.15)
func Split(content string, targetTokens int, overlapPct float64) []Chunk {
	if content == "" {
		return nil
	}
	if targetTokens <= 0 {
		targetTokens = DefaultTargetTokens
	}
	if overlapPct <= 0 {
		// Treat 0 as "use default" so the public ChunkFile(…, 0, 0) call
		// site that just wants both defaults actually gets overlap. To
		// disable overlap explicitly, callers should pass a tiny value
		// (e.g. 1e-9) or — better — call Split with a positive but
		// negligible value.
		overlapPct = DefaultOverlapPct
	}
	if overlapPct > 0.5 {
		// Guard: overlap beyond half the chunk risks infinite loops.
		overlapPct = 0.5
	}

	lines := indexLines(content)
	cum := cumulativeTokens(lines)
	total := cum[len(lines)]

	if total <= targetTokens {
		return []Chunk{{
			Text: content, Seq: 0, StartPos: 0, EndPos: len(content),
		}}
	}

	fences := detectFences(lines, content)
	bps := scanBreakPoints(lines, content, fences)

	var chunks []Chunk
	pos := 0
	seq := 0

	for pos < len(content) {
		startLine := lineIdxAtPos(lines, pos)
		if startLine < 0 {
			break
		}

		remaining := cum[len(lines)] - cum[startLine]
		if remaining <= targetTokens {
			chunks = append(chunks, Chunk{
				Text: content[pos:], Seq: seq,
				StartPos: pos, EndPos: len(content),
			})
			break
		}

		targetLine := advanceLine(cum, startLine, targetTokens)
		windowStartLine := advanceLine(cum, startLine, targetTokens-windowTokens)

		windowStart := lines[windowStartLine].start
		targetEnd := lines[targetLine].end

		// Hunt for the best break point in the window.
		bestPos := -1
		bestScore := -1.0
		for _, bp := range bps {
			if bp.pos <= pos {
				continue
			}
			if bp.pos < windowStart || bp.pos > targetEnd {
				continue
			}
			bpLine := lineIdxAtPos(lines, bp.pos)
			if bpLine < 0 {
				continue
			}
			distance := float64(cum[targetLine+1] - cum[bpLine])
			adjusted := ScoreBreakPoint(float64(bp.score), distance, float64(windowTokens))
			if adjusted > bestScore {
				bestScore = adjusted
				bestPos = bp.pos
			}
		}

		cutoff := targetEnd
		if bestPos > pos {
			cutoff = bestPos
		}

		// Code fence protection: never slice through an open fence.
		if f := fenceContaining(fences, cutoff); f != nil {
			cutoff = f.end
		}

		if cutoff <= pos {
			cutoff = targetEnd
		}
		if cutoff > len(content) {
			cutoff = len(content)
		}

		chunks = append(chunks, Chunk{
			Text: content[pos:cutoff], Seq: seq,
			StartPos: pos, EndPos: cutoff,
		})
		seq++

		if cutoff >= len(content) {
			break
		}

		// Overlap: next chunk starts overlapPct-of-this-chunk-tokens earlier.
		chunkTokens := EstimateTokens(content[pos:cutoff])
		overlapTokens := int(float64(chunkTokens) * overlapPct)
		nextStart := rewindTo(lines, cum, cutoff, overlapTokens)

		// If the rewound position lands inside a fenced code block (which
		// would make the next chunk start mid-fence), drop the overlap for
		// this boundary. Duplicating an entire fenced block just to honour
		// the 15% rule is worse than a dry seam.
		if fenceContaining(fences, nextStart) != nil {
			nextStart = cutoff
		}
		if nextStart <= pos {
			nextStart = cutoff
		}
		pos = nextStart
	}

	return chunks
}

// indexLines splits content into line ranges. Each line's byte range
// includes its trailing newline when present.
func indexLines(content string) []lineInfo {
	var out []lineInfo
	pos := 0
	for pos < len(content) {
		nl := strings.IndexByte(content[pos:], '\n')
		var end int
		if nl < 0 {
			end = len(content)
		} else {
			end = pos + nl + 1
		}
		out = append(out, lineInfo{
			start:  pos,
			end:    end,
			tokens: EstimateTokens(content[pos:end]),
		})
		pos = end
	}
	return out
}

// cumulativeTokens returns cum where cum[i] = total tokens in lines[0..i).
// cum has length len(lines)+1; cum[len(lines)] is the grand total.
func cumulativeTokens(lines []lineInfo) []int {
	cum := make([]int, len(lines)+1)
	for i, l := range lines {
		cum[i+1] = cum[i] + l.tokens
	}
	return cum
}

// lineIdxAtPos returns the index of the line whose byte range contains pos.
// Returns -1 when pos is out of bounds.
func lineIdxAtPos(lines []lineInfo, pos int) int {
	for i, l := range lines {
		if pos >= l.start && pos < l.end {
			return i
		}
		// Handle end-of-file: pos equal to the last line's end.
		if i == len(lines)-1 && pos == l.end {
			return i
		}
	}
	return -1
}

// advanceLine returns the smallest line index i such that tokens from
// startLine through i (inclusive) reach at least n. Clamps to last line.
func advanceLine(cum []int, startLine, n int) int {
	if n <= 0 {
		return startLine
	}
	base := cum[startLine]
	for i := startLine; i < len(cum)-1; i++ {
		if cum[i+1]-base >= n {
			return i
		}
	}
	return len(cum) - 2
}

// rewindTo returns the byte position p ≤ from such that the sum of tokens
// between p and from is ≥ n. p lands on a line start; returns `from` when
// n ≤ 0.
func rewindTo(lines []lineInfo, cum []int, from, n int) int {
	if n <= 0 {
		return from
	}
	// Determine the last fully-included line.
	endLine := lineIdxAtPos(lines, from-1)
	if endLine < 0 {
		return 0
	}
	totalUpTo := cum[endLine+1]
	target := totalUpTo - n
	if target <= 0 {
		return 0
	}
	// Largest i with cum[i] ≤ target (gives maximum-starting line while
	// still reaching the overlap budget).
	for i := endLine; i >= 0; i-- {
		if cum[i] <= target {
			return lines[i].start
		}
	}
	return 0
}

// detectFences walks lines once to pair opening / closing ``` or ~~~
// markers. Unclosed fences run to end-of-content.
func detectFences(lines []lineInfo, content string) []fenceRange {
	var out []fenceRange
	openIdx := -1
	for i, l := range lines {
		if !isFenceLine(content[l.start:l.end]) {
			continue
		}
		if openIdx < 0 {
			openIdx = i
			continue
		}
		out = append(out, fenceRange{
			start: lines[openIdx].start,
			end:   l.end,
		})
		openIdx = -1
	}
	if openIdx >= 0 {
		out = append(out, fenceRange{
			start: lines[openIdx].start,
			end:   lines[len(lines)-1].end,
		})
	}
	return out
}

// fenceContaining returns the fence range strictly containing pos
// (pos between fence start and fence end, not at either boundary).
func fenceContaining(fences []fenceRange, pos int) *fenceRange {
	for i, f := range fences {
		if pos > f.start && pos < f.end {
			return &fences[i]
		}
	}
	return nil
}

func isFenceLine(rawLine string) bool {
	trim := strings.TrimLeft(rawLine, " \t")
	return strings.HasPrefix(trim, "```") || strings.HasPrefix(trim, "~~~")
}

// scanBreakPoints emits one breakPoint per eligible line. Lines inside a
// fenced code block are skipped; the fence markers themselves are emitted
// with score scoreCodeFence.
func scanBreakPoints(lines []lineInfo, content string, _ []fenceRange) []breakPoint {
	var bps []breakPoint
	inFence := false
	for i, l := range lines {
		raw := content[l.start:l.end]
		if isFenceLine(raw) {
			bps = append(bps, breakPoint{pos: l.start, score: scoreCodeFence})
			inFence = !inFence
			continue
		}
		if inFence {
			continue
		}
		trim := strings.TrimRight(strings.TrimLeft(raw, " \t"), "\r\n")
		if s := classifyLine(trim, i); s > 0 {
			bps = append(bps, breakPoint{pos: l.start, score: s})
		}
	}
	return bps
}

var numberedListRE = regexp.MustCompile(`^\d+\.\s`)

func classifyLine(trim string, lineIdx int) int {
	switch {
	case strings.HasPrefix(trim, "# "):
		return scoreH1
	case strings.HasPrefix(trim, "## "):
		return scoreH2
	case strings.HasPrefix(trim, "### "):
		return scoreH3
	case strings.HasPrefix(trim, "#### "):
		return scoreH4
	case strings.HasPrefix(trim, "##### "):
		return scoreH5
	case strings.HasPrefix(trim, "###### "):
		return scoreH6
	}
	if trim == "---" || trim == "***" || trim == "- - -" {
		return scoreHrule
	}
	if trim == "" {
		return scoreBlankLine
	}
	for _, prefix := range []string{"- ", "* ", "+ "} {
		if strings.HasPrefix(trim, prefix) {
			return scoreListItem
		}
	}
	if numberedListRE.MatchString(trim) {
		return scoreListItem
	}
	if lineIdx > 0 {
		return scoreLineBreak
	}
	return 0
}
