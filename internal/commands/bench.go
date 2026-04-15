// recall bench — IR quality benchmark.
//
// Takes a JSONL test file where each line declares a query and the
// set of paths (or docids) that ought to rank near the top for that
// query. For every query, runs the chosen retrieval mode, compares
// the returned top-K against the relevance set, and reports the
// classic IR metrics: precision@{1,5,10}, recall@{5,10}, MRR.
//
// qmd's bench-rerank.ts is a latency / throughput benchmark (a
// different animal); recall's bench focuses on retrieval quality,
// which is the property that moves when we tweak chunk target,
// embedding model, or reranker backend. Both benchmarks coexist
// cleanly — they answer different questions.

package commands

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"sort"
	"strings"

	"github.com/spf13/cobra"

	"github.com/ugurcan-aytar/recall/internal/embed"
	"github.com/ugurcan-aytar/recall/internal/store"
)

type benchOptions struct {
	Mode       string // bm25 | vector | hybrid
	Collection string
	TopK       int
	JSON       bool
	RerankEnab bool
}

var benchOpts benchOptions

type benchEntry struct {
	Query    string   `json:"query"`
	Relevant []string `json:"relevant"`
}

type benchResult struct {
	Query       string   `json:"query"`
	Retrieved   []string `json:"retrieved"`
	RelevantSet []string `json:"relevant"`
	PrecisionAt map[string]float64 `json:"precision_at"`
	RecallAt    map[string]float64 `json:"recall_at"`
	MRR         float64  `json:"mrr"`
}

type benchSummary struct {
	Mode        string         `json:"mode"`
	Collection  string         `json:"collection,omitempty"`
	NumQueries  int            `json:"num_queries"`
	Rerank      bool           `json:"rerank"`
	MacroAvg    map[string]float64 `json:"macro_avg"`
	PerQuery    []benchResult  `json:"per_query,omitempty"`
}

var benchCmd = &cobra.Command{
	Use:   "bench <jsonl-file>",
	Short: "Retrieval quality benchmark (precision@k, recall@k, MRR)",
	Long: `Reads a JSONL test file (one {"query": "...", "relevant": ["path", ...]} per line),
runs each query through the chosen retrieval mode, and reports the usual
IR quality metrics: precision@{1,5,10}, recall@{5,10}, and MRR.

A document counts as "retrieved correctly" when any of its identifiers
(path, absolute_path, or docid) matches an entry in the query's
"relevant" list. This makes the input file tolerant of whichever
identifier is most convenient for the person writing the test set.`,
	Args: cobra.ExactArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		entries, err := loadBenchFile(args[0])
		if err != nil {
			return fmt.Errorf("read bench file %s: %w", args[0], err)
		}
		if len(entries) == 0 {
			return fmt.Errorf("bench file %s has no query entries", args[0])
		}

		s, err := openStore()
		if err != nil {
			return err
		}
		defer s.Close()

		// Per-mode retrieval needs either an embedder (vector/hybrid)
		// or just the store (bm25). Resolve up-front so the first
		// failing mode aborts cleanly instead of running half the
		// bench and then exploding.
		var emb embed.Embedder
		if benchOpts.Mode != "bm25" {
			e, err := openEmbedder()
			if err != nil {
				return fmt.Errorf("open embedder (required for mode=%s): %w", benchOpts.Mode, err)
			}
			defer e.Close()
			emb = e
		}

		topK := benchOpts.TopK
		if topK <= 0 {
			topK = 10
		}

		results := make([]benchResult, 0, len(entries))
		kValues := []int{1, 5, 10}
		for _, e := range entries {
			retrieved, err := runBenchQuery(s, emb, e.Query, benchOpts.Mode, benchOpts.Collection, topK)
			if err != nil {
				return fmt.Errorf("query %q failed: %w", e.Query, err)
			}
			r := benchResult{
				Query:       e.Query,
				Retrieved:   retrieved,
				RelevantSet: e.Relevant,
				PrecisionAt: map[string]float64{},
				RecallAt:    map[string]float64{},
				MRR:         reciprocalRank(retrieved, e.Relevant),
			}
			for _, k := range kValues {
				key := fmt.Sprintf("@%d", k)
				r.PrecisionAt[key] = precisionAt(retrieved, e.Relevant, k)
				if k == 1 {
					continue // recall@1 is noisy; P@1 is the signal at k=1
				}
				r.RecallAt[key] = recallAt(retrieved, e.Relevant, k)
			}
			results = append(results, r)
		}

		summary := aggregate(results, benchOpts.Mode, benchOpts.Collection, benchOpts.RerankEnab)

		if benchOpts.JSON {
			summary.PerQuery = results
			enc := json.NewEncoder(os.Stdout)
			enc.SetIndent("", "  ")
			return enc.Encode(summary)
		}
		printBenchTable(summary, results)
		return nil
	},
}

func init() {
	benchCmd.Flags().StringVar(&benchOpts.Mode, "mode", "hybrid", "retrieval mode: bm25 | vector | hybrid")
	benchCmd.Flags().StringVarP(&benchOpts.Collection, "collection", "c", "", "restrict queries to this collection (default: all)")
	benchCmd.Flags().IntVar(&benchOpts.TopK, "top-k", 10, "how many results to retrieve per query (must be ≥ max metric k)")
	benchCmd.Flags().BoolVar(&benchOpts.JSON, "json", false, "emit per-query + aggregate metrics as JSON")
}

func loadBenchFile(path string) ([]benchEntry, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var out []benchEntry
	sc := bufio.NewScanner(f)
	// Generous buffer — some relevance sets can run long on realistic
	// corpora, and bufio's default 64 KB line cap is easy to hit.
	sc.Buffer(make([]byte, 0, 64*1024), 1<<20)
	line := 0
	for sc.Scan() {
		line++
		raw := strings.TrimSpace(sc.Text())
		if raw == "" || strings.HasPrefix(raw, "//") || strings.HasPrefix(raw, "#") {
			continue
		}
		var e benchEntry
		if err := json.Unmarshal([]byte(raw), &e); err != nil {
			return nil, fmt.Errorf("line %d: %w", line, err)
		}
		if strings.TrimSpace(e.Query) == "" {
			return nil, fmt.Errorf("line %d: empty query", line)
		}
		if len(e.Relevant) == 0 {
			return nil, fmt.Errorf("line %d: empty relevant list", line)
		}
		out = append(out, e)
	}
	if err := sc.Err(); err != nil && err != io.EOF {
		return nil, err
	}
	return out, nil
}

// runBenchQuery executes one query in the chosen retrieval mode and
// returns the ranked list of document identifiers. Identifiers are
// both the document's Path and its DocID so the match check in
// precision/recall can accept either.
func runBenchQuery(s *store.Store, emb embed.Embedder, q, mode, coll string, topK int) ([]string, error) {
	sopts := store.SearchOptions{
		Query:      q,
		Collection: coll,
		Limit:      topK,
	}
	switch mode {
	case "bm25":
		rows, err := s.SearchBM25(sopts)
		if err != nil {
			return nil, err
		}
		return flattenIdentifiers(rows), nil
	case "vector":
		vec, err := emb.EmbedSingle(embed.FormatQueryFor(emb.Family(), q))
		if err != nil {
			return nil, fmt.Errorf("embed query: %w", err)
		}
		rows, err := s.SearchVector(vec, sopts)
		if err != nil {
			return nil, err
		}
		return flattenIdentifiers(rows), nil
	case "hybrid":
		vec, err := emb.EmbedSingle(embed.FormatQueryFor(emb.Family(), q))
		if err != nil {
			return nil, fmt.Errorf("embed query: %w", err)
		}
		bm, err := s.SearchBM25(sopts)
		if err != nil {
			return nil, err
		}
		vs, err := s.SearchVector(vec, sopts)
		if err != nil {
			return nil, err
		}
		fused := store.FuseRRF(bm, vs, store.DefaultFusionOptions())
		if len(fused) > topK {
			fused = fused[:topK]
		}
		ids := make([]string, 0, len(fused))
		for _, r := range fused {
			ids = append(ids, r.Path, r.DocID)
		}
		return dedup(ids), nil
	default:
		return nil, fmt.Errorf("unknown mode %q — want bm25|vector|hybrid", mode)
	}
}

func flattenIdentifiers(rows []store.SearchResult) []string {
	ids := make([]string, 0, len(rows)*2)
	for _, r := range rows {
		ids = append(ids, r.Path, r.DocID)
	}
	return dedup(ids)
}

func dedup(ids []string) []string {
	seen := map[string]struct{}{}
	out := make([]string, 0, len(ids))
	for _, id := range ids {
		if id == "" {
			continue
		}
		if _, ok := seen[id]; ok {
			continue
		}
		seen[id] = struct{}{}
		out = append(out, id)
	}
	return out
}

// hit reports whether any retrieved identifier at position i (0-based)
// matches any entry in the relevance set. Matching is exact string
// compare in either direction — authors can put paths, absolute
// paths, or docids in the bench file and it'll just work.
func hit(retrieved, relevant []string, i int) bool {
	if i < 0 || i >= len(retrieved) {
		return false
	}
	for _, rel := range relevant {
		if retrieved[i] == rel {
			return true
		}
	}
	return false
}

// precisionAt is |hits in top-k| / k.
func precisionAt(retrieved, relevant []string, k int) float64 {
	if k <= 0 {
		return 0
	}
	limit := k
	if limit > len(retrieved) {
		limit = len(retrieved)
	}
	hits := 0
	for i := 0; i < limit; i++ {
		if hit(retrieved, relevant, i) {
			hits++
		}
	}
	return float64(hits) / float64(k)
}

// recallAt is |hits in top-k| / |relevant|.
func recallAt(retrieved, relevant []string, k int) float64 {
	if len(relevant) == 0 {
		return 0
	}
	limit := k
	if limit > len(retrieved) {
		limit = len(retrieved)
	}
	hits := 0
	for i := 0; i < limit; i++ {
		if hit(retrieved, relevant, i) {
			hits++
		}
	}
	return float64(hits) / float64(len(relevant))
}

// reciprocalRank returns 1 / (rank of first relevant, 1-based) or 0
// when no relevant doc appears in the retrieved list.
func reciprocalRank(retrieved, relevant []string) float64 {
	for i := range retrieved {
		if hit(retrieved, relevant, i) {
			return 1.0 / float64(i+1)
		}
	}
	return 0
}

func aggregate(results []benchResult, mode, coll string, rerank bool) benchSummary {
	s := benchSummary{
		Mode:       mode,
		Collection: coll,
		NumQueries: len(results),
		Rerank:     rerank,
		MacroAvg:   map[string]float64{},
	}
	if len(results) == 0 {
		return s
	}
	keys := []string{"P@1", "P@5", "P@10", "R@5", "R@10", "MRR"}
	accum := map[string]float64{}
	for _, r := range results {
		accum["P@1"] += r.PrecisionAt["@1"]
		accum["P@5"] += r.PrecisionAt["@5"]
		accum["P@10"] += r.PrecisionAt["@10"]
		accum["R@5"] += r.RecallAt["@5"]
		accum["R@10"] += r.RecallAt["@10"]
		accum["MRR"] += r.MRR
	}
	n := float64(len(results))
	for _, k := range keys {
		s.MacroAvg[k] = round3(accum[k] / n)
	}
	return s
}

func round3(x float64) float64 {
	return math.Round(x*1000) / 1000
}

func printBenchTable(summary benchSummary, results []benchResult) {
	fmt.Printf("recall bench — mode=%s, collection=%s, queries=%d\n\n",
		summary.Mode,
		fallbackEmpty(summary.Collection, "(all)"),
		summary.NumQueries,
	)
	fmt.Printf("  %-40s  %-5s  %-5s  %-6s  %-5s  %-6s  %-5s\n",
		"query", "P@1", "P@5", "P@10", "R@5", "R@10", "MRR")
	fmt.Println("  " + strings.Repeat("─", 88))

	rs := append([]benchResult(nil), results...)
	sort.SliceStable(rs, func(i, j int) bool { return rs[i].Query < rs[j].Query })
	for _, r := range rs {
		fmt.Printf("  %-40s  %5.2f  %5.2f  %6.2f  %5.2f  %6.2f  %5.2f\n",
			truncateDisplay(r.Query, 40),
			r.PrecisionAt["@1"],
			r.PrecisionAt["@5"],
			r.PrecisionAt["@10"],
			r.RecallAt["@5"],
			r.RecallAt["@10"],
			r.MRR,
		)
	}
	fmt.Println("  " + strings.Repeat("─", 88))
	fmt.Printf("  %-40s  %5.2f  %5.2f  %6.2f  %5.2f  %6.2f  %5.2f\n",
		"macro avg",
		summary.MacroAvg["P@1"],
		summary.MacroAvg["P@5"],
		summary.MacroAvg["P@10"],
		summary.MacroAvg["R@5"],
		summary.MacroAvg["R@10"],
		summary.MacroAvg["MRR"],
	)
}

func truncateDisplay(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n-1] + "…"
}

func fallbackEmpty(s, fb string) string {
	if s == "" {
		return fb
	}
	return s
}
