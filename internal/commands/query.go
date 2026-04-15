package commands

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strings"
	"sync"

	"github.com/spf13/cobra"

	"github.com/ugurcan-aytar/recall/internal/embed"
	"github.com/ugurcan-aytar/recall/internal/expand"
	"github.com/ugurcan-aytar/recall/internal/llm"
	"github.com/ugurcan-aytar/recall/internal/rerank"
	"github.com/ugurcan-aytar/recall/internal/store"
)

var (
	queryOpts          searchFlags
	queryChunkStrategy string
	queryExpand        bool
	queryHyde          bool
	queryIntent        string
	queryRerank        bool
	queryRerankTopN    int
)

// originalListBonus is the extra weight applied to the user's literal
// query when same-side rank lists are merged via store.MergeRankLists.
// qmd uses 1.0 (= "first list counts twice"); recall mirrors that for
// the expansion path so aggressive variants can't shove the user's
// own phrasing off the result list.
const originalListBonus = 1.0

var queryCmd = &cobra.Command{
	Use:   "query <query>",
	Short: "Hybrid search: BM25 + vector + RRF fusion",
	Args:  cobra.MinimumNArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		q := strings.Join(args, " ")

		s, err := openStore()
		if err != nil {
			return err
		}
		defer s.Close()

		// If no embeddings exist yet, fall back to BM25 silently — this is
		// the explicit graceful-degradation rule from CLAUDE.md and ROADMAP
		// "Lazy model loading" #4.
		embedCount, err := s.EmbeddingCount()
		if err != nil {
			return err
		}
		if embedCount == 0 {
			return runBM25Fallback(s, q)
		}

		emb, err := openEmbedder()
		if err != nil {
			if errors.Is(err, embed.ErrLocalEmbedderNotCompiled) {
				fmt.Fprintln(os.Stderr,
					"warning: vector backend unavailable; falling back to BM25 only")
				return runBM25Fallback(s, q)
			}
			return err
		}
		defer emb.Close()

		queryVec, err := embedQueryCached(emb, q)
		if err != nil {
			return fmt.Errorf("embed query: %w", err)
		}

		// Run BM25 and vector concurrently per ROADMAP performance note #1.
		oversample := queryOpts.Limit * 4
		if oversample <= 0 {
			oversample = 40
		}

		// --expand fans the original query out into LLM-generated lex /
		// vec variants; --hyde embeds the model's hypothetical answer
		// passages and uses each as another vector query. Both flags
		// share one expand.Expand call so a user running --expand
		// --hyde pays for one model invocation, not two.
		bm25Queries := []string{q}
		vecQueries := []string{q}
		var hydeVectors [][]float32
		if queryExpand || queryHyde {
			expansion, err := runExpansion(s, q)
			if err != nil {
				return err
			}
			if expansion != nil {
				if queryExpand {
					bm25Queries = append(bm25Queries, expansion.Lex...)
					vecQueries = append(vecQueries, expansion.Vec...)
				}
				if queryHyde {
					// Each HyDE passage is a hypothetical document; embed
					// it as a document so it lands in the same vector
					// space as the real corpus, then SearchVector with
					// each one as an extra query.
					family := emb.Family()
					for _, passage := range expansion.Hyde {
						v, hyErr := emb.EmbedSingle(embed.FormatDocumentFor(family, "", passage))
						if hyErr != nil {
							fmt.Fprintln(os.Stderr,
								"warning: HyDE embed failed for one passage — "+hyErr.Error())
							continue
						}
						hydeVectors = append(hydeVectors, v)
					}
				}
				if queryOpts.Explain {
					fmt.Fprintf(os.Stderr,
						"[explain] expansion produced %d lex + %d vec + %d hyde\n",
						len(expansion.Lex), len(expansion.Vec), len(expansion.Hyde))
				}
			}
		}

		bm25Lists := make([][]store.SearchResult, len(bm25Queries))
		vecLists := make([][]store.SearchResult, len(vecQueries)+len(hydeVectors))
		bm25Errs := make([]error, len(bm25Queries))
		vecErrs := make([]error, len(vecQueries)+len(hydeVectors))

		var wg sync.WaitGroup
		for i, bq := range bm25Queries {
			i, bq := i, bq
			wg.Add(1)
			go func() {
				defer wg.Done()
				bm25Lists[i], bm25Errs[i] = s.SearchBM25(store.SearchOptions{
					Query: bq, Limit: oversample, Collection: queryOpts.Collection,
				})
			}()
		}
		for i, vq := range vecQueries {
			i, vq := i, vq
			wg.Add(1)
			go func() {
				defer wg.Done()
				var v []float32
				if vq == q {
					v = queryVec
				} else {
					var embErr error
					v, embErr = embedQueryCached(emb, vq)
					if embErr != nil {
						vecErrs[i] = fmt.Errorf("embed expanded query %q: %w", vq, embErr)
						return
					}
				}
				vecLists[i], vecErrs[i] = s.SearchVector(v, store.SearchOptions{
					Query: vq, Limit: oversample, Collection: queryOpts.Collection,
				})
			}()
		}
		// HyDE vectors join the same vector-search fan-out — each one
		// is embedded against the doc-side prompt format and run
		// through SearchVector exactly like the real-query embedding.
		baseVecCount := len(vecQueries)
		for hi, v := range hydeVectors {
			hi, v := hi, v
			wg.Add(1)
			go func() {
				defer wg.Done()
				vecLists[baseVecCount+hi], vecErrs[baseVecCount+hi] = s.SearchVector(v,
					store.SearchOptions{Query: q, Limit: oversample, Collection: queryOpts.Collection})
			}()
		}
		wg.Wait()
		for _, err := range bm25Errs {
			if err != nil {
				return fmt.Errorf("bm25: %w", err)
			}
		}
		for _, err := range vecErrs {
			if err != nil {
				return fmt.Errorf("vector: %w", err)
			}
		}

		// Same-side merge — 1 list pass-through, N lists rank-fuse
		// with a bonus on the original (first) entry.
		bm25Results := store.MergeRankLists(bm25Lists, store.DefaultFusionOptions(), originalListBonus)
		vecResults := store.MergeRankLists(vecLists, store.DefaultFusionOptions(), originalListBonus)

		fused := store.FuseRRF(bm25Results, vecResults, store.DefaultFusionOptions())

		// --rerank: send the top-N RRF results through a yes/no
		// relevance LLM (default Qwen2.5-1.5B-Instruct), then sort
		// by rerank score. Position-aware blending lands in feature
		// #5; for v0.2.0 the rerank score takes precedence over the
		// RRF rank for any candidate the LLM scored.
		if queryRerank {
			fused = applyRerank(fused, q)
		}

		limit := queryOpts.Limit
		if limit == 0 {
			limit = defaultLimitForFormat(queryOpts)
		}
		if !queryOpts.All && len(fused) > limit {
			fused = fused[:limit]
		}
		// Fold MinScore on top of the adaptive floor.
		if queryOpts.MinScore > 0 {
			cut := 0
			for cut < len(fused) && fused[cut].FusedScore >= queryOpts.MinScore {
				cut++
			}
			fused = fused[:cut]
		}

		return renderFused(fused, queryOpts)
	},
}

func runBM25Fallback(s *store.Store, q string) error {
	limit := queryOpts.Limit
	if limit == 0 {
		limit = defaultLimitForFormat(queryOpts)
	}
	results, err := s.SearchBM25(store.SearchOptions{
		Query: q, Limit: limit, Collection: queryOpts.Collection,
		MinScore: queryOpts.MinScore, All: queryOpts.All,
	})
	if err != nil {
		return err
	}
	if err := renderResults(results, queryOpts); err != nil {
		return err
	}
	if queryOpts.Explain && !queryOpts.JSON {
		fmt.Println()
		fmt.Println("[explain] no embeddings indexed; RRF fusion skipped, BM25-only results above.")
		fmt.Println("[explain] run `recall embed` (with the local model or RECALL_EMBED_PROVIDER) to enable hybrid search.")
	}
	return nil
}

// renderFused shares the formatter machinery from search.go but overrides
// the score column with the fused value and (when --explain) shows a
// per-result trace.
func renderFused(results []store.FusedResult, f searchFlags) error {
	plain := make([]store.SearchResult, len(results))
	for i, r := range results {
		plain[i] = r.SearchResult
		plain[i].Score = r.FusedScore
	}

	switch {
	case f.JSON:
		return writeFusedJSON(results)
	default:
		if err := renderResults(plain, f); err != nil {
			return err
		}
		if f.Explain {
			fmt.Println()
			for i, r := range results {
				fmt.Printf("[explain %d] %s/%s  rrf=%.4f bonus=%.4f floor=%.4f bm25_rank=%d vec_rank=%d\n",
					i+1, r.CollectionName, r.Path,
					r.Trace.RRFOnly, r.Trace.BonusUsed, r.Trace.FloorUsed,
					r.Trace.BM25Rank, r.Trace.VecRank,
				)
			}
		}
		return nil
	}
}

type fusedJSONRow struct {
	DocID      string  `json:"doc_id"`
	Title      string  `json:"title"`
	Path       string  `json:"path"`
	Collection string  `json:"collection"`
	Score      float64 `json:"score"`
	Snippet    string  `json:"snippet"`
	Explain    *struct {
		BM25Rank   int     `json:"bm25_rank"`
		BM25Score  float64 `json:"bm25_score"`
		VecRank    int     `json:"vec_rank"`
		VecScore   float64 `json:"vec_score"`
		RRFOnly    float64 `json:"rrf_only"`
		Bonus      float64 `json:"bonus"`
		Floor      float64 `json:"floor"`
	} `json:"explain,omitempty"`
}

func writeFusedJSON(results []store.FusedResult) error {
	out := make([]fusedJSONRow, len(results))
	for i, r := range results {
		out[i] = fusedJSONRow{
			DocID:      r.DocID,
			Title:      r.Title,
			Path:       r.Path,
			Collection: r.CollectionName,
			Score:      r.FusedScore,
			Snippet:    stripANSI(r.Snippet),
		}
		if queryOpts.Explain {
			out[i].Explain = &struct {
				BM25Rank   int     `json:"bm25_rank"`
				BM25Score  float64 `json:"bm25_score"`
				VecRank    int     `json:"vec_rank"`
				VecScore   float64 `json:"vec_score"`
				RRFOnly    float64 `json:"rrf_only"`
				Bonus      float64 `json:"bonus"`
				Floor      float64 `json:"floor"`
			}{
				BM25Rank: r.Trace.BM25Rank, BM25Score: r.Trace.BM25Score,
				VecRank: r.Trace.VecRank, VecScore: r.Trace.VecScore,
				RRFOnly: r.Trace.RRFOnly, Bonus: r.Trace.BonusUsed,
				Floor: r.Trace.FloorUsed,
			}
		}
	}
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	return enc.Encode(out)
}

// applyRerank dispatches the top-N RRF candidates to the reranker
// LLM and returns a fused-result slice resorted by rerank score.
// Falls through to the input when the reranker model isn't
// available — same graceful-degradation contract as runExpansion.
//
// FusedScore is overwritten with rerank.Score for the candidates
// that were actually scored. Untouched candidates keep their RRF
// scores so the position-aware blender (feature #5) can fuse them
// with rank later.
func applyRerank(fused []store.FusedResult, q string) []store.FusedResult {
	if len(fused) == 0 {
		return fused
	}
	rr, err := openReranker()
	if err != nil {
		if errors.Is(err, llm.ErrLocalRerankerNotAvailable) {
			fmt.Fprintln(os.Stderr, "warning: --rerank requested but reranker unavailable — "+err.Error())
			return fused
		}
		if os.IsNotExist(err) || strings.Contains(err.Error(), "reranker model not found") {
			fmt.Fprintln(os.Stderr, "warning: --rerank requested but model missing — "+err.Error())
			return fused
		}
		fmt.Fprintln(os.Stderr, "warning: rerank open failed — "+err.Error())
		return fused
	}
	defer rr.Close()

	topN := queryRerankTopN
	if topN <= 0 {
		topN = rerank.DefaultTopN
	}

	candidates := make([]store.SearchResult, len(fused))
	for i, f := range fused {
		candidates[i] = f.SearchResult
	}
	scored, err := rerank.Rerank(context.Background(), rr, q, candidates, rerank.Options{TopN: topN})
	if err != nil {
		fmt.Fprintln(os.Stderr, "warning: rerank failed — "+err.Error())
		return fused
	}

	// Position-aware blend (feature #5): each candidate's final
	// score is BlendBands-weighted between RRF rank and the binary
	// reranker verdict. Top-3 RRF positions get a 75/25 weighting,
	// ranks 4-10 get 60/40, ranks 11+ get 40/60. This protects
	// strong RRF hits when the reranker is uncertain while still
	// letting a confident reranker promote deep candidates.
	blended := rerank.PositionAwareBlend(scored, rerank.DefaultBlendBands)

	// Map back to FusedResult, keeping the original RRF trace so
	// `--explain` still has bm25/vec rank info per row.
	traceByKey := map[string]store.FusionTrace{}
	for _, f := range fused {
		traceByKey[f.DocID] = f.Trace
	}
	out := make([]store.FusedResult, len(blended))
	for i, s := range blended {
		out[i] = store.FusedResult{
			SearchResult: s.Result,
			FusedScore:   s.BlendedScore,
			Trace:        traceByKey[s.Result.DocID],
		}
	}
	return out
}

// runExpansion drives the LLM expansion model. Errors that mean
// "model isn't installed" are downgraded to a stderr warning and a
// nil return so the rest of `recall query` continues with the
// original query unchanged. Hard errors (model present but
// generation failed) propagate so the user notices.
//
// Auto-intent: when --intent isn't set explicitly AND a single
// collection is targeted AND that collection has a context blurb,
// the blurb gets passed as the intent line so the LLM can
// disambiguate domain-specific queries. qmd skipped this for HyDE
// generation and the resulting hypothetical passages were noticeably
// less on-topic; recall fixes that here.
func runExpansion(s *store.Store, q string) (*expand.Expanded, error) {
	gen, err := openGenerator()
	if err != nil {
		if errors.Is(err, llm.ErrLocalGeneratorNotCompiled) {
			fmt.Fprintln(os.Stderr,
				"warning: --expand / --hyde requires the embed_llama build; rebuild from source or drop the flag")
			return nil, nil
		}
		if os.IsNotExist(err) || strings.Contains(err.Error(), "expansion model not found") {
			fmt.Fprintln(os.Stderr, "warning: --expand / --hyde requested but model missing — "+err.Error())
			return nil, nil
		}
		return nil, err
	}
	defer gen.Close()

	intent := queryIntent
	if intent == "" && queryOpts.Collection != "" && s != nil {
		if c, lookupErr := s.GetCollectionByName(queryOpts.Collection); lookupErr == nil &&
			c != nil && c.Context != "" {
			intent = c.Context
		}
	}

	return expand.Expand(gen, q, expand.Options{
		Intent:     intent,
		IncludeLex: true,
	})
}

func init() {
	queryCmd.Flags().IntVarP(&queryOpts.Limit, "limit", "n", 0, "number of results (default 5, 20 for --json/--files)")
	queryCmd.Flags().StringVarP(&queryOpts.Collection, "collection", "c", "", "restrict to a collection")
	queryCmd.Flags().BoolVar(&queryOpts.All, "all", false, "return all matches (ignore -n)")
	queryCmd.Flags().Float64Var(&queryOpts.MinScore, "min-score", 0, "minimum fused-score threshold (in addition to adaptive floor)")
	queryCmd.Flags().BoolVar(&queryOpts.Full, "full", false, "show full document content")
	queryCmd.Flags().BoolVar(&queryOpts.Explain, "explain", false, "show RRF / bonus / floor trace per result")
	queryCmd.Flags().BoolVar(&queryExpand, "expand", false,
		"expand the query into LLM-generated lex + vec variants (requires `recall models download --expansion`)")
	queryCmd.Flags().BoolVar(&queryHyde, "hyde", false,
		"generate a hypothetical answer passage and use its embedding as an extra vector query (requires `recall models download --expansion`)")
	queryCmd.Flags().StringVar(&queryIntent, "intent", "",
		"optional one-line intent passed to the expansion LLM (e.g. \"web performance\"); auto-fills from the active collection's context when unset and a single -c collection is targeted")
	queryCmd.Flags().BoolVar(&queryRerank, "rerank", false,
		"rerank the top-N RRF results through a binary-relevance LLM (requires `recall models download --reranker`)")
	queryCmd.Flags().IntVar(&queryRerankTopN, "rerank-top-n", 0,
		"how many top RRF candidates to rerank (default 30); ignored without --rerank")

	queryCmd.Flags().BoolVar(&queryOpts.JSON, "json", false, "JSON output")
	queryCmd.Flags().BoolVar(&queryOpts.CSV, "csv", false, "CSV output")
	queryCmd.Flags().BoolVar(&queryOpts.MD, "md", false, "Markdown output")
	queryCmd.Flags().BoolVar(&queryOpts.XML, "xml", false, "XML output")
	queryCmd.Flags().BoolVar(&queryOpts.Files, "files", false, "docid,score,filepath,collection")
	queryCmd.Flags().StringVar(&queryChunkStrategy, "chunk-strategy", "auto",
		"informational: chunk strategy used by the indexer (auto|regex|ast). Re-chunking happens via `recall embed -f`.")
}
