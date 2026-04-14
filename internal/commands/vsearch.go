package commands

import (
	"errors"
	"fmt"
	"strings"

	"github.com/spf13/cobra"

	"github.com/ugurcan-aytar/recall/internal/embed"
	"github.com/ugurcan-aytar/recall/internal/store"
)

var vsearchOpts searchFlags

var vsearchCmd = &cobra.Command{
	Use:   "vsearch <query>",
	Short: "Vector semantic search (no reranking)",
	Args:  cobra.MinimumNArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		q := strings.Join(args, " ")

		s, err := openStore()
		if err != nil {
			return err
		}
		defer s.Close()

		// Lazy model load: vsearch is the first non-stub user that needs an
		// embedder. BM25-only commands never reach this code path.
		emb, err := openEmbedder()
		if err != nil {
			if errors.Is(err, embed.ErrLocalEmbedderNotCompiled) {
				return fmt.Errorf("vector search requires the local embedder; "+
					"rebuild with -tags 'sqlite_fts5 embed_llama' (see CLAUDE.md): %w", err)
			}
			return err
		}
		defer emb.Close()

		vec, err := embedQueryCached(emb, q)
		if err != nil {
			return fmt.Errorf("embed query: %w", err)
		}

		limit := vsearchOpts.Limit
		if limit == 0 {
			limit = defaultLimitForFormat(vsearchOpts)
		}

		results, err := s.SearchVector(vec, store.SearchOptions{
			Query:      q,
			Limit:      limit,
			Collection: vsearchOpts.Collection,
			MinScore:   vsearchOpts.MinScore,
			All:        vsearchOpts.All,
		})
		if err != nil {
			return err
		}
		return renderResults(results, vsearchOpts)
	},
}

func init() {
	vsearchCmd.Flags().IntVarP(&vsearchOpts.Limit, "limit", "n", 0, "number of results (default 5, 20 for --json/--files)")
	vsearchCmd.Flags().StringVarP(&vsearchOpts.Collection, "collection", "c", "", "restrict to a collection")
	vsearchCmd.Flags().BoolVar(&vsearchOpts.All, "all", false, "return all matches (ignore -n)")
	vsearchCmd.Flags().Float64Var(&vsearchOpts.MinScore, "min-score", 0, "minimum score threshold")
	vsearchCmd.Flags().BoolVar(&vsearchOpts.Full, "full", false, "show full document content")
	vsearchCmd.Flags().BoolVar(&vsearchOpts.Explain, "explain", false, "include retrieval score traces")
	vsearchCmd.Flags().BoolVar(&vsearchOpts.JSON, "json", false, "JSON output")
	vsearchCmd.Flags().BoolVar(&vsearchOpts.CSV, "csv", false, "CSV output")
	vsearchCmd.Flags().BoolVar(&vsearchOpts.MD, "md", false, "Markdown output")
	vsearchCmd.Flags().BoolVar(&vsearchOpts.XML, "xml", false, "XML output")
	vsearchCmd.Flags().BoolVar(&vsearchOpts.Files, "files", false, "docid,score,filepath,collection")
}
