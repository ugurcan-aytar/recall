package commands

import (
	"encoding/csv"
	"encoding/json"
	"encoding/xml"
	"fmt"
	"os"
	"strings"

	"github.com/spf13/cobra"

	"github.com/ugurcan-aytar/recall/internal/store"
)

// searchFlags captures every flag shared by `search` (and eventually vsearch
// / query). Keeping them on one struct makes it trivial to reuse.
type searchFlags struct {
	Limit       int
	Collection  string
	All         bool
	MinScore    float64
	Full        bool
	LineNumbers bool
	Explain     bool

	JSON  bool
	CSV   bool
	MD    bool
	XML   bool
	Files bool
}

var searchOpts searchFlags

var searchCmd = &cobra.Command{
	Use:   "search <query>",
	Short: "BM25 full-text search (no LLM)",
	Long: `BM25 full-text search (no LLM).

Query syntax:
    recall search rate limiter
        implicit-AND across all words; every token must appear in the doc.

    recall search '"rate limiter"'
        exact phrase match; the two words must appear adjacent.

    recall search 'authentication -JWT'
        exclude docs mentioning JWT (negation via leading '-').

    recall search '"machine learning" algorithm -redis -memcached'
        phrases + words + multiple negations combine freely.

Wrap the whole query in single quotes when it contains shell-meaningful
characters ("..." / * / ? / &), so your shell doesn't eat them before
recall sees them.`,
	Args: cobra.MinimumNArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		q := strings.Join(args, " ")

		s, err := openStore()
		if err != nil {
			return err
		}
		defer s.Close()

		limit := searchOpts.Limit
		if limit == 0 {
			limit = defaultLimitForFormat(searchOpts)
		}

		results, err := s.SearchBM25(store.SearchOptions{
			Query:      q,
			Limit:      limit,
			Collection: searchOpts.Collection,
			MinScore:   searchOpts.MinScore,
			All:        searchOpts.All,
		})
		if err != nil {
			return err
		}

		return renderResults(results, searchOpts)
	},
}

// defaultLimitForFormat matches qmd's behaviour: 5 for human output, 20 for
// machine-parseable formats.
func defaultLimitForFormat(f searchFlags) int {
	if f.JSON || f.CSV || f.XML || f.Files {
		return 20
	}
	return 5
}

// renderResults dispatches to a formatter based on which output flag is set.
// Exactly one formatter fires per invocation.
func renderResults(results []store.SearchResult, f searchFlags) error {
	switch {
	case f.JSON:
		return writeJSON(results)
	case f.CSV:
		return writeCSV(results)
	case f.XML:
		return writeXML(results)
	case f.MD:
		return writeMarkdown(results, f)
	case f.Files:
		return writeFiles(results)
	default:
		return writeText(results, f)
	}
}

type jsonResult struct {
	DocID      string  `json:"doc_id"`
	Title      string  `json:"title"`
	Path       string  `json:"path"`
	Collection string  `json:"collection"`
	Score      float64 `json:"score"`
	Snippet    string  `json:"snippet"`
}

func toJSONResults(results []store.SearchResult) []jsonResult {
	out := make([]jsonResult, len(results))
	for i, r := range results {
		out[i] = jsonResult{
			DocID:      r.DocID,
			Title:      r.Title,
			Path:       r.Path,
			Collection: r.CollectionName,
			Score:      r.Score,
			Snippet:    stripANSI(r.Snippet),
		}
	}
	return out
}

func writeJSON(results []store.SearchResult) error {
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	return enc.Encode(toJSONResults(results))
}

func writeCSV(results []store.SearchResult) error {
	w := csv.NewWriter(os.Stdout)
	defer w.Flush()
	if err := w.Write([]string{"doc_id", "title", "collection", "path", "score", "snippet"}); err != nil {
		return err
	}
	for _, r := range results {
		if err := w.Write([]string{
			r.DocID, r.Title, r.CollectionName, r.Path,
			fmt.Sprintf("%.4f", r.Score), stripANSI(r.Snippet),
		}); err != nil {
			return err
		}
	}
	return nil
}

type xmlResults struct {
	XMLName xml.Name    `xml:"results"`
	Hits    []xmlResult `xml:"result"`
}
type xmlResult struct {
	DocID      string  `xml:"doc_id"`
	Title      string  `xml:"title"`
	Path       string  `xml:"path"`
	Collection string  `xml:"collection"`
	Score      float64 `xml:"score"`
	Snippet    string  `xml:"snippet"`
}

func writeXML(results []store.SearchResult) error {
	out := xmlResults{Hits: make([]xmlResult, len(results))}
	for i, r := range results {
		out.Hits[i] = xmlResult{
			DocID: r.DocID, Title: r.Title, Path: r.Path,
			Collection: r.CollectionName, Score: r.Score,
			Snippet: stripANSI(r.Snippet),
		}
	}
	enc := xml.NewEncoder(os.Stdout)
	enc.Indent("", "  ")
	if err := enc.Encode(out); err != nil {
		return err
	}
	fmt.Println()
	return nil
}

func writeMarkdown(results []store.SearchResult, f searchFlags) error {
	for _, r := range results {
		fmt.Printf("## %s\n", r.Title)
		fmt.Printf("- path: `%s/%s`\n", r.CollectionName, r.Path)
		fmt.Printf("- doc: `#%s`\n", r.DocID)
		fmt.Printf("- score: %.2f\n\n", r.Score)
		fmt.Printf("%s\n\n", stripANSI(r.Snippet))
	}
	return nil
}

func writeFiles(results []store.SearchResult) error {
	// doc_id,score,filepath,collection — machine friendly
	for _, r := range results {
		fmt.Printf("%s,%.4f,%s/%s,%s\n",
			r.DocID, r.Score, r.CollectionName, r.Path, r.CollectionName)
	}
	return nil
}

func writeText(results []store.SearchResult, f searchFlags) error {
	if len(results) == 0 {
		fmt.Println("No results.")
		return nil
	}
	for i, r := range results {
		if i > 0 {
			fmt.Println()
		}
		title := r.Title
		if title == "" {
			title = r.Path
		}
		fmt.Printf("%s/%s  #%s  (score %.2f)\n", r.CollectionName, r.Path, r.DocID, r.Score)
		if title != "" {
			fmt.Printf("  %s\n", title)
		}
		snippet := r.Snippet
		if !colorsEnabled() {
			snippet = stripANSI(snippet)
		}
		fmt.Printf("  %s\n", snippet)
	}
	return nil
}

// stripANSI removes the bold-start / reset codes we embed in snippets.
func stripANSI(s string) string {
	s = strings.ReplaceAll(s, "\x1b[1m", "")
	s = strings.ReplaceAll(s, "\x1b[0m", "")
	return s
}

func init() {
	searchCmd.Flags().IntVarP(&searchOpts.Limit, "limit", "n", 0, "number of results (default 5, 20 for --json/--files)")
	searchCmd.Flags().StringVarP(&searchOpts.Collection, "collection", "c", "", "restrict to a collection")
	searchCmd.Flags().BoolVar(&searchOpts.All, "all", false, "return all matches (ignore -n)")
	searchCmd.Flags().Float64Var(&searchOpts.MinScore, "min-score", 0, "minimum score threshold")
	searchCmd.Flags().BoolVar(&searchOpts.Full, "full", false, "show full document content")
	searchCmd.Flags().BoolVar(&searchOpts.LineNumbers, "line-numbers", false, "add line numbers to snippets")
	searchCmd.Flags().BoolVar(&searchOpts.Explain, "explain", false, "include retrieval score traces")

	searchCmd.Flags().BoolVar(&searchOpts.JSON, "json", false, "JSON output")
	searchCmd.Flags().BoolVar(&searchOpts.CSV, "csv", false, "CSV output")
	searchCmd.Flags().BoolVar(&searchOpts.MD, "md", false, "Markdown output")
	searchCmd.Flags().BoolVar(&searchOpts.XML, "xml", false, "XML output")
	searchCmd.Flags().BoolVar(&searchOpts.Files, "files", false, "docid,score,filepath,collection")
}
