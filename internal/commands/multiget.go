package commands

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strings"

	"github.com/spf13/cobra"

	"github.com/ugurcan-aytar/recall/internal/store"
)

var (
	multiMaxBytes int
	multiJSON     bool
)

var multiGetCmd = &cobra.Command{
	Use:   "multi-get <pattern>",
	Short: "Batch retrieve documents by glob or comma-separated list",
	Args:  cobra.MinimumNArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		spec := strings.Join(args, " ")

		s, err := openStore()
		if err != nil {
			return err
		}
		defer s.Close()

		docs, err := resolveMultiGet(s, spec)
		if err != nil {
			return err
		}

		if multiMaxBytes > 0 {
			filtered := docs[:0]
			for _, d := range docs {
				if len(d.Content) <= multiMaxBytes {
					filtered = append(filtered, d)
				}
			}
			docs = filtered
		}

		if multiJSON {
			return writeMultiGetJSON(docs)
		}
		return writeMultiGetText(docs)
	},
}

// resolveMultiGet supports three input shapes:
//   - comma-separated list of path / #docid tokens: "a.md, b.md, #abc123"
//   - a single glob:                                  "docs/*.md"
//   - a single path or docid:                         "notes/ideas.md"
func resolveMultiGet(s *store.Store, spec string) ([]store.Document, error) {
	if strings.ContainsRune(spec, ',') {
		return resolveCommaSeparated(s, spec)
	}
	if containsGlobMeta(spec) {
		docs, err := s.MultiGetGlob(spec)
		if err != nil {
			return nil, err
		}
		return docs, nil
	}
	doc, err := s.GetDocument(spec)
	if err != nil {
		return nil, err
	}
	return []store.Document{*doc}, nil
}

func resolveCommaSeparated(s *store.Store, spec string) ([]store.Document, error) {
	var out []store.Document
	for _, raw := range strings.Split(spec, ",") {
		token := strings.TrimSpace(raw)
		if token == "" {
			continue
		}
		doc, err := s.GetDocument(token)
		if err != nil {
			if errors.Is(err, store.ErrDocumentNotFound) {
				fmt.Fprintf(os.Stderr, "warning: not found: %s\n", token)
				continue
			}
			return nil, err
		}
		out = append(out, *doc)
	}
	return out, nil
}

func containsGlobMeta(s string) bool {
	return strings.ContainsAny(s, "*?[{")
}

func writeMultiGetText(docs []store.Document) error {
	for i, d := range docs {
		if i > 0 {
			fmt.Println()
			fmt.Println("---")
			fmt.Println()
		}
		if d.Title != "" {
			fmt.Printf("# %s\n", d.Title)
		}
		fmt.Printf("# %s/%s  #%s\n\n", d.CollectionName, d.Path, d.DocID)
		fmt.Println(d.Content)
	}
	return nil
}

type multiGetJSONEntry struct {
	DocID      string `json:"doc_id"`
	Collection string `json:"collection"`
	Path       string `json:"path"`
	Title      string `json:"title"`
	Content    string `json:"content"`
}

func writeMultiGetJSON(docs []store.Document) error {
	out := make([]multiGetJSONEntry, len(docs))
	for i, d := range docs {
		out[i] = multiGetJSONEntry{
			DocID: d.DocID, Collection: d.CollectionName, Path: d.Path,
			Title: d.Title, Content: d.Content,
		}
	}
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	return enc.Encode(out)
}

func init() {
	multiGetCmd.Flags().IntVar(&multiMaxBytes, "max-bytes", 0, "skip files larger than N bytes (0 = no cap)")
	multiGetCmd.Flags().BoolVar(&multiJSON, "json", false, "JSON output")
}
