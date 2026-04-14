package commands

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/spf13/cobra"

	"github.com/ugurcan-aytar/recall/internal/store"
)

var (
	getFromLine int
	getMaxLines int
	getJSON     bool
)

var getCmd = &cobra.Command{
	Use:   "get <path|#docid>[:line]",
	Short: "Retrieve a single document",
	Args:  cobra.ExactArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		spec, line := parseGetSpec(args[0])
		if line > 0 && getFromLine == 0 {
			getFromLine = line
		}

		s, err := openStore()
		if err != nil {
			return err
		}
		defer s.Close()

		doc, err := s.GetDocument(spec)
		if err != nil {
			if errors.Is(err, store.ErrDocumentNotFound) {
				return suggestFallback(s, spec)
			}
			return err
		}

		body := sliceLines(doc.Content, getFromLine, getMaxLines)

		if getJSON {
			enc := json.NewEncoder(os.Stdout)
			enc.SetIndent("", "  ")
			return enc.Encode(map[string]any{
				"doc_id":     doc.DocID,
				"collection": doc.CollectionName,
				"path":       doc.Path,
				"title":      doc.Title,
				"content":    body,
			})
		}

		if doc.Title != "" {
			fmt.Printf("# %s\n", doc.Title)
		}
		fmt.Printf("# %s/%s  #%s\n\n", doc.CollectionName, doc.Path, doc.DocID)
		fmt.Println(body)
		return nil
	},
}

// parseGetSpec accepts `foo.md:42` and returns ("foo.md", 42). Bare specs
// return (spec, 0).
func parseGetSpec(in string) (string, int) {
	idx := strings.LastIndex(in, ":")
	if idx < 0 {
		return in, 0
	}
	n, err := strconv.Atoi(in[idx+1:])
	if err != nil {
		return in, 0
	}
	return in[:idx], n
}

// sliceLines returns content starting at from (1-based) with at most max
// lines. Zero values mean "no limit / from the start".
func sliceLines(content string, from, max int) string {
	if from <= 0 && max <= 0 {
		return content
	}
	lines := strings.Split(content, "\n")
	start := 0
	if from > 1 {
		start = from - 1
		if start > len(lines) {
			start = len(lines)
		}
	}
	end := len(lines)
	if max > 0 && start+max < end {
		end = start + max
	}
	return strings.Join(lines[start:end], "\n")
}

// suggestFallback ranks every indexed document by basename / token
// similarity to the user's query and recommends the top few. Returns a
// descriptive error containing those suggestions, or just "not found"
// when nothing is even close.
func suggestFallback(s *store.Store, spec string) error {
	needle := strings.ToLower(strings.TrimPrefix(spec, "#"))
	needle = strings.TrimSuffix(needle, filepath.Ext(needle))
	if needle == "" {
		return fmt.Errorf("document not found: %s", spec)
	}

	type cand struct {
		coll, path, docID string
		score             int
	}
	var picks []cand

	// MultiGetGlob("**") walks every doc.
	all, err := s.MultiGetGlob("**")
	if err != nil || len(all) == 0 {
		return fmt.Errorf("document not found: %s", spec)
	}

	for _, d := range all {
		hayPath := strings.ToLower(d.Path)
		hayBase := strings.ToLower(filepath.Base(d.Path))
		hayBase = strings.TrimSuffix(hayBase, filepath.Ext(hayBase))

		score := 0
		switch {
		case hayBase == needle:
			score = 100
		case strings.Contains(hayBase, needle):
			score = 60
		case strings.Contains(hayPath, needle):
			score = 30
		}
		if score > 0 {
			picks = append(picks, cand{d.CollectionName, d.Path, d.DocID, score})
		}
	}

	if len(picks) == 0 {
		return fmt.Errorf("document not found: %s", spec)
	}

	sort.Slice(picks, func(i, j int) bool { return picks[i].score > picks[j].score })

	var names []string
	for i, p := range picks {
		if i >= 5 {
			break
		}
		names = append(names, fmt.Sprintf("  %s/%s  #%s", p.coll, p.path, p.docID))
	}
	return fmt.Errorf("document not found: %s\ndid you mean:\n%s", spec, strings.Join(names, "\n"))
}

func init() {
	getCmd.Flags().IntVar(&getFromLine, "from", 0, "start at line number")
	getCmd.Flags().IntVarP(&getMaxLines, "lines", "l", 0, "max lines to return (0 = no limit)")
	getCmd.Flags().BoolVar(&getJSON, "json", false, "JSON output")
}
