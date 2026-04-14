package commands

import (
	"fmt"
	"os"
	"text/tabwriter"

	"github.com/spf13/cobra"
)

var statusCmd = &cobra.Command{
	Use:   "status",
	Short: "Show index health, collection info, and model info",
	RunE: func(cmd *cobra.Command, args []string) error {
		s, err := openStore()
		if err != nil {
			return err
		}
		defer s.Close()

		cols, err := s.ListCollections()
		if err != nil {
			return err
		}

		totalDocs, err := s.TotalDocumentCount()
		if err != nil {
			return err
		}
		totalChunks, err := s.ChunkCount()
		if err != nil {
			return err
		}
		totalEmbeddings, err := s.EmbeddingCount()
		if err != nil {
			return err
		}

		schemaVersion, _, _ := s.GetMetadata("schema_version")
		embedModel, _, _ := s.GetMetadata("embedding_model")
		if embedModel == "" {
			embedModel = "(none — run `recall embed` to generate)"
		}

		fmt.Printf("database:       %s\n", s.Path())
		if info, err := os.Stat(s.Path()); err == nil {
			fmt.Printf("size:           %s\n", humanBytes(info.Size()))
		}
		fmt.Printf("schema:         v%s\n", schemaVersion)
		fmt.Printf("embedding:      %s\n", embedModel)
		fmt.Printf("collections:    %d\n", len(cols))
		fmt.Printf("documents:      %d\n", totalDocs)
		fmt.Printf("chunks:         %d\n", totalChunks)
		fmt.Printf("embeddings:     %d\n", totalEmbeddings)
		fmt.Println()

		if len(cols) == 0 {
			return nil
		}

		tw := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
		fmt.Fprintln(tw, "COLLECTION\tDOCS\tLAST INDEXED\tPATH")
		for _, c := range cols {
			last := "-"
			if !c.LastIndexedAt.IsZero() {
				last = c.LastIndexedAt.Format("2006-01-02 15:04")
			}
			fmt.Fprintf(tw, "%s\t%d\t%s\t%s\n", c.Name, c.DocCount, last, c.Path)
		}
		return tw.Flush()
	},
}

func humanBytes(n int64) string {
	const (
		kb = 1024
		mb = kb * 1024
		gb = mb * 1024
	)
	switch {
	case n >= gb:
		return fmt.Sprintf("%.2f GB", float64(n)/gb)
	case n >= mb:
		return fmt.Sprintf("%.2f MB", float64(n)/mb)
	case n >= kb:
		return fmt.Sprintf("%.2f KB", float64(n)/kb)
	default:
		return fmt.Sprintf("%d B", n)
	}
}
