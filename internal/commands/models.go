package commands

import (
	"fmt"
	"os"
	"text/tabwriter"

	"github.com/spf13/cobra"

	"github.com/ugurcan-aytar/recall/internal/embed"
)

var modelsCmd = &cobra.Command{
	Use:   "models",
	Short: "List, download, or locate embedding models",
}

var modelsListCmd = &cobra.Command{
	Use:   "list",
	Short: "List downloaded GGUF models",
	RunE: func(cmd *cobra.Command, args []string) error {
		dir, err := embed.ModelsDir()
		if err != nil {
			return err
		}
		models, err := embed.ListLocalModels()
		if err != nil {
			return err
		}
		if len(models) == 0 {
			fmt.Printf("No models in %s.\nDownload one with: recall models download\n", dir)
			return nil
		}
		tw := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
		fmt.Fprintln(tw, "NAME\tSIZE\tUPDATED\tPATH")
		for _, m := range models {
			fmt.Fprintf(tw, "%s\t%s\t%s\t%s\n",
				m.Name, humanBytes(m.Size),
				m.UpdatedAt.Format("2006-01-02 15:04"), m.Path)
		}
		return tw.Flush()
	},
}

var (
	modelsDownloadURL       string
	modelsDownloadHash      string
	modelsDownloadDest      string
	modelsDownloadExpansion bool
	modelsDownloadReranker  bool
)

var modelsDownloadCmd = &cobra.Command{
	Use:   "download",
	Short: "Download a recall model: embedder (default), --expansion (--expand/--hyde LLM), or --reranker (--rerank LLM)",
	RunE: func(cmd *cobra.Command, args []string) error {
		if modelsDownloadExpansion && modelsDownloadReranker {
			return fmt.Errorf("--expansion and --reranker are mutually exclusive; pick one per invocation")
		}
		// Pick the default URL + filename based on which model the
		// user is asking for. Explicit --url or --dest still wins.
		defaultURL := embed.DefaultModelURL
		defaultName := embed.DefaultModelName
		label := "embedding model"
		if modelsDownloadExpansion {
			defaultURL = embed.DefaultExpansionModelURL
			defaultName = embed.DefaultExpansionModelName
			label = "expansion / HyDE LLM"
		}
		if modelsDownloadReranker {
			defaultURL = embed.DefaultRerankerModelURL
			defaultName = embed.DefaultRerankerModelName
			label = "reranker LLM"
		}

		dest := modelsDownloadDest
		if dest == "" {
			var err error
			dest, err = embed.ResolveModelPath(defaultName)
			if err != nil {
				return err
			}
		}
		url := modelsDownloadURL
		if url == "" {
			url = defaultURL
		}

		fmt.Printf("Downloading %s to %s\n", label, dest)
		path, err := embed.DownloadModel(embed.DownloadOptions{
			URL:          url,
			DestPath:     dest,
			ExpectedHash: modelsDownloadHash,
			Progress:     downloadProgress,
		})
		if err != nil {
			return err
		}
		fmt.Printf("\nDownloaded %s\n", path)
		return nil
	},
}

var modelsPathCmd = &cobra.Command{
	Use:   "path",
	Short: "Print the models directory",
	RunE: func(cmd *cobra.Command, args []string) error {
		dir, err := embed.ModelsDir()
		if err != nil {
			return err
		}
		fmt.Println(dir)
		return nil
	},
}

// downloadProgress prints a single-line progress indicator. Total may be -1
// when Content-Length is unknown.
func downloadProgress(written, total int64) {
	if total > 0 {
		pct := float64(written) / float64(total) * 100
		fmt.Fprintf(os.Stderr, "\r  %s / %s (%.1f%%)", humanBytes(written), humanBytes(total), pct)
		return
	}
	fmt.Fprintf(os.Stderr, "\r  %s downloaded", humanBytes(written))
}

func init() {
	modelsDownloadCmd.Flags().StringVar(&modelsDownloadURL, "url", "", "override download URL")
	modelsDownloadCmd.Flags().StringVar(&modelsDownloadHash, "sha256", "", "expected SHA-256 (hex)")
	modelsDownloadCmd.Flags().StringVar(&modelsDownloadDest, "dest", "", "override destination path")
	modelsDownloadCmd.Flags().BoolVar(&modelsDownloadExpansion, "expansion", false,
		"download the query-expansion / HyDE LLM (qmd-query-expansion-1.7B, ~1.3 GB) instead of the embedding model")
	modelsDownloadCmd.Flags().BoolVar(&modelsDownloadReranker, "reranker", false,
		"download the reranker LLM (Qwen2.5-1.5B-Instruct, ~1.1 GB) used by --rerank")

	modelsCmd.AddCommand(modelsListCmd, modelsDownloadCmd, modelsPathCmd)
}
