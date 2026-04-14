package commands

import (
	"os"

	"github.com/spf13/cobra"

	"github.com/ugurcan-aytar/recall/internal/store"
)

var (
	dbPath  string
	noColor bool
)

var rootCmd = &cobra.Command{
	Use:   "recall",
	Short: "Lightweight local search engine for your documents",
	Long:  "recall indexes your markdown and text files and lets you search them with BM25, vector similarity, or hybrid fusion. All local, no API calls.",
}

// Execute runs the root command. Called from cmd/recall/main.go.
func Execute() error {
	return rootCmd.Execute()
}

// openStore opens a *store.Store honouring the --db flag, then the
// RECALL_DB_PATH env var, then the default path. Centralised here so every
// subcommand opens the same way.
func openStore() (*store.Store, error) {
	return store.Open(dbPath)
}

// colorsEnabled reports whether ANSI colour codes should appear in output.
// Honours the --no-color flag and the NO_COLOR environment convention.
func colorsEnabled() bool {
	if noColor {
		return false
	}
	if os.Getenv("NO_COLOR") != "" {
		return false
	}
	return true
}

func init() {
	rootCmd.PersistentFlags().StringVar(&dbPath, "db", "", "database path (default ~/.recall/index.db, override via $RECALL_DB_PATH)")
	rootCmd.PersistentFlags().BoolVar(&noColor, "no-color", false, "disable colorized output")

	rootCmd.AddCommand(collectionCmd)
	rootCmd.AddCommand(searchCmd)
	rootCmd.AddCommand(vsearchCmd)
	rootCmd.AddCommand(queryCmd)
	rootCmd.AddCommand(indexCmd)
	rootCmd.AddCommand(embedCmd)
	rootCmd.AddCommand(getCmd)
	rootCmd.AddCommand(multiGetCmd)
	rootCmd.AddCommand(contextCmd)
	rootCmd.AddCommand(statusCmd)
	rootCmd.AddCommand(doctorCmd)
	rootCmd.AddCommand(modelsCmd)
	rootCmd.AddCommand(lsCmd)
	rootCmd.AddCommand(cleanupCmd)
	rootCmd.AddCommand(versionCmd)
}
