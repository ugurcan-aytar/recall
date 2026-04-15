package commands

import (
	"fmt"
	"os"

	"github.com/spf13/cobra"

	"github.com/ugurcan-aytar/recall/internal/store"
)

var (
	dbPath    string
	indexName string
	noColor   bool
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

// resolveStorePath picks the DB file honouring, in order:
//
//  1. --db (explicit file path; highest priority, wins over everything)
//  2. $RECALL_DB_PATH (env var; for CI / scripted path overrides)
//  3. --index <name> (mapped to ~/.recall/indexes/<name>.db)
//  4. DefaultDBPath (~/.recall/index.db)
//
// When --index AND a path-setter (--db or $RECALL_DB_PATH) both
// appear, --index is ignored and a warning goes to stderr — the
// path-setter is more explicit about which file to touch and
// silently dropping it would be worse than the warning noise.
func resolveStorePath() (string, error) {
	envPath := os.Getenv("RECALL_DB_PATH")

	if dbPath != "" || envPath != "" {
		if indexName != "" {
			fmt.Fprintf(os.Stderr,
				"warning: --index %q ignored because %s is set\n",
				indexName,
				func() string {
					if dbPath != "" {
						return "--db"
					}
					return "$RECALL_DB_PATH"
				}(),
			)
		}
		return store.ResolveDBPath(dbPath)
	}

	if indexName != "" {
		return store.ResolveIndexPath(indexName)
	}
	return store.ResolveDBPath("")
}

// openStore opens a *store.Store honouring --db / $RECALL_DB_PATH /
// --index / default precedence. Centralised here so every subcommand
// opens the same way.
func openStore() (*store.Store, error) {
	p, err := resolveStorePath()
	if err != nil {
		return nil, err
	}
	return store.Open(p)
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
	rootCmd.PersistentFlags().StringVar(&indexName, "index", "", "named index shortcut — uses ~/.recall/indexes/<name>.db; ignored when --db or $RECALL_DB_PATH is set")
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
	rootCmd.AddCommand(benchCmd)
	rootCmd.AddCommand(versionCmd)
}
