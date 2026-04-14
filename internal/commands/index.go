package commands

import (
	"fmt"
	"os"
	"os/exec"

	"github.com/spf13/cobra"
)

var indexPull bool

var indexCmd = &cobra.Command{
	Use:   "index",
	Short: "Re-scan and index all collections",
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
		if len(cols) == 0 {
			fmt.Println("No collections to index. Add one with: recall collection add <path>")
			return nil
		}

		for _, c := range cols {
			if indexPull {
				if err := gitPull(c.Path); err != nil {
					fmt.Fprintf(os.Stderr, "warning: git pull failed for %s: %v\n", c.Name, err)
				}
			}
			stats, err := s.IndexCollection(c.ID)
			if err != nil {
				return fmt.Errorf("index %s: %w", c.Name, err)
			}
			fmt.Printf("%s: indexed=%d updated=%d unchanged=%d removed=%d\n",
				c.Name, stats.Indexed, stats.Updated, stats.Unchanged, stats.Removed)
		}
		return nil
	},
}

// gitPull runs `git -C <dir> pull` and returns an error when git fails (or
// when the directory is not a git repo). Output goes to stderr so users can
// see merge conflicts or auth failures.
func gitPull(dir string) error {
	cmd := exec.Command("git", "-C", dir, "pull")
	cmd.Stdout = os.Stderr
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

func init() {
	indexCmd.Flags().BoolVar(&indexPull, "pull", false, "git pull each collection before re-indexing")
}
