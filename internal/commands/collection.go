package commands

import (
	"errors"
	"fmt"
	"os"
	"text/tabwriter"

	"github.com/spf13/cobra"

	"github.com/ugurcan-aytar/recall/internal/store"
)

var collectionCmd = &cobra.Command{
	Use:   "collection",
	Short: "Manage collections (add, remove, list, rename)",
}

var (
	collAddName    string
	collAddMask    string
	collAddContext string
)

var collectionAddCmd = &cobra.Command{
	Use:   "add <path>",
	Short: "Register a folder as a collection",
	Args:  cobra.ExactArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		path := args[0]
		if _, err := os.Stat(path); err != nil {
			return fmt.Errorf("collection path %s: %w", path, err)
		}

		s, err := openStore()
		if err != nil {
			return err
		}
		defer s.Close()

		c, err := s.AddCollection(collAddName, path, collAddMask, collAddContext)
		if err != nil {
			return err
		}
		fmt.Printf("Added collection %q at %s (glob: %s)\n", c.Name, c.Path, c.GlobPattern)
		return nil
	},
}

var collectionRemoveCmd = &cobra.Command{
	Use:   "remove <name>",
	Short: "Remove a collection (cascades to documents)",
	Args:  cobra.ExactArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		s, err := openStore()
		if err != nil {
			return err
		}
		defer s.Close()

		if err := s.RemoveCollection(args[0]); err != nil {
			return err
		}
		fmt.Printf("Removed collection %q\n", args[0])
		return nil
	},
}

var collectionListCmd = &cobra.Command{
	Use:   "list",
	Short: "List registered collections",
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
			fmt.Println("No collections. Add one with: recall collection add <path> --name <name>")
			return nil
		}

		tw := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
		fmt.Fprintln(tw, "NAME\tDOCS\tPATH\tGLOB\tLAST INDEXED")
		for _, c := range cols {
			last := "-"
			if !c.LastIndexedAt.IsZero() {
				last = c.LastIndexedAt.Format("2006-01-02 15:04")
			}
			fmt.Fprintf(tw, "%s\t%d\t%s\t%s\t%s\n",
				c.Name, c.DocCount, c.Path, c.GlobPattern, last)
		}
		return tw.Flush()
	},
}

var collectionRenameCmd = &cobra.Command{
	Use:   "rename <old> <new>",
	Short: "Rename a collection",
	Args:  cobra.ExactArgs(2),
	RunE: func(cmd *cobra.Command, args []string) error {
		s, err := openStore()
		if err != nil {
			return err
		}
		defer s.Close()

		if err := s.RenameCollection(args[0], args[1]); err != nil {
			if errors.Is(err, store.ErrCollectionNotFound) {
				return fmt.Errorf("no collection named %q", args[0])
			}
			return err
		}
		fmt.Printf("Renamed %q → %q\n", args[0], args[1])
		return nil
	},
}

func init() {
	collectionAddCmd.Flags().StringVar(&collAddName, "name", "", "override default name (folder basename)")
	collectionAddCmd.Flags().StringVar(&collAddMask, "mask", "", "override default file glob ("+store.DefaultGlobPattern+")")
	collectionAddCmd.Flags().StringVar(&collAddContext, "context", "", "descriptive context for this collection")

	collectionCmd.AddCommand(
		collectionAddCmd,
		collectionRemoveCmd,
		collectionListCmd,
		collectionRenameCmd,
	)
}
