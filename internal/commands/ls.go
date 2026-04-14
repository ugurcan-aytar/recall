package commands

import (
	"fmt"
	"strings"

	"github.com/spf13/cobra"
)

var lsCmd = &cobra.Command{
	Use:   "ls [collection[/path]]",
	Short: "List files in a collection",
	Args:  cobra.MaximumNArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		s, err := openStore()
		if err != nil {
			return err
		}
		defer s.Close()

		if len(args) == 0 {
			// No arg → list collections.
			cols, err := s.ListCollections()
			if err != nil {
				return err
			}
			for _, c := range cols {
				fmt.Println(c.Name)
			}
			return nil
		}

		collection, subPath := splitCollectionPath(args[0])
		paths, err := s.ListDocumentPaths(collection, subPath)
		if err != nil {
			return err
		}
		for _, p := range paths {
			fmt.Println(p)
		}
		return nil
	},
}

// splitCollectionPath parses "notes/sub/dir" into ("notes", "sub/dir") and
// bare "notes" into ("notes", "").
func splitCollectionPath(spec string) (string, string) {
	parts := strings.SplitN(spec, "/", 2)
	if len(parts) == 1 {
		return parts[0], ""
	}
	return parts[0], parts[1]
}
