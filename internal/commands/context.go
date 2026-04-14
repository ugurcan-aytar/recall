package commands

import (
	"fmt"
	"os"
	"strings"
	"text/tabwriter"

	"github.com/spf13/cobra"
)

var contextCmd = &cobra.Command{
	Use:   "context",
	Short: "Manage path contexts (add, list, rm, check)",
}

var contextAddCmd = &cobra.Command{
	Use:   "add [path] <context>",
	Short: "Add descriptive context for a path or globally",
	Args:  cobra.RangeArgs(1, 2),
	RunE: func(cmd *cobra.Command, args []string) error {
		var spec, text string
		if len(args) == 1 {
			spec, text = "/", args[0]
		} else {
			spec, text = args[0], args[1]
		}

		collection, path := splitContextSpec(spec)

		s, err := openStore()
		if err != nil {
			return err
		}
		defer s.Close()

		if err := s.AddContext(collection, path, text); err != nil {
			return err
		}
		fmt.Printf("Added context for %s\n", formatContextRef(collection, path))
		return nil
	},
}

var contextListCmd = &cobra.Command{
	Use:   "list",
	Short: "List all contexts",
	RunE: func(cmd *cobra.Command, args []string) error {
		s, err := openStore()
		if err != nil {
			return err
		}
		defer s.Close()

		cs, err := s.ListContexts()
		if err != nil {
			return err
		}
		if len(cs) == 0 {
			fmt.Println("No contexts registered.")
			return nil
		}
		tw := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
		fmt.Fprintln(tw, "SCOPE\tPATH\tCONTEXT")
		for _, c := range cs {
			scope := c.Collection
			if scope == "" {
				scope = "(global)"
			}
			fmt.Fprintf(tw, "%s\t%s\t%s\n", scope, c.Path, c.Context)
		}
		return tw.Flush()
	},
}

var contextRmCmd = &cobra.Command{
	Use:   "rm <path>",
	Short: "Remove a context",
	Args:  cobra.ExactArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		collection, path := splitContextSpec(args[0])

		s, err := openStore()
		if err != nil {
			return err
		}
		defer s.Close()

		if err := s.RemoveContext(collection, path); err != nil {
			return err
		}
		fmt.Printf("Removed context for %s\n", formatContextRef(collection, path))
		return nil
	},
}

var contextCheckCmd = &cobra.Command{
	Use:   "check",
	Short: "List collections that do not yet have a context",
	RunE: func(cmd *cobra.Command, args []string) error {
		s, err := openStore()
		if err != nil {
			return err
		}
		defer s.Close()

		missing, err := s.CheckContexts()
		if err != nil {
			return err
		}
		if len(missing) == 0 {
			fmt.Println("All collections have context.")
			return nil
		}
		fmt.Println("Collections missing context:")
		for _, n := range missing {
			fmt.Printf("  %s\n", n)
		}
		return nil
	},
}

// splitContextSpec turns user-facing shapes into (collection, path):
//
//	"/"              → ("", "/")                    global
//	"notes"          → ("notes", "/")               collection root
//	"notes/work"     → ("notes", "work")            path inside collection
//	"recall://notes/ideas" → ("notes", "ideas")     explicit scheme form
func splitContextSpec(spec string) (string, string) {
	if spec == "/" || spec == "" {
		return "", "/"
	}
	spec = strings.TrimPrefix(spec, "recall://")
	spec = strings.TrimPrefix(spec, "qmd://")
	parts := strings.SplitN(spec, "/", 2)
	if len(parts) == 1 {
		return parts[0], "/"
	}
	if parts[1] == "" {
		return parts[0], "/"
	}
	return parts[0], parts[1]
}

func formatContextRef(collection, path string) string {
	if collection == "" {
		return "(global)"
	}
	if path == "" || path == "/" {
		return collection
	}
	return collection + "/" + path
}

func init() {
	contextCmd.AddCommand(contextAddCmd, contextListCmd, contextRmCmd, contextCheckCmd)
}
