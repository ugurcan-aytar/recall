package commands

import (
	"fmt"

	"github.com/spf13/cobra"
)

var cleanupCmd = &cobra.Command{
	Use:   "cleanup",
	Short: "Reclaim disk: drop orphan chunks, stale embeddings, then VACUUM",
	Long: "Removes chunk rows whose document is gone, vector rows whose chunk is\n" +
		"gone (sqlite-vec has no foreign keys, so re-indexing leaves these behind),\n" +
		"then runs SQLite VACUUM to release pages back to the filesystem.",
	RunE: func(cmd *cobra.Command, args []string) error {
		s, err := openStore()
		if err != nil {
			return err
		}
		defer s.Close()

		stats, err := s.Cleanup()
		if err != nil {
			return err
		}

		fmt.Printf("orphan chunks removed:    %d\n", stats.OrphanedChunks)
		fmt.Printf("stale embeddings removed: %d\n", stats.StaleEmbeddings)
		fmt.Printf("size before:              %s\n", humanBytes(stats.BytesBefore))
		fmt.Printf("size after:               %s\n", humanBytes(stats.BytesAfter))
		if delta := stats.BytesBefore - stats.BytesAfter; delta > 0 {
			fmt.Printf("reclaimed:                %s\n", humanBytes(delta))
		}
		return nil
	},
}
