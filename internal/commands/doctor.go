package commands

import (
	"fmt"
	"os"

	"github.com/spf13/cobra"

	"github.com/ugurcan-aytar/recall/internal/embed"
)

var doctorCmd = &cobra.Command{
	Use:   "doctor",
	Short: "Verify database, schema, and (eventually) embedding model setup",
	RunE: func(cmd *cobra.Command, args []string) error {
		ok := true

		s, err := openStore()
		if err != nil {
			fail("open database", err)
			return err
		}
		defer s.Close()

		pass("database opens", s.Path())

		if _, err := s.ListCollections(); err != nil {
			fail("collections table", err)
			ok = false
		} else {
			pass("collections table", "readable")
		}

		total, err := s.TotalDocumentCount()
		if err != nil {
			fail("documents table", err)
			ok = false
		} else {
			pass("documents table", fmt.Sprintf("%d docs", total))
		}

		schemaVersion, present, err := s.GetMetadata("schema_version")
		if err != nil {
			fail("metadata table", err)
			ok = false
		} else if !present {
			warn("schema_version missing", "run `recall index` to trigger migration")
		} else {
			pass("schema version", "v"+schemaVersion)
		}

		// Embedding stack: tell the user exactly which provider would be
		// used today and why, so they don't have to guess what `recall
		// embed` will do.
		if provider := embed.ResolveAPIProvider(); provider != embed.ProviderLocal {
			pass("embedding provider", "API ("+string(provider)+") via $RECALL_EMBED_PROVIDER")
		} else if embed.LocalEmbedderAvailable() {
			modelPath, _ := embed.ResolveModelPath(embed.DefaultModelName)
			if _, err := os.Stat(modelPath); err == nil {
				pass("embedding model", modelPath)
			} else {
				warn("embedding model",
					fmt.Sprintf("%s missing — run `recall models download`", modelPath))
			}
		} else {
			warn("embedding backend",
				"local GGUF not compiled in (rebuild with -tags 'sqlite_fts5 embed_llama' or set RECALL_EMBED_PROVIDER=openai|voyage)")
		}

		if recorded, present, _ := s.GetMetadata("embedding_model"); present && recorded != "" {
			pass("embedded with", recorded)
		}

		if !ok {
			return fmt.Errorf("doctor found problems")
		}
		return nil
	},
}

func pass(name, detail string) {
	fmt.Printf("  ok    %-22s  %s\n", name, detail)
}

func warn(name, detail string) {
	fmt.Printf("  warn  %-22s  %s\n", name, detail)
}

func fail(name string, err error) {
	fmt.Fprintf(os.Stderr, "  fail  %-22s  %v\n", name, err)
}
