package commands

import (
	"fmt"
	"os"

	"github.com/spf13/cobra"

	"github.com/ugurcan-aytar/recall/internal/chunk"
	"github.com/ugurcan-aytar/recall/internal/embed"
	"github.com/ugurcan-aytar/recall/internal/store"
)

const (
	embedBatchSize       = 32
	metadataKeyEmbedName = "embedding_model"
)

var (
	embedForce    bool
	embedStrategy string
)

var embedCmd = &cobra.Command{
	Use:   "embed",
	Short: "Generate vector embeddings for indexed chunks",
	Long: "Embeds chunks that don't yet have a vector. Already-embedded chunks " +
		"are skipped (incremental) unless -f is set. The model name is recorded " +
		"in the database so subsequent runs warn when it changes.",
	RunE: runEmbed,
}

func runEmbed(cmd *cobra.Command, args []string) error {
	s, err := openStore()
	if err != nil {
		return err
	}
	defer s.Close()

	emb, err := openEmbedder()
	if err != nil {
		return err
	}
	defer emb.Close()

	if err := reconcileModelName(s, emb.ModelName(), embedForce); err != nil {
		return err
	}

	var chunks []store.ChunkForEmbed
	if embedForce {
		// -f also re-chunks every doc using --chunk-strategy so a switch
		// from "regex" to "ast" (or vice versa) actually takes effect.
		if err := rechunkAllDocs(s, chunk.ChunkStrategy(embedStrategy)); err != nil {
			return fmt.Errorf("re-chunk: %w", err)
		}
		if err := s.DropAllEmbeddings(); err != nil {
			return fmt.Errorf("clear old embeddings: %w", err)
		}
		chunks, err = s.AllChunksForEmbed()
		if err != nil {
			return err
		}
	} else {
		chunks, err = s.ChunksNeedingEmbed()
		if err != nil {
			return err
		}
	}

	total := len(chunks)
	if total == 0 {
		fmt.Println("Nothing to embed. Run `recall index` first if you added new files.")
		return nil
	}

	fmt.Printf("Embedding %d chunks with %s (dim=%d)…\n",
		total, emb.ModelName(), emb.Dimensions())

	for i := 0; i < total; i += embedBatchSize {
		end := i + embedBatchSize
		if end > total {
			end = total
		}
		batch := chunks[i:end]
		family := emb.Family()
		texts := make([]string, len(batch))
		for j, c := range batch {
			texts[j] = embed.FormatDocumentFor(family, c.DocTitle, c.Content)
		}
		vecs, err := emb.Embed(texts)
		if err != nil {
			return fmt.Errorf("embed batch (%d..%d): %w", i, end, err)
		}
		for j, v := range vecs {
			if err := s.UpsertEmbedding(batch[j].ID, v); err != nil {
				return fmt.Errorf("store embedding for chunk %d: %w", batch[j].ID, err)
			}
		}
		fmt.Fprintf(os.Stderr, "\rEmbedded %d/%d chunks", end, total)
	}
	fmt.Fprintln(os.Stderr)

	if err := s.SetMetadata(metadataKeyEmbedName, emb.ModelName()); err != nil {
		return fmt.Errorf("record embedding model: %w", err)
	}

	fmt.Println("Done.")
	return nil
}

// rechunkAllDocs walks every document and rebuilds its chunks with the
// given strategy. Used by `recall embed -f --chunk-strategy …` so a
// switch from regex to AST actually re-shapes existing chunks.
func rechunkAllDocs(s *store.Store, strategy chunk.ChunkStrategy) error {
	docs, err := s.AllDocuments()
	if err != nil {
		return err
	}
	for _, d := range docs {
		newChunks := chunk.ChunkFile(d.Content, d.Path, strategy, 0, 0)
		if err := s.ReplaceChunks(d.ID, newChunks); err != nil {
			return fmt.Errorf("re-chunk doc %s: %w", d.Path, err)
		}
	}
	return nil
}

// reconcileModelName checks the database's recorded embedding model
// against the active embedder. If they differ, the user is asked to either
// run with -f (which drops old vectors) or accept that a mixed-model
// vector pool will produce poor results.
func reconcileModelName(s *store.Store, currentName string, force bool) error {
	prev, present, err := s.GetMetadata(metadataKeyEmbedName)
	if err != nil {
		return err
	}
	if !present || prev == "" || prev == currentName {
		return nil
	}
	if force {
		fmt.Fprintf(os.Stderr,
			"warning: model changed (%s → %s); -f set, dropping old embeddings\n",
			prev, currentName,
		)
		return nil
	}
	return fmt.Errorf(
		"embedding model changed since last run: have %q in DB, embedder is %q.\n"+
			"Re-run with `recall embed -f` to drop old vectors and re-embed everything.",
		prev, currentName,
	)
}

func init() {
	embedCmd.Flags().BoolVarP(&embedForce, "force", "f", false, "drop existing vectors and re-embed everything")
	embedCmd.Flags().StringVar(&embedStrategy, "chunk-strategy", "auto",
		"chunking strategy when -f triggers a re-chunk: auto | regex | ast")
	embedCmd.Flags().IntVar(&embedWorkersOverride, "workers", 0,
		"parallel embedder workers (0 = single worker, default; capped at 8 for both local and API backends)")
}
