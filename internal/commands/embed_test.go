package commands

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/ugurcan-aytar/recall/internal/embed"
	"github.com/ugurcan-aytar/recall/internal/store"
)

// writeTestFiles is a tiny test helper. (Name avoids collision with
// search.go's writeFiles output formatter.)
func writeTestFiles(root string, files map[string]string) error {
	for name, body := range files {
		p := filepath.Join(root, name)
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			return err
		}
		if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
			return err
		}
	}
	return nil
}

func TestEmbedCmdFlags(t *testing.T) {
	if f := embedCmd.Flags().Lookup("force"); f == nil {
		t.Error("embed missing --force")
	}
	if f := embedCmd.Flags().Lookup("chunk-strategy"); f == nil {
		t.Error("embed missing --chunk-strategy")
	}
	if f := embedCmd.Flags().Lookup("workers"); f == nil {
		t.Error("embed missing --workers")
	}
}

// TestResolveWorkersPrecedence pins the (--workers > $RECALL_EMBED_WORKERS > 0)
// resolution order. The 0 result means "let the embedder default to its own
// single-worker mode" — the actual default each backend picks is checked in
// the embed package's tests.
func TestResolveWorkersPrecedence(t *testing.T) {
	cases := []struct {
		name     string
		flag     int
		envVal   string
		want     int
	}{
		{"flag wins over env", 6, "3", 6},
		{"env used when flag is zero", 0, "5", 5},
		{"unset env returns zero", 0, "", 0},
		{"unparseable env returns zero", 0, "many", 0},
		{"negative env returns zero", 0, "-3", 0},
		{"zero env returns zero", 0, "0", 0},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv("RECALL_EMBED_WORKERS", tc.envVal)
			prev := embedWorkersOverride
			embedWorkersOverride = tc.flag
			t.Cleanup(func() { embedWorkersOverride = prev })
			if got := resolveWorkers(); got != tc.want {
				t.Errorf("resolveWorkers() = %d, want %d", got, tc.want)
			}
		})
	}
}

func TestReconcileModelNameNoChange(t *testing.T) {
	dir := t.TempDir()
	s, err := store.Open(filepath.Join(dir, "i.db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = s.Close() })

	// First call: no prior model recorded → no error.
	if err := reconcileModelName(s, "embeddinggemma", false); err != nil {
		t.Fatalf("first call: %v", err)
	}

	// Record the model and call again with the same name.
	if err := s.SetMetadata(metadataKeyEmbedName, "embeddinggemma"); err != nil {
		t.Fatal(err)
	}
	if err := reconcileModelName(s, "embeddinggemma", false); err != nil {
		t.Fatalf("identical model: %v", err)
	}
}

func TestReconcileModelNameChangeNoForce(t *testing.T) {
	dir := t.TempDir()
	s, err := store.Open(filepath.Join(dir, "i.db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = s.Close() })

	_ = s.SetMetadata(metadataKeyEmbedName, "old-model")
	if err := reconcileModelName(s, "new-model", false); err == nil {
		t.Error("expected error when model differs and -f not set")
	}
}

func TestReconcileModelNameChangeForceOK(t *testing.T) {
	dir := t.TempDir()
	s, err := store.Open(filepath.Join(dir, "i.db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = s.Close() })

	_ = s.SetMetadata(metadataKeyEmbedName, "old")
	if err := reconcileModelName(s, "new", true); err != nil {
		t.Errorf("force should bypass: %v", err)
	}
}

// TestEmbedRunWithMockEmbedder exercises the full embed pipeline using
// MockEmbedder via SetEmbedderOverride.
func TestEmbedRunWithMockEmbedder(t *testing.T) {
	dir := t.TempDir()
	dbPath = filepath.Join(dir, "i.db")
	t.Cleanup(func() { dbPath = "" })

	s, err := store.Open(dbPath)
	if err != nil {
		t.Fatal(err)
	}
	collDir := t.TempDir()
	if err := writeTestFiles(collDir, map[string]string{"a.md": "# a\nbody"}); err != nil {
		t.Fatal(err)
	}
	c, _ := s.AddCollection("n", collDir, "", "")
	if _, err := s.IndexCollection(c.ID); err != nil {
		t.Fatal(err)
	}
	_ = s.Close()

	mock := embed.NewMockEmbedder(store.EmbeddingDimensions)
	SetEmbedderOverride(mock)
	t.Cleanup(func() { SetEmbedderOverride(nil); resetQueryCacheForTest() })

	embedForce = false
	if err := runEmbed(embedCmd, nil); err != nil {
		t.Fatalf("runEmbed: %v", err)
	}

	// Verify embeddings landed.
	s2, _ := store.Open(dbPath)
	defer s2.Close()
	n, _ := s2.EmbeddingCount()
	if n == 0 {
		t.Error("no embeddings after runEmbed")
	}
	model, _, _ := s2.GetMetadata(metadataKeyEmbedName)
	if model != mock.ModelName() {
		t.Errorf("metadata model = %q, want %q", model, mock.ModelName())
	}
}
