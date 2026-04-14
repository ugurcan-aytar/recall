package embed

import (
	"crypto/sha256"
	"encoding/hex"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestModelsDirHonoursEnv(t *testing.T) {
	t.Setenv("RECALL_MODELS_DIR", "/tmp/recall-models-test")
	dir, err := ModelsDir()
	if err != nil {
		t.Fatal(err)
	}
	if dir != "/tmp/recall-models-test" {
		t.Errorf("ModelsDir = %q", dir)
	}
}

func TestModelsDirDefault(t *testing.T) {
	t.Setenv("RECALL_MODELS_DIR", "")
	dir, err := ModelsDir()
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasSuffix(dir, ".recall/models") {
		t.Errorf("default models dir = %q", dir)
	}
}

func TestResolveModelPath(t *testing.T) {
	t.Setenv("RECALL_MODELS_DIR", "/tmp/m")
	p, err := ResolveModelPath("foo.gguf")
	if err != nil {
		t.Fatal(err)
	}
	if p != "/tmp/m/foo.gguf" {
		t.Errorf("path = %q", p)
	}
}

func TestListLocalModelsMissingDir(t *testing.T) {
	t.Setenv("RECALL_MODELS_DIR", filepath.Join(t.TempDir(), "nope"))
	out, err := ListLocalModels()
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 0 {
		t.Errorf("missing dir should return empty list, got %+v", out)
	}
}

func TestListLocalModelsFiltersGguf(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("RECALL_MODELS_DIR", dir)
	for _, name := range []string{"a.gguf", "b.gguf", "c.txt", "d.bin"} {
		if err := os.WriteFile(filepath.Join(dir, name), []byte("x"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	out, err := ListLocalModels()
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 2 {
		t.Errorf("got %d, want 2 (.gguf only)", len(out))
	}
	for _, m := range out {
		if !strings.HasSuffix(m.Name, ".gguf") {
			t.Errorf("non-gguf in list: %s", m.Name)
		}
	}
}

func TestDownloadModelHTTP(t *testing.T) {
	body := []byte("hello, this is a fake gguf payload")
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Let net/http compute Content-Length from the actual body.
		_, _ = w.Write(body)
	}))
	defer srv.Close()

	dir := t.TempDir()
	dest := filepath.Join(dir, "fake.gguf")

	sum := sha256.Sum256(body)
	wantHash := hex.EncodeToString(sum[:])

	var seenWritten, seenTotal int64
	path, err := DownloadModel(DownloadOptions{
		URL:          srv.URL,
		DestPath:     dest,
		ExpectedHash: wantHash,
		Progress: func(w, tot int64) {
			seenWritten = w
			seenTotal = tot
		},
	})
	if err != nil {
		t.Fatalf("DownloadModel: %v", err)
	}
	if path != dest {
		t.Errorf("returned path = %q", path)
	}
	on, err := os.ReadFile(dest)
	if err != nil {
		t.Fatal(err)
	}
	if string(on) != string(body) {
		t.Errorf("file content mismatch")
	}
	_ = seenWritten
	_ = seenTotal
}

func TestDownloadModelChecksumMismatch(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte("payload"))
	}))
	defer srv.Close()

	dir := t.TempDir()
	dest := filepath.Join(dir, "x.gguf")

	_, err := DownloadModel(DownloadOptions{
		URL:          srv.URL,
		DestPath:     dest,
		ExpectedHash: "deadbeef",
	})
	if err == nil {
		t.Fatal("expected checksum-mismatch error")
	}
	if _, statErr := os.Stat(dest); !os.IsNotExist(statErr) {
		t.Error("partial file should have been cleaned up")
	}
}

func TestDownloadModelKeepsExistingMatch(t *testing.T) {
	dir := t.TempDir()
	dest := filepath.Join(dir, "x.gguf")
	body := []byte("already here")
	if err := os.WriteFile(dest, body, 0o644); err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(body)
	hash := hex.EncodeToString(sum[:])

	// Server intentionally returns a different payload — if the function
	// runs it, the SHA will mismatch and we'd see an error.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte("would corrupt"))
	}))
	defer srv.Close()

	path, err := DownloadModel(DownloadOptions{
		URL:          srv.URL,
		DestPath:     dest,
		ExpectedHash: hash,
	})
	if err != nil {
		t.Fatalf("DownloadModel: %v", err)
	}
	got, _ := os.ReadFile(path)
	if string(got) != string(body) {
		t.Errorf("existing file was overwritten: %q", got)
	}
}

func TestDownloadModelHTTPError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "nope", http.StatusNotFound)
	}))
	defer srv.Close()

	dir := t.TempDir()
	dest := filepath.Join(dir, "x.gguf")
	_, err := DownloadModel(DownloadOptions{URL: srv.URL, DestPath: dest})
	if err == nil {
		t.Error("expected error on HTTP 404")
	}
}
