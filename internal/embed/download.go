package embed

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// DefaultModelName is the GGUF embedder recall ships with by default.
//
// nomic-embed-text-v1.5 was chosen over Google's embeddinggemma because
// the latter is gated on HuggingFace (requires an account and license
// acceptance). nomic-embed-text-v1.5 is Apache 2.0, ungated, ~146 MB
// quantised, 768-dim output — matches our vec0 schema directly.
const DefaultModelName = "nomic-embed-text-v1.5.Q8_0.gguf"

// ResolveActiveModelPath returns the GGUF file recall should load,
// honouring $RECALL_EMBED_MODEL for the override. The env var may hold:
//
//   - a bare filename ("my-embed.gguf") — joined with [ModelsDir]
//   - an absolute path ("/opt/models/my-embed.gguf") — used as-is
//
// When unset, falls back to ResolveModelPath(DefaultModelName).
// Callers get the path only; the file may not exist yet, in which case
// the embedder constructor will error.
func ResolveActiveModelPath() (string, error) {
	if v := strings.TrimSpace(os.Getenv("RECALL_EMBED_MODEL")); v != "" {
		if filepath.IsAbs(v) {
			return v, nil
		}
		return ResolveModelPath(v)
	}
	return ResolveModelPath(DefaultModelName)
}

// DefaultModelURL is the canonical HuggingFace location for the bundled
// model. Override per-call via DownloadOptions.URL when mirroring locally.
const DefaultModelURL = "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.Q8_0.gguf"

// DefaultExpansionModelName is the GGUF generation model recall uses
// for query expansion (--expand) and HyDE (--hyde). The model is a
// fine-tune of Qwen3-1.7B that emits structured `lex: …`, `vec: …`,
// `hyde: …` lines — exactly the shape recall's expansion + HyDE
// pipeline consumes — so one model load covers both features.
//
// Source: tobil/qmd-query-expansion-1.7B-gguf (MIT, ungated).
// Override per-installation with $RECALL_EXPAND_MODEL.
const DefaultExpansionModelName = "qmd-query-expansion-1.7B-q4_k_m.gguf"

// DefaultExpansionModelURL is the canonical HuggingFace location.
const DefaultExpansionModelURL = "https://huggingface.co/tobil/qmd-query-expansion-1.7B-gguf/resolve/main/qmd-query-expansion-1.7B-q4_k_m.gguf"

// ResolveActiveExpansionModelPath returns the GGUF file recall should
// load for query expansion / HyDE, honouring $RECALL_EXPAND_MODEL.
// Bare filename joins with [ModelsDir]; absolute path passes through.
func ResolveActiveExpansionModelPath() (string, error) {
	if v := strings.TrimSpace(os.Getenv("RECALL_EXPAND_MODEL")); v != "" {
		if filepath.IsAbs(v) {
			return v, nil
		}
		return ResolveModelPath(v)
	}
	return ResolveModelPath(DefaultExpansionModelName)
}

// DefaultRerankerModelName is the cross-encoder reranker recall
// loads for --rerank. v0.2.4 switched from a Qwen2.5-1.5B-Instruct
// binary-yes/no fallback (documented below) to BAAI's
// bge-reranker-v2-m3 at Q4_K_M (~418 MB). bge-reranker-v2-m3 is
// explicitly supported by llama.cpp's reranking path (PR #9510
// added rerank to libllama / llama-server / llama-embedding using
// this model as the reference), Apache 2.0, multilingual (568 M
// params, bge-m3 base), and published in GGUF by multiple
// reputable uploaders. It returns continuous relevance logits
// (roughly [-12, +8] in practice) via llama-server's /v1/rerank
// endpoint — real gradient signal, not the binary-yes/no echo we
// were stuck with.
//
// History: pre-v0.2.4 recall used Qwen2.5-1.5B-Instruct with a
// binary-yes/no prompt, because the reranker had to go through the
// same /v1/chat/completions surface the expansion path used — the
// old backend (dianlight/gollama.cpp → godeps/gollama before that)
// didn't expose llama.cpp's rank-pooling surface. v0.2.2's
// subprocess pattern means we now talk to llama-server directly;
// llama-server's `--reranking` flag + `/v1/rerank` endpoint turn
// out to have shipped long ago — the "not yet supported" comment
// in previous recall versions was just wrong.
//
// Override per-installation with $RECALL_RERANK_MODEL.
const DefaultRerankerModelName = "bge-reranker-v2-m3-Q4_K_M.gguf"

// DefaultRerankerModelURL is the canonical HuggingFace location.
const DefaultRerankerModelURL = "https://huggingface.co/gpustack/bge-reranker-v2-m3-GGUF/resolve/main/bge-reranker-v2-m3-Q4_K_M.gguf"

// ResolveActiveRerankerModelPath returns the GGUF file recall should
// load for the reranker, honouring $RECALL_RERANK_MODEL. Bare
// filename joins with [ModelsDir]; absolute path passes through.
func ResolveActiveRerankerModelPath() (string, error) {
	if v := strings.TrimSpace(os.Getenv("RECALL_RERANK_MODEL")); v != "" {
		if filepath.IsAbs(v) {
			return v, nil
		}
		return ResolveModelPath(v)
	}
	return ResolveModelPath(DefaultRerankerModelName)
}

// DownloadOptions configures a model fetch.
type DownloadOptions struct {
	URL          string                       // download source; "" ⇒ DefaultModelURL
	DestPath     string                       // absolute file path; "" ⇒ ResolveModelPath(DefaultModelName)
	ExpectedHash string                       // hex-encoded SHA-256; "" ⇒ skip verification
	Progress     func(written, total int64)   // optional, called periodically
	HTTPClient   *http.Client                 // overridable for tests
}

// DownloadModel fetches a GGUF file to disk, optionally verifying its
// SHA-256 digest. Existing files at DestPath that already match the hash
// are kept (the function is a no-op).
func DownloadModel(opts DownloadOptions) (string, error) {
	dest := opts.DestPath
	if dest == "" {
		var err error
		dest, err = ResolveModelPath(DefaultModelName)
		if err != nil {
			return "", err
		}
	}
	url := opts.URL
	if url == "" {
		url = DefaultModelURL
	}

	if existing, ok := alreadyMatches(dest, opts.ExpectedHash); ok {
		return existing, nil
	}

	if err := os.MkdirAll(filepath.Dir(dest), 0o755); err != nil {
		return "", fmt.Errorf("create models dir: %w", err)
	}

	client := opts.HTTPClient
	if client == nil {
		client = &http.Client{Timeout: 0} // large files; rely on per-read deadlines
	}

	req, err := http.NewRequest(http.MethodGet, url, nil)
	if err != nil {
		return "", fmt.Errorf("build request: %w", err)
	}
	req.Header.Set("User-Agent", "recall-cli")

	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("download %s: %w", url, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("download %s: HTTP %d", url, resp.StatusCode)
	}

	tmp := dest + ".part"
	f, err := os.Create(tmp)
	if err != nil {
		return "", fmt.Errorf("create %s: %w", tmp, err)
	}

	hasher := sha256.New()
	written, err := io.Copy(io.MultiWriter(f, hasher), progressReader(resp.Body, resp.ContentLength, opts.Progress))
	if cerr := f.Close(); err == nil {
		err = cerr
	}
	if err != nil {
		_ = os.Remove(tmp)
		return "", fmt.Errorf("write %s: %w", tmp, err)
	}

	got := hex.EncodeToString(hasher.Sum(nil))
	if opts.ExpectedHash != "" && !strings.EqualFold(got, opts.ExpectedHash) {
		_ = os.Remove(tmp)
		return "", fmt.Errorf("checksum mismatch: got %s, want %s (downloaded %d bytes from %s)",
			got, opts.ExpectedHash, written, url)
	}

	if err := os.Rename(tmp, dest); err != nil {
		return "", fmt.Errorf("rename %s → %s: %w", tmp, dest, err)
	}
	return dest, nil
}

// ResolveModelPath returns the absolute on-disk path for a model file,
// honouring $RECALL_MODELS_DIR (defaults to ~/.recall/models/).
func ResolveModelPath(name string) (string, error) {
	dir, err := ModelsDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(dir, name), nil
}

// ModelsDir returns the directory recall uses for GGUF files.
func ModelsDir() (string, error) {
	if v := os.Getenv("RECALL_MODELS_DIR"); v != "" {
		if strings.HasPrefix(v, "~/") {
			home, err := os.UserHomeDir()
			if err != nil {
				return "", err
			}
			return filepath.Join(home, v[2:]), nil
		}
		return v, nil
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("get home dir: %w", err)
	}
	return filepath.Join(home, ".recall", "models"), nil
}

// ListLocalModels returns every .gguf file currently under ModelsDir, with
// its absolute path and on-disk size in bytes.
type LocalModel struct {
	Name      string
	Path      string
	Size      int64
	UpdatedAt time.Time
}

func ListLocalModels() ([]LocalModel, error) {
	dir, err := ModelsDir()
	if err != nil {
		return nil, err
	}
	entries, err := os.ReadDir(dir)
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	var out []LocalModel
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".gguf") {
			continue
		}
		info, err := e.Info()
		if err != nil {
			continue
		}
		out = append(out, LocalModel{
			Name:      e.Name(),
			Path:      filepath.Join(dir, e.Name()),
			Size:      info.Size(),
			UpdatedAt: info.ModTime(),
		})
	}
	return out, nil
}

// alreadyMatches returns (path, true) when dest exists and (no expected hash
// was supplied OR the existing file's hash matches).
func alreadyMatches(dest, expected string) (string, bool) {
	info, err := os.Stat(dest)
	if err != nil || info.IsDir() {
		return "", false
	}
	if expected == "" {
		return dest, true
	}
	got, err := fileSHA256(dest)
	if err != nil {
		return "", false
	}
	return dest, strings.EqualFold(got, expected)
}

func fileSHA256(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

// progressReader wraps an io.Reader so a callback receives byte counts as
// data flows. cb(written, total) is called at most every 256 KB; total
// may be -1 when Content-Length is unknown.
func progressReader(r io.Reader, total int64, cb func(written, total int64)) io.Reader {
	if cb == nil {
		return r
	}
	return &progressR{r: r, total: total, cb: cb}
}

type progressR struct {
	r         io.Reader
	total     int64
	written   int64
	lastEmit  int64
	cb        func(int64, int64)
}

func (p *progressR) Read(b []byte) (int, error) {
	n, err := p.r.Read(b)
	if n > 0 {
		p.written += int64(n)
		if p.written-p.lastEmit >= 256*1024 || err != nil {
			p.cb(p.written, p.total)
			p.lastEmit = p.written
		}
	}
	return n, err
}
