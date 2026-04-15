// Package llamacpp owns the lifecycle of the llama.cpp prebuilt
// binary that recall shells out to for local embedding (and, in a
// future release, generation). Pre-v0.2.2 recall linked llama.cpp
// in-process via dianlight/gollama.cpp; that backend's Go struct
// drifted from the bundled C library and silently produced all-zero
// embeddings. v0.2.2 reverts to the simplest possible path: download
// the official llama.cpp release archive, run llama-server as a
// subprocess, talk to it over a Unix socket. No CGo on the inference
// hot path, no struct ABI guesswork, and the model load amortises
// across the entire embed run.
//
// The pinned version is set via DefaultVersion below. Override per
// install with $RECALL_LLAMACPP_VERSION (useful for trying a newer
// release without bumping recall itself).
package llamacpp

import (
	"archive/zip"
	"compress/gzip"
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"

	"archive/tar"
)

// DefaultVersion is the llama.cpp release tag recall validated
// against. Bump alongside the recall release that introduces the
// change so users aren't surprised by API or binary drift.
const DefaultVersion = "b8798"

// ServerBinary is the entry-point program inside the release archive.
// llama-server is the only binary recall actually invokes; the rest
// of the archive ships dylibs that llama-server links against.
const ServerBinary = "llama-server"

// ErrUnsupportedPlatform reports a GOOS/GOARCH combo we don't have
// an upstream prebuilt binary for. Callers should propagate it
// up so the user gets a clear "fall back to RECALL_EMBED_PROVIDER"
// hint instead of a download crash.
var ErrUnsupportedPlatform = errors.New("llama.cpp: no prebuilt binary for this platform")

// assetForPlatform returns the asset filename within the GitHub
// release matching the current GOOS/GOARCH. Naming taken from
// https://github.com/ggml-org/llama.cpp/releases — kept here in
// one map so adding a platform is a one-line change.
func assetForPlatform(version string) (asset string, isZip bool, err error) {
	key := runtime.GOOS + "/" + runtime.GOARCH
	switch key {
	case "darwin/arm64":
		// macOS Apple Silicon ships Metal GPU support by default.
		return fmt.Sprintf("llama-%s-bin-macos-arm64.tar.gz", version), false, nil
	case "darwin/amd64":
		return fmt.Sprintf("llama-%s-bin-macos-x64.tar.gz", version), false, nil
	case "linux/amd64":
		return fmt.Sprintf("llama-%s-bin-ubuntu-x64.tar.gz", version), false, nil
	case "linux/arm64":
		return fmt.Sprintf("llama-%s-bin-ubuntu-arm64.tar.gz", version), false, nil
	case "windows/amd64":
		return fmt.Sprintf("llama-%s-bin-win-cpu-x64.zip", version), true, nil
	default:
		return "", false, fmt.Errorf("%w: %s", ErrUnsupportedPlatform, key)
	}
}

// Version returns the llama.cpp release tag recall will use. Env
// override wins so the user can roll forward without rebuilding.
func Version() string {
	if v := os.Getenv("RECALL_LLAMACPP_VERSION"); v != "" {
		return v
	}
	return DefaultVersion
}

// BinDir is where recall keeps the extracted llama.cpp tree. We
// version the directory so a version bump downloads alongside the
// old one rather than overwriting mid-flight.
//
//	~/.recall/bin/llamacpp/b8798/llama-server
//	~/.recall/bin/llamacpp/b8798/libllama.dylib
//	...
func BinDir() (string, error) {
	if d := os.Getenv("RECALL_LLAMACPP_DIR"); d != "" {
		return filepath.Join(d, Version()), nil
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("resolve home dir: %w", err)
	}
	return filepath.Join(home, ".recall", "bin", "llamacpp", Version()), nil
}

// ServerPath returns the absolute path to the llama-server binary.
// Existence is not implied — call EnsureBinary first.
func ServerPath() (string, error) {
	dir, err := BinDir()
	if err != nil {
		return "", err
	}
	exe := ServerBinary
	if runtime.GOOS == "windows" {
		exe += ".exe"
	}
	return filepath.Join(dir, exe), nil
}

// IsInstalled returns true when a usable llama-server is on disk
// for the active version. recall doctor prints this without
// triggering a download.
func IsInstalled() bool {
	p, err := ServerPath()
	if err != nil {
		return false
	}
	st, err := os.Stat(p)
	if err != nil {
		return false
	}
	return !st.IsDir() && st.Mode()&0o111 != 0
}

var ensureMu sync.Mutex

// EnsureBinary makes sure llama-server exists on disk, downloading
// + extracting the upstream archive on first use. Subsequent calls
// are no-ops. Safe for concurrent invocation; the mutex is fair
// enough for the one-shot CLI use case.
func EnsureBinary(ctx context.Context) (string, error) {
	ensureMu.Lock()
	defer ensureMu.Unlock()

	if IsInstalled() {
		return ServerPath()
	}

	asset, isZip, err := assetForPlatform(Version())
	if err != nil {
		return "", err
	}
	dir, err := BinDir()
	if err != nil {
		return "", err
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", fmt.Errorf("create %s: %w", dir, err)
	}

	url := fmt.Sprintf("https://github.com/ggml-org/llama.cpp/releases/download/%s/%s", Version(), asset)
	fmt.Fprintf(os.Stderr, "recall: downloading llama.cpp %s prebuilt (%s)…\n", Version(), asset)

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return "", fmt.Errorf("build download request: %w", err)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("download %s: %w", url, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("download %s: HTTP %s", url, resp.Status)
	}

	if isZip {
		if err := extractZip(resp.Body, dir); err != nil {
			return "", fmt.Errorf("extract zip: %w", err)
		}
	} else {
		if err := extractTarGz(resp.Body, dir); err != nil {
			return "", fmt.Errorf("extract tar.gz: %w", err)
		}
	}

	srv, err := ServerPath()
	if err != nil {
		return "", err
	}
	if !IsInstalled() {
		return "", fmt.Errorf("download finished but %s missing or not executable", srv)
	}
	// Sanity probe — fail loudly here instead of at first embed.
	if err := probeBinary(srv); err != nil {
		return "", fmt.Errorf("downloaded binary failed sanity probe: %w", err)
	}
	return srv, nil
}

// probeBinary runs `llama-server --version` and checks the process
// exits cleanly. Catches the "downloaded a Linux build on macOS"
// class of mistake before the user hits a real embed call.
func probeBinary(path string) error {
	ctx, cancel := context.WithTimeout(context.Background(), 30_000_000_000)
	defer cancel()
	out, err := exec.CommandContext(ctx, path, "--version").CombinedOutput()
	if err != nil {
		return fmt.Errorf("%s --version: %w (output: %s)", filepath.Base(path), err, strings.TrimSpace(string(out)))
	}
	return nil
}

// keepFile reports whether an entry from the llama.cpp archive is
// worth extracting. recall only ever invokes llama-server; the
// rest of the archive (llama-cli, llama-bench, llama-perplexity,
// llama-llava-cli, llama-tts, llama-quantize, llama-imatrix …) is
// ~100 MB of CLIs we never touch. Filter them out on the way in
// and cut the install footprint roughly in half.
//
// Rules:
//  1. Keep llama-server (+ .exe on Windows).
//  2. Keep every lib*.dylib / lib*.so / lib*.dll. llama-server
//     hard-links against libllama, libggml, libggml-base, and
//     libmtmd at build time (mtmd is bundled into the binary's
//     @rpath even though recall never uses multimodal features);
//     it also dlopens the libggml-cpu-* variants at runtime. A
//     simple lib-prefix rule keeps every shared object without
//     having to track which are linked vs dlopened per platform.
//  3. Skip every other executable (llama-cli, llama-bench, etc.).
//  4. Keep LICENSE (tiny, good hygiene for a redistributed binary).
//
// Returns true if the entry should land on disk.
func keepFile(name string) bool {
	base := filepath.Base(name)
	switch base {
	case "llama-server", "llama-server.exe":
		return true
	case "LICENSE":
		return true
	}
	// Any shared library — llama-server links several of them by
	// name (libllama, libggml, libmtmd on macOS) and dlopens the
	// libggml-cpu-* variants at runtime. Filtering mtmd out caused
	// dyld to abort on macOS; keep the whole lib*.(so|dylib|dll)
	// set and trust the archive to ship only what llama-server
	// needs.
	if strings.HasPrefix(base, "lib") {
		return true
	}
	return false
}

// extractTarGz unpacks a release archive into dir, flattening the
// top-level "llama-bXXXX/" directory so users find llama-server
// directly under BinDir. Only files matching keepFile are written;
// everything else is skipped.
func extractTarGz(r io.Reader, dir string) error {
	gz, err := gzip.NewReader(r)
	if err != nil {
		return err
	}
	defer gz.Close()
	tr := tar.NewReader(gz)
	for {
		hdr, err := tr.Next()
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return err
		}
		name := stripTopDir(hdr.Name)
		if name == "" {
			continue
		}
		if !keepFile(name) && hdr.Typeflag != tar.TypeDir {
			continue
		}
		out := filepath.Join(dir, name)
		switch hdr.Typeflag {
		case tar.TypeDir:
			if err := os.MkdirAll(out, 0o755); err != nil {
				return err
			}
		case tar.TypeReg, tar.TypeRegA:
			if err := writeFile(out, tr, os.FileMode(hdr.Mode)); err != nil {
				return err
			}
		case tar.TypeSymlink:
			// llama.cpp archives ship dylib version symlinks
			// (libllama.dylib → libllama.0.dylib). Honour them —
			// but only if keepFile cleared the target name, which
			// the guard above already checked.
			_ = os.Remove(out)
			if err := os.Symlink(hdr.Linkname, out); err != nil {
				return fmt.Errorf("symlink %s → %s: %w", out, hdr.Linkname, err)
			}
		}
	}
}

func extractZip(r io.Reader, dir string) error {
	// Buffer to disk because zip.NewReader needs an io.ReaderAt.
	tmp, err := os.CreateTemp("", "recall-llamacpp-*.zip")
	if err != nil {
		return err
	}
	defer os.Remove(tmp.Name())
	if _, err := io.Copy(tmp, r); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	zr, err := zip.OpenReader(tmp.Name())
	if err != nil {
		return err
	}
	defer zr.Close()
	for _, f := range zr.File {
		name := stripTopDir(f.Name)
		if name == "" {
			continue
		}
		if !f.FileInfo().IsDir() && !keepFile(name) {
			continue
		}
		out := filepath.Join(dir, name)
		if f.FileInfo().IsDir() {
			if err := os.MkdirAll(out, 0o755); err != nil {
				return err
			}
			continue
		}
		rc, err := f.Open()
		if err != nil {
			return err
		}
		err = writeFile(out, rc, f.Mode())
		rc.Close()
		if err != nil {
			return err
		}
	}
	return nil
}

// stripTopDir removes the top-level directory from a tar/zip entry
// path. The llama.cpp archive layout is "llama-bXXXX/llama-server",
// "llama-bXXXX/libggml.dylib", etc. — we want everything to land
// flat in BinDir.
func stripTopDir(name string) string {
	name = strings.TrimPrefix(name, "./")
	idx := strings.IndexByte(name, '/')
	if idx < 0 {
		return ""
	}
	return name[idx+1:]
}

func writeFile(path string, r io.Reader, mode os.FileMode) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, mode)
	if err != nil {
		return err
	}
	if _, err := io.Copy(f, r); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}
