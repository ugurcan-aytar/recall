// llamacpp.Server wraps a single running llama-server subprocess.
// The process is bound to a Unix socket inside the OS temp dir so
// recall doesn't fight other tools for TCP ports and the socket
// can never collide with another recall invocation.
//
// Lifecycle:
//
//	srv, err := llamacpp.StartServer(ctx, llamacpp.ServerOptions{...})
//	defer srv.Close()
//	body, err := srv.PostJSON(ctx, "/v1/embeddings", req)
//
// The server prints llama.cpp boot logs on its own stderr; recall
// tees them through to its parent stderr so the user sees Metal
// init messages on first run.
package llamacpp

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"sync"
	"time"
)

// ServerOptions configures a llama-server boot.
type ServerOptions struct {
	ModelPath   string // .gguf path; required
	Embedding   bool   // pass --embedding (lock the server to embedding-only)
	Pooling     string // "mean" | "cls" | "last" | "rank"; "" = model default
	Context     int    // n_ctx; 0 = let llama.cpp choose from model
	BatchSize   int    // -b; 0 = llama.cpp default
	UBatchSize  int    // -ub; 0 = llama.cpp default
	Parallel    int    // -np; 0 = let llama.cpp pick
	Threads     int    // -t; 0 = llama.cpp default
	GPULayers   int    // -ngl; -1 = unset; 0 = force CPU
	StderrSink  io.Writer // optional log destination, defaults to os.Stderr
}

// Server is a running llama-server bound to a Unix socket.
type Server struct {
	cmd        *exec.Cmd
	socketPath string
	httpClient *http.Client
	closeOnce  sync.Once
}

// StartServer downloads (if needed), execs llama-server, and waits
// for it to begin accepting requests. Returns a Server the caller
// must Close() — failure to do so leaves an orphaned subprocess
// and dangling .sock file.
func StartServer(ctx context.Context, opts ServerOptions) (*Server, error) {
	if opts.ModelPath == "" {
		return nil, errors.New("llamacpp.StartServer: ModelPath is required")
	}
	if runtime.GOOS == "windows" {
		// Windows doesn't have proper Unix sockets in net/http until
		// recent Go versions, and llama-server's --host parsing also
		// differs. v0.2.2 ships macOS + Linux only; Windows users
		// keep the API embedder fallback.
		return nil, fmt.Errorf("%w: llama-server pattern not yet supported on windows", ErrUnsupportedPlatform)
	}

	bin, err := EnsureBinary(ctx)
	if err != nil {
		return nil, err
	}

	socketPath, err := makeSocketPath()
	if err != nil {
		return nil, err
	}

	args := []string{
		"--host", socketPath,
		// No --port: llama-server treats --port 0 as "invalid TCP
		// port" and refuses to bind even when --host is a Unix
		// socket. Omitting the flag lets the socket bind cleanly.
		"--no-webui",
		// Suppress llama-server's startup warmup, which writes a
		// short chat template exchange ("Hi there<|im_end|>", etc.)
		// to the same stream as user-facing output and confuses
		// recall query / vsearch. The first real request takes a
		// few milliseconds longer; functionally identical.
		"--no-warmup",
		"-m", opts.ModelPath,
	}
	if opts.Embedding {
		args = append(args, "--embedding")
	}
	if opts.Pooling != "" {
		args = append(args, "--pooling", opts.Pooling)
	}
	if opts.Context > 0 {
		args = append(args, "-c", strconv.Itoa(opts.Context))
	}
	if opts.BatchSize > 0 {
		args = append(args, "-b", strconv.Itoa(opts.BatchSize))
	}
	if opts.UBatchSize > 0 {
		args = append(args, "-ub", strconv.Itoa(opts.UBatchSize))
	}
	if opts.Parallel > 0 {
		args = append(args, "-np", strconv.Itoa(opts.Parallel))
	}
	if opts.Threads > 0 {
		args = append(args, "-t", strconv.Itoa(opts.Threads))
	}
	if opts.GPULayers >= 0 {
		args = append(args, "-ngl", strconv.Itoa(opts.GPULayers))
	}

	// llama-server is chatty: Metal init, model loader, chat template
	// preview, per-request slot logs. None of that is useful in the
	// CLI happy path, and the chat-template preview specifically
	// looks like garbage in `recall query` output. Default to a
	// rotating per-server log file under the socket dir; surface to
	// real stderr only when the user opts in via $RECALL_LLAMACPP_LOG=stderr
	// or by passing opts.StderrSink.
	stderr := opts.StderrSink
	if stderr == nil {
		switch os.Getenv("RECALL_LLAMACPP_LOG") {
		case "stderr":
			stderr = os.Stderr
		case "off", "discard":
			stderr = io.Discard
		default:
			logPath := filepath.Join(filepath.Dir(socketPath), "server.log")
			f, err := os.OpenFile(logPath, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o644)
			if err != nil {
				stderr = io.Discard
			} else {
				stderr = f
			}
		}
	}

	cmd := exec.CommandContext(ctx, bin, args...)
	cmd.Stdout = stderr // llama-server's logs go to stdout for some lines
	cmd.Stderr = stderr
	// Make sure the dylibs next to the binary are findable without
	// LD_LIBRARY_PATH / DYLD_LIBRARY_PATH gymnastics — they ship in
	// the same directory and llama.cpp's @loader_path rpath handles
	// that automatically as long as we keep the layout flat.
	cmd.Dir = filepath.Dir(bin)
	if err := cmd.Start(); err != nil {
		os.Remove(socketPath)
		return nil, fmt.Errorf("start llama-server: %w", err)
	}

	srv := &Server{
		cmd:        cmd,
		socketPath: socketPath,
		httpClient: newUnixHTTPClient(socketPath),
	}

	if err := srv.waitReady(ctx); err != nil {
		_ = srv.Close()
		return nil, err
	}
	return srv, nil
}

// Close stops the subprocess and removes the socket file. Idempotent.
func (s *Server) Close() error {
	var firstErr error
	s.closeOnce.Do(func() {
		if s.cmd != nil && s.cmd.Process != nil {
			// SIGTERM gives llama-server a chance to flush stats;
			// the OS reaps the rest if it ignores us.
			_ = s.cmd.Process.Signal(os.Interrupt)
			done := make(chan error, 1)
			go func() { done <- s.cmd.Wait() }()
			select {
			case err := <-done:
				if err != nil && !isExpectedExitErr(err) {
					firstErr = err
				}
			case <-time.After(5 * time.Second):
				_ = s.cmd.Process.Kill()
				<-done
			}
		}
		_ = os.Remove(s.socketPath)
	})
	return firstErr
}

// PostJSON marshals body, posts it to path on the server, and
// unmarshals the response into out (if non-nil).
func (s *Server) PostJSON(ctx context.Context, path string, body, out any) error {
	buf, err := json.Marshal(body)
	if err != nil {
		return fmt.Errorf("marshal request: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, "http://llamacpp"+path, bytes.NewReader(buf))
	if err != nil {
		return fmt.Errorf("build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := s.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("post %s: %w", path, err)
	}
	defer resp.Body.Close()
	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("read response: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("llama-server %s returned HTTP %d: %s", path, resp.StatusCode, truncate(string(respBody), 256))
	}
	if out == nil {
		return nil
	}
	if err := json.Unmarshal(respBody, out); err != nil {
		return fmt.Errorf("unmarshal response: %w", err)
	}
	return nil
}

// SocketPath exposes the server's Unix socket for callers that need
// to construct their own HTTP client.
func (s *Server) SocketPath() string { return s.socketPath }

// HTTPClient returns the configured Unix-socket client.
func (s *Server) HTTPClient() *http.Client { return s.httpClient }

// waitReady polls /health until the server responds 200 OK or the
// context cancels. The first successful response usually arrives
// within ~2 seconds on Apple Silicon for nomic-embed.
func (s *Server) waitReady(ctx context.Context) error {
	deadline := time.Now().Add(60 * time.Second)
	for {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		if time.Now().After(deadline) {
			return errors.New("llama-server did not become ready within 60s")
		}
		// Subprocess may have already crashed (bad model, missing
		// dylib, …). Detect that without leaving the user staring
		// at a hang.
		if s.cmd.ProcessState != nil && s.cmd.ProcessState.Exited() {
			return fmt.Errorf("llama-server exited before becoming ready (code %d)", s.cmd.ProcessState.ExitCode())
		}

		req, err := http.NewRequestWithContext(ctx, http.MethodGet, "http://llamacpp/health", nil)
		if err != nil {
			return err
		}
		resp, err := s.httpClient.Do(req)
		if err == nil {
			resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				return nil
			}
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(150 * time.Millisecond):
		}
	}
}

func makeSocketPath() (string, error) {
	dir, err := os.MkdirTemp("", "recall-llama-")
	if err != nil {
		return "", fmt.Errorf("create socket dir: %w", err)
	}
	// llama-server treats --host as a Unix socket only when the
	// path ends in ".sock"; anything else is parsed as an IP/host
	// and bind fails. Keep the suffix.
	return filepath.Join(dir, "s.sock"), nil
}

func newUnixHTTPClient(socketPath string) *http.Client {
	tr := &http.Transport{
		DialContext: func(ctx context.Context, _, _ string) (net.Conn, error) {
			var d net.Dialer
			return d.DialContext(ctx, "unix", socketPath)
		},
		// Disable connection reuse heuristics that don't apply
		// to a single-process Unix peer; keep things simple.
		MaxIdleConns:          4,
		IdleConnTimeout:       30 * time.Second,
		ResponseHeaderTimeout: 5 * time.Minute, // long batches can take a while
	}
	return &http.Client{Transport: tr}
}

func isExpectedExitErr(err error) bool {
	var exitErr *exec.ExitError
	if errors.As(err, &exitErr) {
		// SIGINT-induced exit on macOS shows up as exit status -1 or 130.
		return true
	}
	return false
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}
