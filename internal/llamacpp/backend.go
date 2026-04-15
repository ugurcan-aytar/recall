// Package llamacpp owns the process-wide gollama.cpp backend
// lifecycle. dianlight/gollama.cpp's Backend_init / Backend_free
// touch global llama.cpp state — calling them from independent
// goroutines is undefined behaviour. Both internal/embed and
// internal/llm need the backend up before they can construct
// models, so this tiny shared package wraps the init in a
// sync.Once and the recommended download-on-first-use pattern.
//
// Backend_free is intentionally never called: recall is a one-shot
// CLI, the OS reclaims everything on exit. Long-running embedders
// (brain consumers) pay one Backend_init at process start and
// hold the cost for the rest of the run.
package llamacpp

import (
	"fmt"
	"sync"

	gollama "github.com/dianlight/gollama.cpp"
)

var (
	once sync.Once
	err  error
)

// EnsureBackend brings the gollama.cpp backend up exactly once
// per process. The first call may block for a multi-second
// shared-library download (~100 MB to ~/.cache/gollama/libs/);
// subsequent calls return the cached error (if any) immediately.
func EnsureBackend() error {
	once.Do(func() {
		// Try the cheap path first. If the platform shared library
		// isn't on disk yet, dianlight returns an init error; the
		// recommended recovery is download-then-retry.
		if e := gollama.Backend_init(); e == nil {
			return
		}
		if dlErr := gollama.LoadLibraryWithVersion(""); dlErr != nil {
			err = fmt.Errorf("download llama.cpp library: %w", dlErr)
			return
		}
		if e := gollama.Backend_init(); e != nil {
			err = fmt.Errorf("init llama.cpp backend after download: %w", e)
		}
	})
	return err
}
