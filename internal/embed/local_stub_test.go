//go:build !embed_llama

package embed

import (
	"errors"
	"testing"
)

func TestLocalEmbedderStubReturnsError(t *testing.T) {
	if LocalEmbedderAvailable() {
		t.Fatal("LocalEmbedderAvailable should be false in default build")
	}
	_, err := NewLocalEmbedder(LocalEmbedderOptions{ModelPath: "/tmp/missing.gguf"})
	if err == nil {
		t.Fatal("expected error from stub NewLocalEmbedder")
	}
	if !errors.Is(err, ErrLocalEmbedderNotCompiled) {
		t.Errorf("error should wrap ErrLocalEmbedderNotCompiled, got %v", err)
	}
}
