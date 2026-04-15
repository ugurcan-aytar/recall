package commands

import (
	"fmt"
	"os"

	"github.com/ugurcan-aytar/recall/internal/embed"
	"github.com/ugurcan-aytar/recall/internal/llm"
)

// generatorOverride lets tests inject a pre-built Generator so they
// don't need libbinding.a + a real GGUF model. When non-nil,
// openGenerator returns it instead of building a real local
// generator.
var generatorOverride llm.Generator

// openGenerator returns the LLM Generator implied by the current
// environment. Today there's only one path — local GGUF behind the
// `embed_llama` build tag — but the function exists as the lazy-load
// entry point so future API generators (Anthropic / OpenAI / etc.)
// can slot in alongside the local one without touching every caller.
//
// Stub builds (default `go build`) return llm.ErrLocalGeneratorNotCompiled
// so the --expand / --rerank / --hyde paths can branch on it.
func openGenerator() (llm.Generator, error) {
	if generatorOverride != nil {
		return generatorOverride, nil
	}
	if !llm.LocalGeneratorAvailable() {
		return nil, llm.ErrLocalGeneratorNotCompiled
	}
	modelPath, err := embed.ResolveActiveExpansionModelPath()
	if err != nil {
		return nil, err
	}
	if _, err := os.Stat(modelPath); err != nil {
		return nil, fmt.Errorf(
			"expansion model not found at %s — run `recall models download --expansion` "+
				"or set RECALL_EXPAND_MODEL: %w",
			modelPath, err,
		)
	}
	return llm.NewLocalGenerator(llm.LocalGeneratorOptions{ModelPath: modelPath})
}

// SetGeneratorOverride is exported for cross-package tests that need
// to inject a MockGenerator into the commands package without
// reaching into private state. Pass nil to clear.
func SetGeneratorOverride(g llm.Generator) { generatorOverride = g }
