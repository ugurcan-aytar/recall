package commands

import (
	"fmt"
	"os"

	"github.com/ugurcan-aytar/recall/internal/embed"
	"github.com/ugurcan-aytar/recall/internal/llm"
)

// generatorOverride lets tests inject a pre-built Generator for the
// expansion / HyDE path so they don't need libbinding.a + a real
// GGUF model. When non-nil, openGenerator returns it instead of
// building a real local generator.
var generatorOverride llm.Generator

// rerankerOverride lets tests inject a pre-built Reranker for the
// --rerank path. Pre-v0.2.4 this was a Generator (binary yes/no
// prompt); v0.2.4 switched to a real cross-encoder via
// llm.NewLocalReranker + llama-server's /v1/rerank endpoint.
var rerankerOverride llm.Reranker

// openGenerator returns the LLM Generator implied by the current
// environment for expansion / HyDE. Today there's only one path —
// local GGUF behind the `embed_llama` build tag — but the function
// exists as the lazy-load entry point so future API generators
// (Anthropic / OpenAI / etc.) can slot in alongside the local one
// without touching every caller.
//
// Stub builds (default `go build`) return llm.ErrLocalGeneratorNotCompiled
// so the --expand / --hyde paths can branch on it.
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

// openReranker returns the cross-encoder reranker implied by the
// current environment for --rerank. v0.2.4 switched the reranker
// from a binary-yes/no Generator (Qwen2.5-1.5B-Instruct) to a real
// cross-encoder via llm.NewLocalReranker and llama-server's
// /v1/rerank endpoint (default model: bge-reranker-v2-m3).
func openReranker() (llm.Reranker, error) {
	if rerankerOverride != nil {
		return rerankerOverride, nil
	}
	if !llm.LocalRerankerAvailable() {
		return nil, llm.ErrLocalRerankerNotAvailable
	}
	modelPath, err := embed.ResolveActiveRerankerModelPath()
	if err != nil {
		return nil, err
	}
	if _, err := os.Stat(modelPath); err != nil {
		return nil, fmt.Errorf(
			"reranker model not found at %s — run `recall models download --reranker` "+
				"or set RECALL_RERANK_MODEL: %w",
			modelPath, err,
		)
	}
	return llm.NewLocalReranker(llm.LocalRerankerOptions{ModelPath: modelPath})
}

// SetGeneratorOverride is exported for cross-package tests that need
// to inject a MockGenerator into the commands package without
// reaching into private state. Pass nil to clear.
func SetGeneratorOverride(g llm.Generator) { generatorOverride = g }

// SetRerankerOverride injects a pre-built Reranker for the --rerank
// path (tests). Pass nil to clear.
func SetRerankerOverride(r llm.Reranker) { rerankerOverride = r }
