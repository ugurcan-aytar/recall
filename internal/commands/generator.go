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

// rerankGeneratorOverride is the same hook for the reranker model.
// Expansion and reranking ship with different default models
// (qmd-query-expansion-1.7B vs Qwen2.5-1.5B-Instruct) because the
// expansion fine-tune can't follow generic yes/no prompts — see
// CLAUDE.md "Reranker fallback" for the empirical evidence.
var rerankGeneratorOverride llm.Generator

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

// openRerankGenerator returns the LLM Generator implied by the
// current environment for the reranker (--rerank). Same shape as
// openGenerator but consults RECALL_RERANK_MODEL and the reranker
// default (Qwen2.5-1.5B-Instruct).
func openRerankGenerator() (llm.Generator, error) {
	if rerankGeneratorOverride != nil {
		return rerankGeneratorOverride, nil
	}
	if !llm.LocalGeneratorAvailable() {
		return nil, llm.ErrLocalGeneratorNotCompiled
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
	return llm.NewLocalGenerator(llm.LocalGeneratorOptions{ModelPath: modelPath})
}

// SetGeneratorOverride is exported for cross-package tests that need
// to inject a MockGenerator into the commands package without
// reaching into private state. Pass nil to clear.
func SetGeneratorOverride(g llm.Generator) { generatorOverride = g }

// SetRerankGeneratorOverride is the same hook for the reranker
// generator. Pass nil to clear.
func SetRerankGeneratorOverride(g llm.Generator) { rerankGeneratorOverride = g }
