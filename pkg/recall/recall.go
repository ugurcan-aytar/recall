// Package recall is the public Go API for the recall search engine.
//
// External consumers (in particular the brain CLI) import this package
// instead of touching internal/. The surface is intentionally small: open
// an Engine, register collections, index, embed, and search.
//
//	eng, err := recall.NewEngine(recall.WithDBPath("./index.db"))
//	if err != nil { ... }
//	defer eng.Close()
//	_ = eng.AddCollection("notes", "/path/to/notes")
//	_, _ = eng.Index()
//	_, _ = eng.Embed(myEmbedder)
//	results, _ := eng.SearchHybrid("rate limiting")
//
// Stability: pre-1.0. Function and option names may change between
// minor versions until v1.0.0.
package recall

import (
	"errors"
	"fmt"

	"github.com/ugurcan-aytar/recall/internal/embed"
	"github.com/ugurcan-aytar/recall/internal/store"
)

// SearchResult is one document's worth of search output. Aliased
// to [store.SearchResult] so external consumers can pass return
// values from Engine.SearchBM25 / SearchVector straight into
// [Rerank] without the internal import.
type SearchResult = store.SearchResult

// FusedResult is the RRF-fused hybrid-search output. Aliased to
// [store.FusedResult]; embeds SearchResult.
type FusedResult = store.FusedResult

// Engine is recall's high-level handle. Wraps a *store.Store and (when
// supplied) an Embedder. Safe for concurrent use.
type Engine struct {
	store *store.Store
}

// Option configures a new Engine.
type Option func(*config)

type config struct {
	dbPath string
}

// WithDBPath overrides the SQLite database location. Defaults to
// ~/.recall/index.db (or $RECALL_DB_PATH when set).
func WithDBPath(path string) Option {
	return func(c *config) { c.dbPath = path }
}

// ResolveIndexPath maps a --index name to its ~/.recall/indexes/<name>.db
// file path. Library consumers (brain) use it to share the same
// named-index convention the CLI's --index flag enforces. Name
// sanitisation: alphanumerics, dash, underscore only — no path
// separators, no "..". See [store.ResolveIndexPath] for the full doc.
func ResolveIndexPath(name string) (string, error) {
	return store.ResolveIndexPath(name)
}

// NewEngine opens (or creates) the database and prepares the engine.
func NewEngine(opts ...Option) (*Engine, error) {
	cfg := &config{}
	for _, o := range opts {
		o(cfg)
	}
	s, err := store.Open(cfg.dbPath)
	if err != nil {
		return nil, fmt.Errorf("open store: %w", err)
	}
	return &Engine{store: s}, nil
}

// Close releases all resources held by the engine.
func (e *Engine) Close() error {
	return e.store.Close()
}

// ---- collections ----------------------------------------------------------

// AddCollection registers a folder as a collection. name "" defaults to
// the folder basename, glob "" uses store.DefaultGlobPattern.
func (e *Engine) AddCollection(name, path string, glob, context string) (*store.Collection, error) {
	return e.store.AddCollection(name, path, glob, context)
}

// RemoveCollection deletes a collection and cascades to its documents.
func (e *Engine) RemoveCollection(name string) error {
	return e.store.RemoveCollection(name)
}

// ListCollections returns every registered collection with doc counts.
func (e *Engine) ListCollections() ([]store.Collection, error) {
	return e.store.ListCollections()
}

// ---- indexing -------------------------------------------------------------

// IndexResult aggregates per-collection IndexStats for a single Index call.
type IndexResult struct {
	PerCollection map[string]store.IndexStats
}

// Index re-scans every registered collection.
func (e *Engine) Index() (*IndexResult, error) {
	cols, err := e.store.ListCollections()
	if err != nil {
		return nil, err
	}
	out := &IndexResult{PerCollection: map[string]store.IndexStats{}}
	for _, c := range cols {
		stats, err := e.store.IndexCollection(c.ID)
		if err != nil {
			return nil, fmt.Errorf("index %s: %w", c.Name, err)
		}
		out.PerCollection[c.Name] = stats
	}
	return out, nil
}

// ---- embedding ------------------------------------------------------------

// EmbedResult reports how many chunks were embedded.
type EmbedResult struct {
	Embedded int
	Total    int
}

// Embed embeds every chunk that is not yet embedded. force=true drops all
// existing embeddings and re-embeds the entire corpus.
func (e *Engine) Embed(emb embed.Embedder, force bool) (*EmbedResult, error) {
	if emb == nil {
		return nil, errors.New("embedder is required")
	}
	if emb.Dimensions() != store.EmbeddingDimensions {
		return nil, fmt.Errorf(
			"embedder produces %d-dim vectors but chunk_embeddings expects %d",
			emb.Dimensions(), store.EmbeddingDimensions,
		)
	}

	var chunks []store.ChunkForEmbed
	var err error
	if force {
		chunks, err = e.store.AllChunksForEmbed()
		if err != nil {
			return nil, err
		}
		if err := e.store.DropAllEmbeddings(); err != nil {
			return nil, err
		}
	} else {
		chunks, err = e.store.ChunksNeedingEmbed()
		if err != nil {
			return nil, err
		}
	}

	family := emb.Family()
	const batchSize = 32 // matches commands/embed.go's orchestrator batch
	for i := 0; i < len(chunks); i += batchSize {
		end := i + batchSize
		if end > len(chunks) {
			end = len(chunks)
		}
		batch := chunks[i:end]
		texts := make([]string, len(batch))
		for j, ch := range batch {
			texts[j] = embed.FormatDocumentFor(family, ch.DocTitle, ch.Content)
		}
		// Backend-internal worker pool (when configured) parallelises
		// here. The orchestrator stays sequential at the batch
		// boundary so write-back to SQLite is one transaction's worth
		// at a time.
		vecs, err := emb.Embed(texts)
		if err != nil {
			return nil, fmt.Errorf("embed chunks %d..%d: %w", i, end, err)
		}
		for j, v := range vecs {
			if err := e.store.UpsertEmbedding(batch[j].ID, v); err != nil {
				return nil, err
			}
		}
	}
	if err := e.store.SetMetadata("embedding_model", emb.ModelName()); err != nil {
		return nil, err
	}
	return &EmbedResult{Embedded: len(chunks), Total: len(chunks)}, nil
}

// ---- search ---------------------------------------------------------------

// SearchOption mutates a search request.
type SearchOption func(*store.SearchOptions)

// WithLimit caps the number of returned results.
func WithLimit(n int) SearchOption {
	return func(o *store.SearchOptions) { o.Limit = n }
}

// WithCollection restricts the search to one collection by name.
func WithCollection(name string) SearchOption {
	return func(o *store.SearchOptions) { o.Collection = name }
}

// WithMinScore drops results below the given score.
func WithMinScore(s float64) SearchOption {
	return func(o *store.SearchOptions) { o.MinScore = s }
}

// SearchBM25 runs a full-text search.
func (e *Engine) SearchBM25(query string, opts ...SearchOption) ([]store.SearchResult, error) {
	so := store.SearchOptions{Query: query}
	for _, o := range opts {
		o(&so)
	}
	return e.store.SearchBM25(so)
}

// SearchVector runs a vector-only search using emb to embed the query.
func (e *Engine) SearchVector(emb embed.Embedder, query string, opts ...SearchOption) ([]store.SearchResult, error) {
	if emb == nil {
		return nil, errors.New("embedder is required")
	}
	v, err := emb.EmbedSingle(embed.FormatQueryFor(emb.Family(), query))
	if err != nil {
		return nil, err
	}
	so := store.SearchOptions{Query: query}
	for _, o := range opts {
		o(&so)
	}
	return e.store.SearchVector(v, so)
}

// SearchHybrid runs BM25 + vector and fuses with default RRF parameters.
// When emb is nil OR no embeddings exist yet, gracefully degrades to BM25.
func (e *Engine) SearchHybrid(emb embed.Embedder, query string, opts ...SearchOption) ([]store.FusedResult, error) {
	so := store.SearchOptions{Query: query}
	for _, o := range opts {
		o(&so)
	}

	bm25, err := e.store.SearchBM25(so)
	if err != nil {
		return nil, fmt.Errorf("bm25: %w", err)
	}

	count, err := e.store.EmbeddingCount()
	if err != nil || count == 0 || emb == nil {
		// Graceful degradation — promote BM25 results to FusedResult shape.
		out := make([]store.FusedResult, len(bm25))
		for i, r := range bm25 {
			out[i] = store.FusedResult{SearchResult: r, FusedScore: r.Score}
		}
		return out, nil
	}

	v, err := emb.EmbedSingle(embed.FormatQueryFor(emb.Family(), query))
	if err != nil {
		return nil, err
	}
	vec, err := e.store.SearchVector(v, so)
	if err != nil {
		return nil, fmt.Errorf("vector: %w", err)
	}
	return store.FuseRRF(bm25, vec, store.DefaultFusionOptions()), nil
}

// ---- retrieval ------------------------------------------------------------

// Get fetches a single document by relative path or "#docid".
func (e *Engine) Get(spec string) (*store.Document, error) {
	return e.store.GetDocument(spec)
}

// MultiGet returns every document matching a glob pattern.
func (e *Engine) MultiGet(pattern string) ([]store.Document, error) {
	return e.store.MultiGetGlob(pattern)
}

// ---- context --------------------------------------------------------------

// AddContext attaches a description to a (collection, path).
func (e *Engine) AddContext(collection, path, text string) error {
	return e.store.AddContext(collection, path, text)
}

// ListContexts returns every registered context.
func (e *Engine) ListContexts() ([]store.PathContext, error) {
	return e.store.ListContexts()
}

// Store exposes the underlying *store.Store. Provided for advanced
// callers; most code should use the methods above.
func (e *Engine) Store() *store.Store { return e.store }
