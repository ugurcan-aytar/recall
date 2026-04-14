package store

import (
	sqlite_vec "github.com/asg017/sqlite-vec-go-bindings/cgo"
)

// EmbeddingDimensions is the fixed vector width of chunk_embeddings. The
// value must match whatever Embedder the user runs — the vec0 virtual
// table is created with this dimension at migration time and is not
// resizable without a schema migration.
//
// 768 matches nomic-embed-text-v1.5 (the default local model — Apache
// 2.0, ungated) and most other small transformer embedders. The OpenAI
// API embedder requests `dimensions: 768` to fit the same schema.
const EmbeddingDimensions = 768

// init registers sqlite-vec as an auto-loaded SQLite extension. This is
// the ONE init() the project allows (see CLAUDE.md "No init() functions
// except for sqlite-vec registration"). It must run before any call to
// sql.Open("sqlite3", ...).
func init() {
	sqlite_vec.Auto()
}

// SerializeEmbedding wraps sqlite_vec.SerializeFloat32 so callers outside
// this package don't import the binding directly.
func SerializeEmbedding(v []float32) ([]byte, error) {
	return sqlite_vec.SerializeFloat32(v)
}
