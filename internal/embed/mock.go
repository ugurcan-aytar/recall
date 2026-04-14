package embed

import (
	"errors"
	"hash/fnv"
	"math"
)

// MockEmbedder is a deterministic embedder used by every test in the
// project. It produces stable vectors derived from the FNV-1a hash of the
// input text. CLAUDE.md mandates that no test downloads or loads a real
// GGUF model — use this type instead.
//
// Lives in a non-test file (rather than embed_test.go) so other packages'
// _test.go files can import and reuse it.
type MockEmbedder struct {
	Dims   int    // 0 ⇒ store.EmbeddingDimensions
	Name   string // "" ⇒ "mock-embed"
	closed bool
}

// NewMockEmbedder returns a MockEmbedder with the given dimensionality.
// Pass 0 to use the default (768, matching the chunk_embeddings schema).
func NewMockEmbedder(dims int) *MockEmbedder {
	if dims <= 0 {
		dims = 768
	}
	return &MockEmbedder{Dims: dims, Name: "mock-embed"}
}

// Embed returns one deterministic vector per input.
func (m *MockEmbedder) Embed(texts []string) ([][]float32, error) {
	if m.closed {
		return nil, errors.New("mock embedder is closed")
	}
	out := make([][]float32, len(texts))
	for i, t := range texts {
		out[i] = m.vectorFor(t)
	}
	return out, nil
}

// EmbedSingle is the one-shot variant.
func (m *MockEmbedder) EmbedSingle(text string) ([]float32, error) {
	v, err := m.Embed([]string{text})
	if err != nil {
		return nil, err
	}
	return v[0], nil
}

// Dimensions reports the vector width.
func (m *MockEmbedder) Dimensions() int {
	if m.Dims <= 0 {
		return 768
	}
	return m.Dims
}

// ModelName reports the stable mock identifier.
func (m *MockEmbedder) ModelName() string {
	if m.Name == "" {
		return "mock-embed"
	}
	return m.Name
}

// Close releases the (no-op) resources.
func (m *MockEmbedder) Close() error {
	m.closed = true
	return nil
}

// vectorFor produces a deterministic, non-zero vector for the given text.
// Each component is sin(seed × (i+1)), where seed comes from FNV-1a over
// the text. Identical text always yields identical vectors.
func (m *MockEmbedder) vectorFor(text string) []float32 {
	h := fnv.New32a()
	_, _ = h.Write([]byte(text))
	seed := float64(h.Sum32())

	dim := m.Dimensions()
	v := make([]float32, dim)
	var norm float64
	for i := 0; i < dim; i++ {
		raw := math.Sin(seed * float64(i+1))
		v[i] = float32(raw)
		norm += raw * raw
	}
	// L2-normalise so that cosine distance is meaningful.
	norm = math.Sqrt(norm)
	if norm == 0 {
		return v
	}
	for i := range v {
		v[i] = float32(float64(v[i]) / norm)
	}
	return v
}

// Static interface-conformance check.
var _ Embedder = (*MockEmbedder)(nil)
