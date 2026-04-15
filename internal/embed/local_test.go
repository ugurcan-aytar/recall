package embed

import (
	"context"
	"encoding/json"
	"errors"
	"sync/atomic"
	"testing"
)

// fakeServer captures /v1/embeddings requests and returns canned
// vectors so we can unit-test the localEmbedder wire format
// without spinning up a real llama-server subprocess.
type fakeServer struct {
	dims        int
	calls       atomic.Int32
	failNext    bool
	dropOne     bool
	wrongDim    bool
	flipIndices bool
}

func (f *fakeServer) PostJSON(_ context.Context, path string, body, out any) error {
	f.calls.Add(1)
	if path != "/v1/embeddings" {
		return errors.New("fakeServer: unexpected path " + path)
	}
	if f.failNext {
		f.failNext = false
		return errors.New("fakeServer: synthetic failure")
	}
	bs, _ := json.Marshal(body)
	var req embeddingsRequest
	if err := json.Unmarshal(bs, &req); err != nil {
		return err
	}
	resp := embeddingsResponse{}
	for i, t := range req.Input {
		dims := f.dims
		if f.wrongDim {
			dims = f.dims + 1
		}
		v := make([]float32, dims)
		// Make vectors text-dependent so identity bugs surface.
		for j := range v {
			v[j] = float32(j+1) + float32(len(t))
		}
		idx := i
		if f.flipIndices {
			idx = len(req.Input) - 1 - i
		}
		resp.Data = append(resp.Data, struct {
			Embedding []float32 `json:"embedding"`
			Index     int       `json:"index"`
		}{Embedding: v, Index: idx})
	}
	if f.dropOne && len(resp.Data) > 0 {
		resp.Data = resp.Data[:len(resp.Data)-1]
	}
	rs, _ := json.Marshal(resp)
	return json.Unmarshal(rs, out)
}

func (f *fakeServer) Close() error { return nil }

func newFakeEmbedder(t *testing.T, workers int) (*localEmbedder, *fakeServer) {
	t.Helper()
	fs := &fakeServer{dims: 8}
	e := &localEmbedder{
		server:    fs,
		modelName: "fake",
		family:    FamilyNomic,
		dims:      8,
		workers:   workers,
		sem:       make(chan struct{}, max1(workers)),
	}
	return e, fs
}

func max1(n int) int {
	if n < 1 {
		return 1
	}
	return n
}

func TestLocalEmbedderEmbedSingle(t *testing.T) {
	e, fs := newFakeEmbedder(t, 1)
	v, err := e.EmbedSingle("hello")
	if err != nil {
		t.Fatalf("EmbedSingle: %v", err)
	}
	if len(v) != 8 {
		t.Fatalf("dims = %d, want 8", len(v))
	}
	if got := fs.calls.Load(); got != 1 {
		t.Fatalf("server calls = %d, want 1", got)
	}
}

func TestLocalEmbedderEmbedSequential(t *testing.T) {
	e, fs := newFakeEmbedder(t, 1)
	out, err := e.Embed([]string{"a", "bb", "ccc"})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(out) != 3 {
		t.Fatalf("len(out)=%d", len(out))
	}
	for i, v := range out {
		if len(v) != 8 {
			t.Fatalf("vec %d dims=%d", i, len(v))
		}
	}
	// Distinct text lengths must produce distinct vectors (catches
	// cross-input slot reuse bugs).
	if out[0][0] == out[1][0] || out[1][0] == out[2][0] {
		t.Fatalf("vectors are not text-dependent: %v %v %v", out[0][:2], out[1][:2], out[2][:2])
	}
	if got := fs.calls.Load(); got != 3 {
		t.Fatalf("server calls = %d, want 3 (one per text in sequential mode)", got)
	}
}

func TestLocalEmbedderEmbedParallel(t *testing.T) {
	e, fs := newFakeEmbedder(t, 4)
	texts := []string{"a", "bb", "ccc", "dddd", "eeeee", "ffffff", "ggggggg", "hhhhhhhh"}
	out, err := e.Embed(texts)
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(out) != len(texts) {
		t.Fatalf("len(out)=%d, want %d", len(out), len(texts))
	}
	for i, v := range out {
		if len(v) != 8 {
			t.Fatalf("vec %d dims=%d", i, len(v))
		}
		// Vectors are text-length encoded; verify ordering preserved.
		want := float32(1) + float32(len(texts[i]))
		if v[0] != want {
			t.Fatalf("vec %d[0]=%v, want %v — ordering broken", i, v[0], want)
		}
	}
	if got := fs.calls.Load(); got != int32(len(texts)) {
		t.Fatalf("server calls = %d, want %d", got, len(texts))
	}
}

func TestLocalEmbedderHonoursServerIndex(t *testing.T) {
	e, fs := newFakeEmbedder(t, 1)
	fs.flipIndices = true // server returns vectors with reversed Index
	out, err := e.Embed([]string{"alpha", "bravo"})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	// With flipIndices, vec for "alpha" carries Index=1 — we must
	// place it at out[1], not out[0].
	wantIdx0 := float32(1) + float32(len("bravo"))
	wantIdx1 := float32(1) + float32(len("alpha"))
	if out[0][0] != wantIdx0 || out[1][0] != wantIdx1 {
		t.Fatalf("Index field ignored; got out[0][0]=%v out[1][0]=%v want %v %v",
			out[0][0], out[1][0], wantIdx0, wantIdx1)
	}
}

func TestLocalEmbedderRejectsDimMismatch(t *testing.T) {
	e, fs := newFakeEmbedder(t, 1)
	fs.wrongDim = true
	_, err := e.EmbedSingle("hi")
	if err == nil {
		t.Fatal("expected error on dim mismatch, got nil")
	}
}

func TestLocalEmbedderRejectsShortResponse(t *testing.T) {
	e, fs := newFakeEmbedder(t, 1)
	fs.dropOne = true
	_, err := e.Embed([]string{"a", "b"})
	if err == nil {
		t.Fatal("expected error when server returns fewer vectors than requested")
	}
}

func TestLocalEmbedderEmbedAfterClose(t *testing.T) {
	e, _ := newFakeEmbedder(t, 1)
	if err := e.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if _, err := e.EmbedSingle("x"); err == nil {
		t.Fatal("expected error after Close, got nil")
	}
}

func TestLocalEmbedderEmptyInputReturnsNil(t *testing.T) {
	e, fs := newFakeEmbedder(t, 1)
	out, err := e.Embed(nil)
	if err != nil {
		t.Fatalf("Embed(nil): %v", err)
	}
	if out != nil {
		t.Fatalf("expected nil for empty input, got %v", out)
	}
	if fs.calls.Load() != 0 {
		t.Fatal("Empty input should not hit the server")
	}
}

func TestLocalEmbedderMetadataAccessors(t *testing.T) {
	e, _ := newFakeEmbedder(t, 1)
	if e.Dimensions() != 8 {
		t.Errorf("Dimensions=%d", e.Dimensions())
	}
	if e.ModelName() != "fake" {
		t.Errorf("ModelName=%q", e.ModelName())
	}
	if e.Family() != FamilyNomic {
		t.Errorf("Family=%q", e.Family())
	}
}
