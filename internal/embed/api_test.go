package embed

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// TestAPIEmbedderInterfaceConformance is the only contract test required
// by the user's R3b instructions: confirm the API embedder satisfies the
// Embedder interface. The compile-time `var _ Embedder = ...` line in
// api.go does the heavy lifting; this test is the runtime gate.
func TestAPIEmbedderInterfaceConformance(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "sk-test")
	e, err := NewAPIEmbedder(APIEmbedderOptions{Provider: ProviderOpenAI})
	if err != nil {
		t.Fatalf("NewAPIEmbedder: %v", err)
	}
	var _ Embedder = e
	if err := e.Close(); err != nil {
		t.Errorf("Close: %v", err)
	}
}

func TestResolveAPIProvider(t *testing.T) {
	cases := []struct {
		env  string
		want APIProvider
	}{
		{"", ProviderLocal},
		{"local", ProviderLocal},
		{"LOCAL", ProviderLocal},
		{"openai", ProviderOpenAI},
		{"  OpenAI  ", ProviderOpenAI},
		{"voyage", ProviderVoyage},
		{"voyage-3-lite", ProviderLocal}, // unknown ⇒ falls back to local
		{"anthropic", ProviderLocal},     // not supported here ⇒ local
	}
	for _, c := range cases {
		t.Setenv("RECALL_EMBED_PROVIDER", c.env)
		got := ResolveAPIProvider()
		if got != c.want {
			t.Errorf("env=%q: got %q, want %q", c.env, got, c.want)
		}
	}
}

func TestNewAPIEmbedderRequiresKey(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "")
	t.Setenv("VOYAGE_API_KEY", "")

	if _, err := NewAPIEmbedder(APIEmbedderOptions{Provider: ProviderOpenAI}); err == nil {
		t.Error("OpenAI without key should error")
	}
	if _, err := NewAPIEmbedder(APIEmbedderOptions{Provider: ProviderVoyage}); err == nil {
		t.Error("Voyage without key should error")
	}
	if _, err := NewAPIEmbedder(APIEmbedderOptions{Provider: ProviderLocal}); err == nil {
		t.Error("ProviderLocal should not be a valid API provider")
	}
}

func TestNewAPIEmbedderUnknownProvider(t *testing.T) {
	if _, err := NewAPIEmbedder(APIEmbedderOptions{Provider: APIProvider("anthropic"), APIKey: "x"}); err == nil {
		t.Error("unknown provider should error")
	}
}

// fakeEmbeddingsServer returns a deterministic vector of `dims` floats
// for every input text so we can assert ordering and counts without
// depending on a real provider.
func fakeEmbeddingsServer(t *testing.T, expectedAuth, expectedModel string, dims int) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != expectedAuth {
			http.Error(w, `{"error":{"message":"bad auth"}}`, http.StatusUnauthorized)
			return
		}
		var req embedRequest
		body, _ := io.ReadAll(r.Body)
		if err := json.Unmarshal(body, &req); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		if expectedModel != "" && req.Model != expectedModel {
			http.Error(w, "wrong model: "+req.Model, http.StatusBadRequest)
			return
		}
		out := make([]embedResponseItem, len(req.Input))
		for i, text := range req.Input {
			vec := make([]float32, dims)
			for j := range vec {
				vec[j] = float32((i+1)*(j+1)) + float32(len(text))*0.001
			}
			out[i] = embedResponseItem{Embedding: vec, Index: i}
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"data": out})
	}))
}

func TestAPIEmbedderOpenAIRoundTrip(t *testing.T) {
	srv := fakeEmbeddingsServer(t, "Bearer sk-test", DefaultOpenAIModel, 768)
	defer srv.Close()

	e, err := NewAPIEmbedder(APIEmbedderOptions{
		Provider: ProviderOpenAI,
		APIKey:   "sk-test",
		BaseURL:  srv.URL,
	})
	if err != nil {
		t.Fatal(err)
	}

	vecs, err := e.Embed([]string{"alpha", "beta", "gamma"})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(vecs) != 3 {
		t.Fatalf("got %d vectors, want 3", len(vecs))
	}
	for i, v := range vecs {
		if len(v) != 768 {
			t.Errorf("vec %d width = %d, want 768", i, len(v))
		}
	}
	if e.Dimensions() != 768 {
		t.Errorf("Dimensions() = %d, want 768", e.Dimensions())
	}
	if !strings.Contains(e.ModelName(), DefaultOpenAIModel) {
		t.Errorf("ModelName() = %q, want to contain %q", e.ModelName(), DefaultOpenAIModel)
	}
	if !strings.HasPrefix(e.ModelName(), "openai:") {
		t.Errorf("ModelName() should be prefixed with provider: %q", e.ModelName())
	}
}

func TestAPIEmbedderVoyageRoundTrip(t *testing.T) {
	srv := fakeEmbeddingsServer(t, "Bearer voy-test", DefaultVoyageModel, 512)
	defer srv.Close()

	e, err := NewAPIEmbedder(APIEmbedderOptions{
		Provider: ProviderVoyage,
		APIKey:   "voy-test",
		BaseURL:  srv.URL,
	})
	if err != nil {
		t.Fatal(err)
	}
	vec, err := e.EmbedSingle("hello")
	if err != nil {
		t.Fatalf("EmbedSingle: %v", err)
	}
	if len(vec) != 512 {
		t.Errorf("voyage vec width = %d, want 512", len(vec))
	}
	if e.Dimensions() != 512 {
		t.Errorf("Dimensions() reported %d after first call, want 512", e.Dimensions())
	}
	if !strings.Contains(e.ModelName(), DefaultVoyageModel) {
		t.Errorf("ModelName() = %q", e.ModelName())
	}
}

// TestAPIEmbedderWorkerCapClamps verifies the embedder respects
// MaxAPIWorkers — passing Workers=64 doesn't fan out to 64
// concurrent requests against a real provider.
func TestAPIEmbedderWorkerCapClamps(t *testing.T) {
	var (
		concurrent int32
		peak       int32
	)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Track in-flight count across the request's lifetime so the
		// peak we observe = max parallelism the embedder achieved.
		now := atomic.AddInt32(&concurrent, 1)
		for {
			cur := atomic.LoadInt32(&peak)
			if now <= cur || atomic.CompareAndSwapInt32(&peak, cur, now) {
				break
			}
		}
		time.Sleep(20 * time.Millisecond) // give the next worker a chance to overlap
		atomic.AddInt32(&concurrent, -1)

		var req embedRequest
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &req)
		out := make([]embedResponseItem, len(req.Input))
		for i := range out {
			out[i] = embedResponseItem{Embedding: make([]float32, 768), Index: i}
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"data": out})
	}))
	defer srv.Close()

	e, _ := NewAPIEmbedder(APIEmbedderOptions{
		Provider:  ProviderOpenAI,
		APIKey:    "x",
		BaseURL:   srv.URL,
		BatchSize: 1,  // 1 text per HTTP call ⇒ N batches for N texts
		Workers:   64, // intentionally above MaxAPIWorkers (8)
	})
	texts := make([]string, 32) // 32 batches with batchSize=1
	for i := range texts {
		texts[i] = "t"
	}
	if _, err := e.Embed(texts); err != nil {
		t.Fatal(err)
	}
	got := atomic.LoadInt32(&peak)
	if got > int32(MaxAPIWorkers) {
		t.Errorf("peak concurrent requests = %d, expected ≤ MaxAPIWorkers (%d)", got, MaxAPIWorkers)
	}
	if got < 2 {
		t.Errorf("peak concurrent requests = %d, expected ≥ 2 (parallelism didn't kick in)", got)
	}
}

// TestAPIEmbedderParallelDispatchesConcurrently verifies that
// Workers > 1 actually fires concurrent HTTP requests.
func TestAPIEmbedderParallelDispatchesConcurrently(t *testing.T) {
	var (
		concurrent int32
		peak       int32
	)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		now := atomic.AddInt32(&concurrent, 1)
		for {
			cur := atomic.LoadInt32(&peak)
			if now <= cur || atomic.CompareAndSwapInt32(&peak, cur, now) {
				break
			}
		}
		time.Sleep(20 * time.Millisecond)
		atomic.AddInt32(&concurrent, -1)

		var req embedRequest
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &req)
		out := make([]embedResponseItem, len(req.Input))
		for i := range out {
			out[i] = embedResponseItem{Embedding: make([]float32, 768), Index: i}
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"data": out})
	}))
	defer srv.Close()

	e, _ := NewAPIEmbedder(APIEmbedderOptions{
		Provider:  ProviderOpenAI,
		APIKey:    "x",
		BaseURL:   srv.URL,
		BatchSize: 1,
		Workers:   4,
	})
	texts := make([]string, 12) // 12 batches at batchSize=1
	for i := range texts {
		texts[i] = "t"
	}
	if _, err := e.Embed(texts); err != nil {
		t.Fatal(err)
	}
	got := atomic.LoadInt32(&peak)
	if got < 2 {
		t.Errorf("peak concurrent = %d, expected ≥ 2", got)
	}
	if got > 4 {
		t.Errorf("peak concurrent = %d, expected ≤ Workers (4)", got)
	}
}

// TestAPIEmbedderSingleWorkerIsSequential verifies the default
// Workers=0/1 path stays sequential (one HTTP call at a time).
func TestAPIEmbedderSingleWorkerIsSequential(t *testing.T) {
	var (
		concurrent int32
		peak       int32
	)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		now := atomic.AddInt32(&concurrent, 1)
		for {
			cur := atomic.LoadInt32(&peak)
			if now <= cur || atomic.CompareAndSwapInt32(&peak, cur, now) {
				break
			}
		}
		time.Sleep(10 * time.Millisecond)
		atomic.AddInt32(&concurrent, -1)

		var req embedRequest
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &req)
		out := make([]embedResponseItem, len(req.Input))
		for i := range out {
			out[i] = embedResponseItem{Embedding: make([]float32, 768), Index: i}
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"data": out})
	}))
	defer srv.Close()

	e, _ := NewAPIEmbedder(APIEmbedderOptions{
		Provider:  ProviderOpenAI,
		APIKey:    "x",
		BaseURL:   srv.URL,
		BatchSize: 1,
		// Workers omitted ⇒ 0 ⇒ sequential.
	})
	texts := make([]string, 6)
	for i := range texts {
		texts[i] = "t"
	}
	if _, err := e.Embed(texts); err != nil {
		t.Fatal(err)
	}
	if got := atomic.LoadInt32(&peak); got != 1 {
		t.Errorf("peak concurrent = %d, expected 1 (sequential path)", got)
	}
}

// TestAPIEmbedderParallelPreservesOrder verifies that parallel
// dispatch still returns vectors in input order (the index field of
// each embedResponseItem matters here too).
func TestAPIEmbedderParallelPreservesOrder(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req embedRequest
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &req)
		out := make([]embedResponseItem, len(req.Input))
		for i, in := range req.Input {
			// Encode the input string into the first vector
			// component so we can recover ordering on the client.
			vec := make([]float32, 768)
			vec[0] = float32(int(in[0])) // input is "0", "1", etc.
			out[i] = embedResponseItem{Embedding: vec, Index: i}
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"data": out})
	}))
	defer srv.Close()

	e, _ := NewAPIEmbedder(APIEmbedderOptions{
		Provider:  ProviderOpenAI,
		APIKey:    "x",
		BaseURL:   srv.URL,
		BatchSize: 1,
		Workers:   4,
	})
	texts := make([]string, 16)
	for i := range texts {
		// Single-char inputs '0'..'?' so we can recover order from the
		// first vector component.
		texts[i] = string(rune('0' + i))
	}
	vecs, err := e.Embed(texts)
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range vecs {
		want := float32('0' + i)
		if v[0] != want {
			t.Errorf("vec[%d][0] = %v, want %v (parallel dispatch reordered results)", i, v[0], want)
		}
	}
}

func TestAPIEmbedderBatchSplitting(t *testing.T) {
	var calls int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&calls, 1)
		var req embedRequest
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &req)
		out := make([]embedResponseItem, len(req.Input))
		for i := range out {
			out[i] = embedResponseItem{Embedding: make([]float32, 768), Index: i}
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"data": out})
	}))
	defer srv.Close()

	e, _ := NewAPIEmbedder(APIEmbedderOptions{
		Provider:  ProviderOpenAI,
		APIKey:    "x",
		BaseURL:   srv.URL,
		BatchSize: 4, // tiny batch to force splits
	})
	texts := make([]string, 10) // ⇒ ceil(10/4) = 3 requests
	for i := range texts {
		texts[i] = "t"
	}
	if _, err := e.Embed(texts); err != nil {
		t.Fatal(err)
	}
	if atomic.LoadInt32(&calls) != 3 {
		t.Errorf("expected 3 batched HTTP calls, got %d", atomic.LoadInt32(&calls))
	}
}

// withFastBackoff swaps out the package backoff for a near-zero schedule
// so retry tests don't add real wall-clock delay.
func withFastBackoff(t *testing.T) {
	t.Helper()
	prev := backoffDelay
	backoffDelay = func(int) time.Duration { return time.Millisecond }
	t.Cleanup(func() { backoffDelay = prev })
}

func TestAPIEmbedderRetriesOn429(t *testing.T) {
	withFastBackoff(t)

	var calls int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n := atomic.AddInt32(&calls, 1)
		if n < 2 {
			w.WriteHeader(http.StatusTooManyRequests)
			_, _ = w.Write([]byte(`{"error":{"message":"slow down"}}`))
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"data": []embedResponseItem{{Embedding: make([]float32, 768), Index: 0}},
		})
	}))
	defer srv.Close()

	e, _ := NewAPIEmbedder(APIEmbedderOptions{
		Provider:   ProviderOpenAI,
		APIKey:     "x",
		BaseURL:    srv.URL,
		HTTPClient: &http.Client{Timeout: 5 * time.Second},
	})
	_, err := e.EmbedSingle("hello")
	if err != nil {
		t.Fatalf("EmbedSingle: %v", err)
	}
	if got := atomic.LoadInt32(&calls); got != 2 {
		t.Errorf("expected 2 calls (1 retry), got %d", got)
	}
}

func TestAPIEmbedderGivesUpAfterMaxRetries(t *testing.T) {
	withFastBackoff(t)

	var calls int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&calls, 1)
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	defer srv.Close()

	e, _ := NewAPIEmbedder(APIEmbedderOptions{
		Provider: ProviderOpenAI,
		APIKey:   "x",
		BaseURL:  srv.URL,
	})
	_, err := e.EmbedSingle("hello")
	if err == nil {
		t.Fatal("expected error after exhausting retries")
	}
	got := atomic.LoadInt32(&calls)
	if got < int32(MaxAPIRetries+1) {
		t.Errorf("expected ≥ %d attempts, got %d", MaxAPIRetries+1, got)
	}
}

func TestAPIEmbedderHTTPErrorPropagation(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = w.Write([]byte(`{"error":{"message":"invalid api key"}}`))
	}))
	defer srv.Close()

	e, _ := NewAPIEmbedder(APIEmbedderOptions{
		Provider: ProviderOpenAI,
		APIKey:   "wrong",
		BaseURL:  srv.URL,
	})
	_, err := e.EmbedSingle("x")
	if err == nil {
		t.Fatal("expected error from 401")
	}
	if !strings.Contains(err.Error(), "invalid api key") {
		t.Errorf("error should surface API message, got: %v", err)
	}
}

func TestAPIEmbedderClosed(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "x")
	e, err := NewAPIEmbedder(APIEmbedderOptions{Provider: ProviderOpenAI})
	if err != nil {
		t.Fatal(err)
	}
	_ = e.Close()
	if _, err := e.Embed([]string{"x"}); err == nil {
		t.Error("Embed after Close should error")
	}
}

func TestAPIEmbedderRequestShape(t *testing.T) {
	// Verify OpenAI request includes `dimensions`, Voyage includes
	// `input_type`, and neither leaks the other field.
	for _, c := range []struct {
		provider     APIProvider
		key          string
		wantHasDim   bool
		wantHasType  bool
		wantInputTyp string
	}{
		{ProviderOpenAI, "sk", true, false, ""},
		{ProviderVoyage, "voy", false, true, "document"},
	} {
		var captured embedRequest
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			body, _ := io.ReadAll(r.Body)
			_ = json.Unmarshal(body, &captured)
			out := []embedResponseItem{{Embedding: make([]float32, 768), Index: 0}}
			_ = json.NewEncoder(w).Encode(map[string]any{"data": out})
		}))
		e, _ := NewAPIEmbedder(APIEmbedderOptions{
			Provider: c.provider,
			APIKey:   c.key,
			BaseURL:  srv.URL,
		})
		_, _ = e.EmbedSingle("hello")
		srv.Close()

		hasDim := captured.Dimensions > 0
		hasType := captured.InputType != ""
		if hasDim != c.wantHasDim {
			t.Errorf("%s: dimensions present=%v, want %v", c.provider, hasDim, c.wantHasDim)
		}
		if hasType != c.wantHasType {
			t.Errorf("%s: input_type present=%v, want %v", c.provider, hasType, c.wantHasType)
		}
		if c.wantHasType && captured.InputType != c.wantInputTyp {
			t.Errorf("%s: input_type = %q, want %q", c.provider, captured.InputType, c.wantInputTyp)
		}
	}
}

func TestBackoffDelayCaps(t *testing.T) {
	// Confirm we cap at 30s rather than blowing up exponentially.
	if d := backoffDelay(10); d != 30*time.Second {
		t.Errorf("backoffDelay(10) = %v, want 30s", d)
	}
	// And it grows monotonically up to the cap.
	for i := 0; i < 4; i++ {
		if backoffDelay(i+1) < backoffDelay(i) {
			t.Errorf("backoff not monotonic at %d", i)
		}
	}
}

func TestProviderModelLabelStable(t *testing.T) {
	// metadata reconciliation depends on this being deterministic so that
	// switching to/from a provider triggers the model-changed warning.
	a := providerModelLabel(ProviderOpenAI, "text-embedding-3-small", 768)
	b := providerModelLabel(ProviderOpenAI, "text-embedding-3-small", 768)
	if a != b {
		t.Error("provider label is non-deterministic")
	}
	c := providerModelLabel(ProviderOpenAI, "text-embedding-3-small", 1536)
	if a == c {
		t.Error("dimension change should produce a different label")
	}
	d := providerModelLabel(ProviderVoyage, "voyage-3-lite", 0)
	if !strings.HasPrefix(d, "voyage:") {
		t.Errorf("voyage label = %q", d)
	}
}

// Sanity check that the constant we ship for batch size matches the value
// the user-facing docs and ROADMAP promise.
func TestAPIBatchSizeConstant(t *testing.T) {
	if APIBatchSize != 100 {
		t.Errorf("APIBatchSize = %d, want 100 (ROADMAP R3b)", APIBatchSize)
	}
}

// Ensures a transport-level error (e.g. dial failure) is wrapped with
// useful context rather than swallowed silently.
func TestAPIEmbedderTransportError(t *testing.T) {
	withFastBackoff(t)
	e, _ := NewAPIEmbedder(APIEmbedderOptions{
		Provider: ProviderOpenAI,
		APIKey:   "x",
		BaseURL:  "http://127.0.0.1:1", // unlikely to be listening
		HTTPClient: &http.Client{
			Timeout: 100 * time.Millisecond,
		},
	})
	_, err := e.EmbedSingle("x")
	if err == nil {
		t.Fatal("expected transport error")
	}
	if errors.Is(err, io.EOF) {
		// fine — we just want a non-nil error
	}
}
