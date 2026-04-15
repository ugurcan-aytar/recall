package llm

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"sync/atomic"
	"testing"
)

// fakeGenServer fakes the llama-server HTTP surface for tests.
// Returns a canned response (or an error when failNext is set).
type fakeGenServer struct {
	response string
	lastReq  chatRequest
	calls    atomic.Int32
	failNext bool
	closeCnt atomic.Int32
}

func (f *fakeGenServer) PostJSON(_ context.Context, path string, body, out any) error {
	f.calls.Add(1)
	if path != "/v1/chat/completions" {
		return errors.New("fakeGenServer: unexpected path " + path)
	}
	if f.failNext {
		f.failNext = false
		return errors.New("fakeGenServer: synthetic failure")
	}
	bs, _ := json.Marshal(body)
	_ = json.Unmarshal(bs, &f.lastReq)

	resp := chatResponse{
		Choices: []chatResponseChoice{
			{Message: chatMessage{Role: "assistant", Content: f.response}},
		},
	}
	rs, _ := json.Marshal(resp)
	return json.Unmarshal(rs, out)
}

func (f *fakeGenServer) Close() error {
	f.closeCnt.Add(1)
	return nil
}

func newFakeGen(t *testing.T, response string) (*localGenerator, *fakeGenServer) {
	t.Helper()
	fs := &fakeGenServer{response: response}
	return &localGenerator{
		server:    fs,
		modelName: "fake-gen",
	}, fs
}

func TestLocalGeneratorGenerateDefaults(t *testing.T) {
	g, fs := newFakeGen(t, "yes")
	out, err := g.Generate("Is this passage about X?")
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if out != "yes" {
		t.Fatalf("Generate output = %q, want %q", out, "yes")
	}
	if fs.calls.Load() != 1 {
		t.Fatalf("server calls = %d", fs.calls.Load())
	}
	// Defaults applied server-side via our request struct.
	if fs.lastReq.MaxTokens != DefaultMaxTokens {
		t.Errorf("MaxTokens = %d, want %d", fs.lastReq.MaxTokens, DefaultMaxTokens)
	}
	if fs.lastReq.Temperature != 0 {
		t.Errorf("Temperature = %v, want 0 (greedy)", fs.lastReq.Temperature)
	}
	if fs.lastReq.TopK != 1 {
		t.Errorf("TopK = %d, want 1 (greedy)", fs.lastReq.TopK)
	}
	if len(fs.lastReq.Messages) != 1 || fs.lastReq.Messages[0].Role != "user" {
		t.Errorf("messages = %+v, want single user role", fs.lastReq.Messages)
	}
}

func TestLocalGeneratorRespectsMaxTokens(t *testing.T) {
	g, fs := newFakeGen(t, "anything")
	_, err := g.Generate("short prompt", WithMaxTokens(32))
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if fs.lastReq.MaxTokens != 32 {
		t.Fatalf("MaxTokens = %d, want 32", fs.lastReq.MaxTokens)
	}
}

func TestLocalGeneratorZeroMaxTokensFallsBack(t *testing.T) {
	g, fs := newFakeGen(t, "anything")
	_, err := g.Generate("short", WithMaxTokens(0))
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if fs.lastReq.MaxTokens != DefaultMaxTokens {
		t.Fatalf("WithMaxTokens(0) did not fall back to DefaultMaxTokens, got %d", fs.lastReq.MaxTokens)
	}
}

func TestLocalGeneratorSurfacesServerError(t *testing.T) {
	g, fs := newFakeGen(t, "")
	fs.failNext = true
	_, err := g.Generate("x")
	if err == nil {
		t.Fatal("expected error when server fails, got nil")
	}
	if !strings.Contains(err.Error(), "synthetic failure") {
		t.Errorf("error did not wrap server error: %v", err)
	}
}

func TestLocalGeneratorPassesPromptVerbatim(t *testing.T) {
	g, fs := newFakeGen(t, "out")
	in := "/no_think Expand this query: product market fit\nQuery intent: growth"
	_, err := g.Generate(in)
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if len(fs.lastReq.Messages) != 1 || fs.lastReq.Messages[0].Content != in {
		t.Fatalf("server received mangled prompt:\nwant: %q\n got: %+v", in, fs.lastReq.Messages)
	}
}

func TestLocalGeneratorModelName(t *testing.T) {
	g, _ := newFakeGen(t, "x")
	if g.ModelName() != "fake-gen" {
		t.Errorf("ModelName = %q", g.ModelName())
	}
}

func TestLocalGeneratorCloseIdempotent(t *testing.T) {
	g, fs := newFakeGen(t, "x")
	if err := g.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if err := g.Close(); err != nil {
		t.Fatalf("Close (2nd): %v", err)
	}
	if fs.closeCnt.Load() != 1 {
		t.Errorf("server Close called %d times, want exactly 1", fs.closeCnt.Load())
	}
	if _, err := g.Generate("x"); err == nil {
		t.Error("Generate after Close should error")
	}
}

func TestLocalGeneratorSerialisesConcurrentCalls(t *testing.T) {
	// Two goroutines calling Generate at the same time must not
	// race on the embedded fakeGenServer's lastReq field. The
	// mutex in localGenerator makes the calls serial; the race
	// detector catches any lapse.
	g, fs := newFakeGen(t, "ok")
	done := make(chan struct{})
	go func() {
		for i := 0; i < 50; i++ {
			_, _ = g.Generate("a")
		}
		close(done)
	}()
	for i := 0; i < 50; i++ {
		_, _ = g.Generate("b")
	}
	<-done
	if fs.calls.Load() != 100 {
		t.Fatalf("server calls = %d, want 100", fs.calls.Load())
	}
}
