package llm

import "testing"

func TestMockGeneratorDefault(t *testing.T) {
	m := NewMockGenerator(nil)
	m.Default = "fallback response"
	got, err := m.Generate("anything", WithMaxTokens(64))
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if got != "fallback response" {
		t.Errorf("Generate = %q, want %q", got, "fallback response")
	}
	calls := m.Calls()
	if len(calls) != 1 || calls[0] != "anything" {
		t.Errorf("Calls = %+v", calls)
	}
}

func TestMockGeneratorMappedResponse(t *testing.T) {
	m := NewMockGenerator(map[string]string{
		"hello": "world",
	})
	m.Default = "default"

	got, _ := m.Generate("hello")
	if got != "world" {
		t.Errorf("got %q, want world", got)
	}
	got, _ = m.Generate("unknown")
	if got != "default" {
		t.Errorf("got %q, want default", got)
	}
}

func TestMockGeneratorClosed(t *testing.T) {
	m := NewMockGenerator(nil)
	if err := m.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := m.Generate("x"); err == nil {
		t.Error("expected error after Close")
	}
}

func TestMockGeneratorDefaultModelName(t *testing.T) {
	m := &MockGenerator{}
	if got := m.ModelName(); got != "mock-gen" {
		t.Errorf("default ModelName = %q, want mock-gen", got)
	}
	m.Name = "custom"
	if got := m.ModelName(); got != "custom" {
		t.Errorf("custom ModelName = %q", got)
	}
}

func TestWithMaxTokensRejectsNonPositive(t *testing.T) {
	o := GenerateOptions{MaxTokens: 100}
	WithMaxTokens(0)(&o)
	if o.MaxTokens != 100 {
		t.Errorf("WithMaxTokens(0) should be a no-op; MaxTokens = %d", o.MaxTokens)
	}
	WithMaxTokens(-50)(&o)
	if o.MaxTokens != 100 {
		t.Errorf("WithMaxTokens(-50) should be a no-op; MaxTokens = %d", o.MaxTokens)
	}
	WithMaxTokens(200)(&o)
	if o.MaxTokens != 200 {
		t.Errorf("WithMaxTokens(200) = %d, want 200", o.MaxTokens)
	}
}

func TestNewLocalGeneratorStubReturnsNotCompiled(t *testing.T) {
	if LocalGeneratorAvailable() {
		t.Skip("binary built with embed_llama — stub path not applicable")
	}
	_, err := NewLocalGenerator(LocalGeneratorOptions{ModelPath: "/tmp/nope.gguf"})
	if err == nil {
		t.Fatal("expected error on stub build")
	}
}
