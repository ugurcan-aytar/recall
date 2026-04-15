package llm

import (
	"errors"
	"sync"
)

// MockGenerator is a deterministic generator for tests. Either map
// specific prompts to canned responses (Responses) or return Default
// for everything else. Records the prompts it was called with so
// tests can assert on the wire format.
type MockGenerator struct {
	// Responses maps prompt strings to responses. When the incoming
	// prompt isn't in the map, MockGenerator returns Default.
	Responses map[string]string

	// Default is returned when an incoming prompt is not in
	// Responses. "" is fine — empty string is a valid generator
	// output (the parser will produce an empty Expanded struct).
	Default string

	// Name surfaces through ModelName(). "" defaults to "mock-gen".
	Name string

	mu     sync.Mutex
	calls  []string
	closed bool
}

// NewMockGenerator returns a generator wired with the given canned
// responses. Pass nil if you only need the Default behaviour.
func NewMockGenerator(responses map[string]string) *MockGenerator {
	if responses == nil {
		responses = map[string]string{}
	}
	return &MockGenerator{Responses: responses, Name: "mock-gen"}
}

// Generate looks the prompt up in Responses, falls back to Default,
// and records the call.
func (m *MockGenerator) Generate(prompt string, _ ...GenerateOption) (string, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.closed {
		return "", errors.New("mock generator is closed")
	}
	m.calls = append(m.calls, prompt)
	if r, ok := m.Responses[prompt]; ok {
		return r, nil
	}
	return m.Default, nil
}

// ModelName reports the stable mock identifier.
func (m *MockGenerator) ModelName() string {
	if m.Name == "" {
		return "mock-gen"
	}
	return m.Name
}

// Calls returns a copy of every prompt the mock has seen, in order.
// Useful for asserting on the exact wire format the caller sends.
func (m *MockGenerator) Calls() []string {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := make([]string, len(m.calls))
	copy(out, m.calls)
	return out
}

// Close releases (no-op) resources. Safe to call repeatedly.
func (m *MockGenerator) Close() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.closed = true
	return nil
}

// Static interface-conformance check.
var _ Generator = (*MockGenerator)(nil)
