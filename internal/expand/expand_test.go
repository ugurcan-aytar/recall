package expand_test

import (
	"errors"
	"testing"

	"github.com/ugurcan-aytar/recall/internal/expand"
	"github.com/ugurcan-aytar/recall/internal/llm"
)

func TestExpandSendsExpectedPrompt(t *testing.T) {
	gen := llm.NewMockGenerator(nil)
	gen.Default = "lex: rate limiter\nvec: how does rate limiting work\nhyde: A rate limiter caps requests."

	if _, err := expand.Expand(gen, "rate limiter", expand.Options{IncludeLex: true}); err != nil {
		t.Fatalf("Expand: %v", err)
	}
	calls := gen.Calls()
	if len(calls) != 1 {
		t.Fatalf("calls = %d, want 1", len(calls))
	}
	want := expand.PromptPrefix + "rate limiter"
	if calls[0] != want {
		t.Errorf("prompt = %q, want %q", calls[0], want)
	}
}

func TestExpandThreadsIntent(t *testing.T) {
	gen := llm.NewMockGenerator(nil)
	gen.Default = "vec: query"

	_, err := expand.Expand(gen, "performance", expand.Options{
		Intent:     "web page load times",
		IncludeLex: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	calls := gen.Calls()
	if len(calls) != 1 {
		t.Fatal("expected one call")
	}
	want := expand.PromptPrefix + "performance\nQuery intent: web page load times"
	if calls[0] != want {
		t.Errorf("prompt with intent = %q, want %q", calls[0], want)
	}
}

func TestExpandRejectsEmptyQuery(t *testing.T) {
	gen := llm.NewMockGenerator(nil)
	if _, err := expand.Expand(gen, "   ", expand.Options{}); err == nil {
		t.Error("expected error on empty query")
	}
}

func TestExpandRequiresGenerator(t *testing.T) {
	if _, err := expand.Expand(nil, "x", expand.Options{}); err == nil {
		t.Error("expected error on nil generator")
	}
}

func TestExpandPropagatesGeneratorError(t *testing.T) {
	gen := &erroringGen{err: errors.New("model died")}
	_, err := expand.Expand(gen, "x", expand.Options{})
	if err == nil {
		t.Fatal("expected propagated error")
	}
}

type erroringGen struct{ err error }

func (e *erroringGen) Generate(string, ...llm.GenerateOption) (string, error) {
	return "", e.err
}
func (e *erroringGen) ModelName() string { return "erroring" }
func (e *erroringGen) Close() error      { return nil }

func TestParseHappyPath(t *testing.T) {
	raw := "lex: rate limiter\nvec: how does rate limiting handle bursts\nhyde: A rate limiter using token bucket caps requests per second."
	got := expand.Parse(raw, "rate limiter", true)

	if got.Original != "rate limiter" {
		t.Errorf("Original = %q", got.Original)
	}
	if len(got.Lex) != 1 || got.Lex[0] != "rate limiter" {
		t.Errorf("Lex = %+v", got.Lex)
	}
	if len(got.Vec) != 1 || got.Vec[0] != "how does rate limiting handle bursts" {
		t.Errorf("Vec = %+v", got.Vec)
	}
	if len(got.Hyde) != 1 {
		t.Errorf("Hyde = %+v", got.Hyde)
	}
}

func TestParseDropsLexWhenIncludeLexFalse(t *testing.T) {
	raw := "lex: rate limiter\nvec: query"
	got := expand.Parse(raw, "rate limiter", false)
	if len(got.Lex) != 0 {
		t.Errorf("Lex should be empty when IncludeLex=false; got %+v", got.Lex)
	}
	if len(got.Vec) != 1 {
		t.Errorf("Vec should still be parsed; got %+v", got.Vec)
	}
}

func TestParseDropsLexWithNoOverlap(t *testing.T) {
	// "rate limiter" → tokens {rate, limiter}
	// "machine learning" → tokens {machine, learning}
	// No overlap → drop.
	raw := "lex: machine learning\nvec: ok"
	got := expand.Parse(raw, "rate limiter", true)
	if len(got.Lex) != 0 {
		t.Errorf("Lex should drop variant with zero token overlap; got %+v", got.Lex)
	}
}

func TestParseSkipsMalformedLines(t *testing.T) {
	raw := "garbage line\n: missing type\nlex:\nlex: ok\n   \nthink: irrelevant"
	got := expand.Parse(raw, "ok", true)
	if len(got.Lex) != 1 || got.Lex[0] != "ok" {
		t.Errorf("Lex = %+v, want [ok]", got.Lex)
	}
}

func TestParseEmpty(t *testing.T) {
	got := expand.Parse("", "rate limiter", true)
	if got.Original != "rate limiter" {
		t.Errorf("Original = %q", got.Original)
	}
	if len(got.Lex)+len(got.Vec)+len(got.Hyde) != 0 {
		t.Errorf("expected empty buckets, got %+v", got)
	}
}

func TestParseWhitespaceTolerant(t *testing.T) {
	raw := "  lex:    rate limiter   \n  VEC:  query  \n  Hyde: passage  "
	got := expand.Parse(raw, "rate", true)
	if len(got.Lex) != 1 {
		t.Errorf("Lex = %+v", got.Lex)
	}
	if len(got.Vec) != 1 || got.Vec[0] != "query" {
		t.Errorf("Vec = %+v", got.Vec)
	}
	if len(got.Hyde) != 1 || got.Hyde[0] != "passage" {
		t.Errorf("Hyde = %+v", got.Hyde)
	}
}
