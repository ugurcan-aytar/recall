package commands

import (
	"strings"
	"testing"
)

func TestVersionDefault(t *testing.T) {
	if Version == "" {
		t.Fatal("Version must have a non-empty default for `recall version`")
	}
}

func TestVersionCmdDefined(t *testing.T) {
	if versionCmd.Use != "version" {
		t.Errorf("Use = %q", versionCmd.Use)
	}
}

func TestResolveVersionFallsBackToBuildInfo(t *testing.T) {
	// When ldflags haven't injected, resolveVersion may pull commit/date
	// from runtime/debug in VCS-aware builds. We can't assert specific
	// values, but the function must return without panicking and must
	// preserve the linker-injected Version unless it's still "dev".
	v, _, _ := resolveVersion()
	if v == "" {
		t.Error("resolved version is empty")
	}
}

func TestOrDash(t *testing.T) {
	if orDash("") != "-" {
		t.Error("empty should become -")
	}
	if orDash("abc") != "abc" {
		t.Error("non-empty should pass through")
	}
}

func TestVersionLdflagsHonoured(t *testing.T) {
	// Simulate ldflags injection by mutating the package-level vars.
	prevV, prevC, prevD := Version, Commit, BuildDate
	t.Cleanup(func() { Version, Commit, BuildDate = prevV, prevC, prevD })

	Version = "v9.9.9"
	Commit = "abcdef0"
	BuildDate = "2026-04-14"

	v, c, d := resolveVersion()
	if v != "v9.9.9" {
		t.Errorf("Version = %q, want v9.9.9", v)
	}
	if c != "abcdef0" {
		t.Errorf("Commit = %q", c)
	}
	if d != "2026-04-14" {
		t.Errorf("BuildDate = %q", d)
	}
	if !strings.Contains(versionCmd.Short, "version") {
		t.Errorf("Short = %q, want to mention version", versionCmd.Short)
	}
}
