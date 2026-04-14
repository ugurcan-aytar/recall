package commands

import "testing"

func TestCleanupCmdDefined(t *testing.T) {
	if cleanupCmd.Use != "cleanup" {
		t.Errorf("Use = %q", cleanupCmd.Use)
	}
	if cleanupCmd.Short == "" {
		t.Error("Short is empty")
	}
	if cleanupCmd.RunE == nil {
		t.Error("cleanup must have a RunE (no longer a stub)")
	}
}
