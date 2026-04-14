package commands

import "testing"

func TestDoctorCmdDefined(t *testing.T) {
	if doctorCmd.Use != "doctor" {
		t.Errorf("Use = %q", doctorCmd.Use)
	}
	if doctorCmd.Short == "" {
		t.Error("Short is empty")
	}
}
