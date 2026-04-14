package commands

import (
	"fmt"
	"runtime"
	"runtime/debug"
	"time"

	"github.com/spf13/cobra"
)

// These vars are populated at link time via -ldflags. The Makefile and
// goreleaser pass actual values; raw `go build` falls back to the
// runtime/debug.BuildInfo block below.
var (
	Version   = "dev"
	Commit    = ""
	BuildDate = ""
)

var versionCmd = &cobra.Command{
	Use:   "version",
	Short: "Print recall version, build date, commit, and Go version",
	Run: func(cmd *cobra.Command, args []string) {
		v, c, d := resolveVersion()
		fmt.Printf("recall %s\n", v)
		fmt.Printf("  commit:     %s\n", orDash(c))
		fmt.Printf("  built:      %s\n", orDash(d))
		fmt.Printf("  go:         %s\n", runtime.Version())
		fmt.Printf("  os/arch:    %s/%s\n", runtime.GOOS, runtime.GOARCH)
	},
}

// resolveVersion prefers ldflags-injected values; falls back to whatever
// the Go toolchain stamped via runtime/debug (works for `go install` and
// VCS-aware builds).
func resolveVersion() (version, commit, buildDate string) {
	version = Version
	commit = Commit
	buildDate = BuildDate

	info, ok := debug.ReadBuildInfo()
	if !ok {
		return
	}
	if version == "dev" && info.Main.Version != "" && info.Main.Version != "(devel)" {
		version = info.Main.Version
	}
	for _, s := range info.Settings {
		switch s.Key {
		case "vcs.revision":
			if commit == "" {
				commit = s.Value
				if len(commit) > 12 {
					commit = commit[:12]
				}
			}
		case "vcs.time":
			if buildDate == "" {
				if t, err := time.Parse(time.RFC3339, s.Value); err == nil {
					buildDate = t.Format("2006-01-02")
				}
			}
		}
	}
	return
}

func orDash(s string) string {
	if s == "" {
		return "-"
	}
	return s
}
