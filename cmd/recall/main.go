package main

import (
	"fmt"
	"os"

	"github.com/ugurcan-aytar/recall/internal/commands"
)

func main() {
	if err := commands.Execute(); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}
