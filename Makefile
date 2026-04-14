# recall — convenience targets.
#
# mattn/go-sqlite3 gates FTS5 behind the `sqlite_fts5` build tag, so every
# go invocation in this project needs it. `make build` / `make test` handle
# that for you; otherwise use `go build -tags sqlite_fts5 ./...`.

TAGS := sqlite_fts5

# Version metadata stamped into `recall version`. CI / goreleaser overrides
# these via -ldflags too; the values here just give dev builds friendly
# output instead of "dev / - / -".
VERSION   ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo dev)
COMMIT    ?= $(shell git rev-parse --short HEAD 2>/dev/null || echo "")
BUILDDATE ?= $(shell date -u +%Y-%m-%d)

LDFLAGS := -X github.com/ugurcan-aytar/recall/internal/commands.Version=$(VERSION) \
           -X github.com/ugurcan-aytar/recall/internal/commands.Commit=$(COMMIT) \
           -X github.com/ugurcan-aytar/recall/internal/commands.BuildDate=$(BUILDDATE)

.PHONY: build test test-race vet fmt install clean demos

build:
	go build -tags $(TAGS) -ldflags "$(LDFLAGS)" -o recall ./cmd/recall

# Render the README demo GIFs into assets/ via charmbracelet/vhs.
# Requires `brew install vhs`. Each .tape script in demos/ writes its own
# assets/<name>.gif so re-rendering one doesn't touch the others.
demos:
	@command -v vhs >/dev/null || { echo "vhs not installed: brew install vhs"; exit 1; }
	go build -tags $(TAGS) -ldflags "$(LDFLAGS)" -o demos/recall ./cmd/recall
	@for tape in demos/*.tape; do echo ">> $$tape"; vhs "$$tape"; done
	@rm -f demos/recall

test:
	go test -tags $(TAGS) ./...

test-race:
	go test -tags $(TAGS) -race ./...

vet:
	go vet -tags $(TAGS) ./...

fmt:
	gofmt -w .

install:
	go install -tags $(TAGS) -ldflags "$(LDFLAGS)" ./cmd/recall

clean:
	rm -f recall coverage.out
