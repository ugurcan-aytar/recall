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

.PHONY: build test test-race vet fmt install clean

build:
	go build -tags $(TAGS) -ldflags "$(LDFLAGS)" -o recall ./cmd/recall

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
