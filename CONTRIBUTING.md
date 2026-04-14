# Contributing to recall

Thanks for your interest in improving recall. This document covers everything you need to get a working dev environment, write a change, and open a pull request.

## Development setup

Requirements:

- **Go 1.24+** (check with `go version`)
- **CGo toolchain** — a C compiler and `make`. On macOS, Xcode Command Line Tools cover this. On Debian/Ubuntu: `sudo apt install build-essential`.
- **Git**

Clone and build:

```bash
git clone https://github.com/ugurcan-aytar/recall.git
cd recall
make build   # or: go build -tags sqlite_fts5 -o recall ./cmd/recall
make test    # or: go test -tags sqlite_fts5 ./...
```

recall needs SQLite's FTS5 module, which `mattn/go-sqlite3` compiles in only when the `sqlite_fts5` build tag is set. The Makefile handles this; raw `go` commands need `-tags sqlite_fts5`.

From Phase R3 onward the embedding code depends on llama.cpp via `go-llama.cpp`. That dependency needs a static library built once before `go build`:

```bash
# CPU only — works everywhere
cd third_party/go-llama.cpp     # exact path lands in R3
make libbinding.a
cd ../..

# macOS with Metal GPU acceleration
BUILD_TYPE=metal make libbinding.a
export CGO_LDFLAGS="-framework Foundation -framework Metal -framework MetalKit -framework MetalPerformanceShaders"

# Linux with CUDA
BUILD_TYPE=cublas make libbinding.a
export CGO_LDFLAGS="-lcublas -lcudart -L/usr/local/cuda/lib64/"

CGO_ENABLED=1 go build -o recall ./cmd/recall
```

Until Phase R3 lands, recall builds without any C dependencies beyond `mattn/go-sqlite3` (Phase R1).

## Code style

- **`gofmt` on every file.** CI will reject unformatted code. Run `gofmt -w .` or enable format-on-save in your editor.
- **`go vet ./...`** must pass cleanly.
- **One file per CLI subcommand** inside `internal/commands/`.
- **One responsibility per package.** `store` does data access, `chunk` does chunking, `embed` does embedding. Orchestration lives in `pkg/recall` or `internal/commands`.
- **No clever abstractions.** Three similar lines beats a premature interface.
- **No `init()` functions** except the one registering sqlite-vec.
- **Error messages are actionable.** `"failed to open database at /x/y: permission denied"` beats `"db error"`.

## Testing

Every new `.go` file in `internal/` and `pkg/` must ship with a `_test.go` alongside it. Use table-driven tests and `t.TempDir()` for anything that touches the filesystem. Do not download real GGUF models in tests — `internal/embed/embed_test.go` provides a deterministic `MockEmbedder`.

```bash
make test                                           # all tests
make test-race                                      # race detector (CI does this)
go test -tags sqlite_fts5 -coverprofile=coverage.out ./...
go tool cover -html=coverage.out
```

Coverage target on `internal/store/`, `internal/chunk/`, `internal/embed/`, and `pkg/recall/` is 80%+.

## Commit messages

recall follows [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add adaptive min-score floor to fusion
fix: prevent FTS5 trigger from double-inserting on update
docs: document RECALL_EMBED_PROVIDER behavior
refactor: extract chunk break-point scoring into its own file
test: cover empty-collection case in Index()
chore: bump go-sqlite3 to latest patch
```

Separate unrelated changes into separate commits. Commit messages explain **why**; the diff already shows **what**.

## Pull request flow

1. Fork the repo and create a branch off `main` (`git checkout -b feat/my-change`).
2. Make your change. Add or update tests.
3. Run `go build ./... && go test -race ./...` locally — this has to pass before you open the PR.
4. Push and open a PR. Fill out the template. Link related issues.
5. CI runs on Ubuntu and macOS. It has to be green.
6. A maintainer reviews. Expect questions about trade-offs, naming, and scope.
7. Squash-merge when approved.

Small, focused PRs land faster than big ones. If you're unsure whether a change fits recall's scope, open an issue first — "should we add X?" questions are cheap to discuss and save rewriting.

## Scope boundaries

recall is a search engine. It indexes, chunks, embeds, searches, and retrieves. It does **not**:

- Call LLMs or build prompts — that belongs in [brain](https://github.com/ugurcan-aytar/brain).
- Modify, create, or delete files in user collections. recall is a reader.
- Make HTTP calls in the default path. `RECALL_EMBED_PROVIDER=openai` is an explicit opt-in, never a default.

Features that cross these lines will get pushed back in review. If you need behavior that belongs on the other side of the boundary, propose it in brain instead.

## License

By contributing you agree your contributions are licensed under the MIT License (see [LICENSE](LICENSE)).
