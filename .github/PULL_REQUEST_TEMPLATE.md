## Summary

One or two sentences describing what this PR changes and why.

## Changes

- Bullet list of user-visible changes. Lead with behavior, not files.

## Related

Closes #... / Related to #...

## Test plan

- [ ] `go build ./...` passes
- [ ] `go test -race ./...` passes
- [ ] New code has accompanying tests
- [ ] Manually verified on: (e.g. `recall search "query"`, `recall query "query"`)

## Checklist

- [ ] Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/)
- [ ] `gofmt` / `go vet` clean
- [ ] Docs (README, CHANGELOG) updated if behavior changed
- [ ] No new HTTP calls in the default path
- [ ] No `init()` functions added (outside of sqlite-vec registration)
