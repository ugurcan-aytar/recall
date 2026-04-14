# Security Policy

## Supported versions

recall is pre-1.0 and ships active development on `main`. Security fixes land on the latest tagged minor release. Older minors are not backported.

| Version | Supported |
|---|---|
| Latest `v0.x` tag | Yes |
| Older tags | No — please upgrade |

## Reporting a vulnerability

**Please do not open a public GitHub issue for security bugs.**

Instead, report privately via either:

- [GitHub security advisories](https://github.com/ugurcan-aytar/recall/security/advisories/new) (preferred — keeps the discussion on the repo)
- Email: `aytarugurcan@gmail.com` with the subject line `recall security report`

Please include:

- The version or commit you tested against
- A description of the issue and its impact
- A minimal reproduction (commands, sample files, or a patch)
- Any mitigation you've already considered

## Disclosure timeline

| Day | What happens |
|---|---|
| 0 | Report received. Acknowledgement within **72 hours**. |
| 1–7 | Triage: confirm the issue and assess severity. |
| 7–30 | Fix developed, reviewed, and tested. |
| 30–90 | Release a patched version. Coordinate public disclosure with the reporter. |
| 90+ | Public advisory published (credit given unless the reporter prefers anonymity). |

We aim to ship a fix within **30 days** for high-severity issues and **90 days** for everything else. If a fix will take longer, we'll say so explicitly.

## Scope

In scope:

- The recall CLI and public `pkg/recall` API.
- Database handling, file system access, and network code (embedding API fallbacks).
- Supply-chain issues involving recall's direct dependencies when exploited through recall.

Out of scope:

- Vulnerabilities in third-party dependencies that cannot be triggered by recall's code paths. (Report those upstream.)
- Denial-of-service from indexing pathologically large local files. recall is a local tool; resource limits are the operator's responsibility.
- Issues that require pre-existing local root / admin access on the user's machine.

Thanks for helping keep recall safe.
