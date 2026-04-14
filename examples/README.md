# examples/

A small fictional knowledge base used by the README quick-start, the
end-to-end validation in CLAUDE.md, and anyone trying recall for the
first time. Nothing here is real — names, services, dates, and numbers
are made up.

```bash
recall collection add ./examples --name examples \
  --context "Sample notes including meeting notes, technical docs, and journal entries"
recall index
recall search "rate limiter"
```

Files cover the kinds of writing recall is built for:

- `meeting-notes-2026-04-08.md` — sprint planning notes
- `meeting-notes-2026-04-15.md` — quarterly retro
- `technical-doc.md` — design doc covering auth, retries, circuit breakers
- `runbook-payments.md` — operations runbook for the payments service
- `journal-2026-04-12.md` — a personal engineering journal entry
- `incident-2026-03-22.md` — incident report
