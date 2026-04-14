# Q1 2026 Retro — 2026-04-15

## What went well

- Shipped the new authentication flow on schedule. Zero rollback events
  in the first week.
- The on-call rotation rework reduced after-hours pages by ~40%.
- Customer support flagged way fewer "I can't log in" tickets after the
  password reset email redesign.

## What didn't

- The rate limiter incident on March 22 (see `incident-2026-03-22.md`)
  was a wake-up call. We didn't have alerting for queue depth.
- Code review SLA slipped twice. Reviews are sitting open for 3+ days.
- We let the chunking algorithm regression land because nobody owned the
  retrieval evaluation harness.

## Action items

1. Add alerting for the rate-limiter queue depth metric. Owner: Cem.
2. Adopt a 24-hour code-review SLA. Owner: Deniz.
3. Set up a weekly retrieval-quality review meeting. Owner: Aslı.
4. Document the runbook for the payments service. Owner: Burak. (see
   `runbook-payments.md`).

## Decisions

- We are not going to chase real-time fraud detection this quarter.
  Batched scoring is good enough until we hit 100M events/day.
- Sticking with PostgreSQL for the primary store. The Spanner trial
  didn't justify the migration cost.
