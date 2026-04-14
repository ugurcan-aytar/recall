# Sprint Planning — 2026-04-08

Attendees: Aslı, Burak, Cem, Deniz, Esra.

## Carryover from last sprint

- The authentication middleware refactor is still in review. Burak will
  ping the reviewers today.
- Rate limiter telemetry dashboard is blocked on a Grafana migration.

## This sprint

We picked the following stories from the backlog:

1. Move JWT validation into the middleware layer (Burak)
2. Add per-tenant rate limiting to the public API (Cem)
3. Replace the ad-hoc retry helper with a real circuit breaker (Deniz)
4. Cut a release of the search service with the new ranking weights (Esra)
5. Triage the inbox of unresolved customer reports about login (Aslı)

## Action items

- Cem to write a one-pager on the rate limiter rollout plan by Thursday.
- Deniz to spike the circuit breaker library options (sony/gobreaker vs
  hashicorp/golang-lru) and bring back a recommendation.
- Aslı to prioritise the login bug list and pull the top 3 into the sprint.

## Notes

We agreed to defer the multi-region failover discussion until the
infrastructure team has the new VPC peering ready. Tracking this in
JIRA-PLAT-742.
