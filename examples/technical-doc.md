---
title: Resilient Service Patterns
status: draft
last-reviewed: 2026-03-30
---

# Resilient Service Patterns

This document collects the patterns we use across our services to stay
up under partial failure. It is meant for new engineers ramping up on
how the platform thinks about reliability.

## Authentication

We standardised on stateless authentication using signed JWT tokens.
Tokens carry the subject, the issued-at and expires-at timestamps, and
a list of scopes. Validation happens in the middleware layer of every
service so business code never has to think about identity.

Refresh tokens have a 30-day lifetime and rotate on every use. The
revocation list lives in Redis and is checked on each refresh.

## Retries

Every outbound HTTP call goes through the shared `httpx` client which
implements bounded retries with exponential backoff and full jitter. The
default policy is:

- 3 attempts maximum
- Base delay 100ms, multiplier 2, cap at 5s
- Retry only on idempotent methods (GET, PUT, DELETE) and 5xx / 429

Services with stricter latency budgets can override the policy via
client construction.

## Circuit Breakers

Where retries are insufficient — typically because the downstream is
truly down rather than slow — we wrap the call in a circuit breaker.
The circuit breaker pattern stops sending requests to a struggling
dependency, gives it room to recover, then probes with a single request
before fully reopening.

We use the `sony/gobreaker` library. Failure thresholds are per-service
and live in configuration. The default is "open the circuit after 5
consecutive failures" with a 30-second cool-down.

## Rate Limiting

Per-tenant rate limits live at the gateway. We use a sliding-window
algorithm backed by Redis. Limits are configured per plan tier and
exposed via the `X-RateLimit-*` response headers.

Internal services do not enforce additional rate limits; the gateway is
the single source of truth.

## Observability

Three pillars apply: logs, metrics, traces.

- Logs are JSON-formatted, shipped to OpenSearch, retained 30 days.
- Metrics are Prometheus, scraped every 15s, retained 90 days.
- Traces are OpenTelemetry, sampled at 5% by default, 100% for errors.

Every service exposes a `/health` endpoint distinct from `/ready`.
Health is "the process is running"; readiness is "the process can serve
traffic right now".
