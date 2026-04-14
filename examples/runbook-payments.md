# Runbook — Payments Service

Owner: Burak. On-call: see PagerDuty rotation `payments-oncall`.

## Service overview

The payments service brokers between our checkout flow and Stripe. It
owns the idempotency layer, the retry queue for failed charges, and
the webhook receiver for asynchronous Stripe events.

Stack: Go 1.24, PostgreSQL 16, Redis 7, Stripe Go SDK.

## Common alerts

### `PaymentChargeFailureRateHigh`

Charge failure rate exceeded 2% over a 5-minute window.

Likely causes (in order of frequency):

1. Stripe is having an incident. Check https://status.stripe.com.
2. A specific issuer is rejecting our cards. Run the issuer breakdown
   query in `payments-debug.sql` and look for spikes by `issuer_country`.
3. Our retry queue is backed up. See `PaymentsRetryQueueDepth` below.

If Stripe is up and there's no issuer-specific spike, page the team
lead and start drafting an incident summary.

### `PaymentsRetryQueueDepth`

The retry queue has more than 1000 pending charges.

This usually means the worker pods are unhealthy or paused. Check:

```bash
kubectl get pods -n payments -l app=retry-worker
```

Restart unhealthy pods. If the queue is climbing despite healthy
workers, check the per-batch latency — Stripe might be rate-limiting us.

### `PaymentsWebhookSignatureFailures`

We're rejecting Stripe webhooks because signature verification fails.

This is almost always a secret rotation issue. Check that the current
webhook signing secret in Vault matches what's configured in the Stripe
dashboard. If they differ, restore the older secret and figure out what
changed before rotating again.

## Manual interventions

### Replay a webhook

```bash
payments-cli webhook replay --event-id evt_xxx
```

This re-fetches the event from Stripe and runs it through our processor.
Idempotent.

### Force-close a stuck charge

If a charge is stuck in `requires_action` for more than 24 hours and
the customer has reached out, you can force-close it:

```bash
payments-cli charge close --charge-id ch_xxx --reason customer-request
```

This records the close in our audit log and notifies Stripe to abandon
the charge.

## Recent post-mortems

- 2026-03-22 incident: rate limiter exhaustion. See
  `incident-2026-03-22.md`.
