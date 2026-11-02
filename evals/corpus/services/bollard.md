# Bollard (ETA prediction service)

Bollard predicts when each shipment will arrive. It recomputes a
shipment's estimated time of arrival whenever its status changes, and
records the ETA drift once the shipment is delivered, so the model can
be checked against what actually happened.

## Ownership

- **Team:** Fleet Data
- **Language:** Python
- **On-call rotation:** Fleet Data

## Interfaces

An internal gRPC API returns the current prediction for a shipment,
with a confidence interval, and the history of earlier predictions.
Predictions are also written onto the shipment's record in the
`freight` database, where other services read them.

## Dependencies

It consumes the `shipments.events` stream for status changes and reads
lane history from the `freight` database. The prediction model is
retrained weekly, outside the service, and loaded at startup.

## Runtime limits

These limits apply to every request the service accepts on its internal API.
They are enforced in the service itself rather than at a gateway, so a caller
inside the platform meets the same limits as any other.

- **Request timeout:** 1,500 milliseconds. A request still running then is cancelled
  and answered with HTTP 504, and its partial work is rolled back.
- **Request size:** at most 256 KB per request body. A larger body is refused
  with HTTP 413 before it is read.
- **Concurrency:** 64 requests in flight per replica. Requests beyond
  that wait in a queue of the same length, and are refused with HTTP 503 once
  the queue is full.
- **Per-caller rate limit:** 200 requests per second for each calling
  service, identified by its service token. A caller over the limit receives
  HTTP 429.

Raising any of these needs a change reviewed by the owning team, because each
one is sized against the database connection pool described under Storage.

## Retries and failure handling

When a call the service makes to one of its dependencies fails, it retries
before giving up. Only connection errors, timeouts and 5xx responses are
retried; a 4xx response is treated as final.

- **Attempts:** 4 in total, including the first.
- **Backoff:** exponential, starting at 100 milliseconds and capped at
  20 seconds between attempts, with full jitter.
- **Dead letters:** work that has failed every attempt is moved to a
  dead-letter table, where it is kept for 6 hours so the on-call can
  inspect and replay it. After that it is dropped and counted in the
  `dead_letter_expired` metric.

Every operation is idempotent, keyed on the request's `Idempotency-Key` header,
so a retried request is never applied twice.

## Storage

The service keeps its state in its own schema of the `freight` PostgreSQL
database. No other service reads that schema directly; they go through the
internal API.

- **Record retention:** records are kept for 400 days, then deleted by a
  purge that runs inside the service every night.
- **Backups:** a snapshot every 6 hours, kept for 35 days like every database
  backup.
- **Restore:** restoring from the latest snapshot is expected to take about
  45 minutes. That is the recovery time objective for this schema.

A restore is the on-call's decision, made together with the owning team's
engineering manager, and it loses every write made since the snapshot.

## Scaling

The service runs as a stateless deployment behind the internal load balancer in
both regions. The standby region runs the minimum replica count until a
failover.

- **Replicas:** at least 3 and at most 12 per region.
- **Autoscaling:** a replica is added when average CPU stays above 65% for
  3 minutes.
- **Scale-down:** replicas are removed one at a time, no more often than once
  every 10 minutes.

If the maximum is reached and latency is still rising, the on-call may raise it
by hand for up to 24 hours, and must open a ticket for the owning team.

## Alerts

Alerts are sent by Klaxon, and each one links to this page.

- **Page:** when more than 3% of requests fail with a 5xx response for
  10 minutes in a row.
- **Ticket:** when p99 latency stays above 900 milliseconds for 30 minutes. Tickets go to
  the owning team's queue, not to the on-call.
- **Disk:** a page when the schema's disk usage reaches 80% of its
  allocation.

An alert silenced during an incident must be listed in the on-call handoff
note.

## Service level objective

The SLO is 99.5% monthly availability, measured as the share of internal API
requests answered without a 5xx response. Requests refused with 429 or 413
count as successes, since the service behaved as designed. The error budget is
reviewed at the owning team's weekly operations meeting.

## Maintenance

Routine maintenance (schema migrations, index rebuilds and minor upgrades of
dependencies) happens in a weekly window: Tuesdays, 14:00 to 15:00 UTC. Maintenance outside
that window needs the engineering manager's approval. During the window the
service may refuse writes for up to 5 minutes, and callers are expected to
retry.
