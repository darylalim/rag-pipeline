# Mooring (address validation service)

Mooring checks and normalizes the addresses on a shipment: it
standardizes the street, city and postal code, flags an address that
does not exist, and adds coordinates for the delivery point.

## Ownership

- **Team:** Fleet Data
- **Language:** Rust
- **On-call rotation:** Fleet Data

## Interfaces

An internal gRPC API validates one address or a batch of up to 50.
Each answer says whether the address was confirmed, corrected or
rejected, with the corrected form when there is one.

## Dependencies

It answers from postal reference data that it loads at startup and
refreshes every week; it makes no outside call for a validation.

## Runtime limits

These limits apply to every request the service accepts on its internal API.
They are enforced in the service itself rather than at a gateway, so a caller
inside the platform meets the same limits as any other.

- **Request timeout:** 600 milliseconds. A request still running then is cancelled
  and answered with HTTP 504, and its partial work is rolled back.
- **Request size:** at most 96 KB per request body. A larger body is refused
  with HTTP 413 before it is read.
- **Concurrency:** 96 requests in flight per replica. Requests beyond
  that wait in a queue of the same length, and are refused with HTTP 503 once
  the queue is full.
- **Per-caller rate limit:** 400 requests per second for each calling
  service, identified by its service token. A caller over the limit receives
  HTTP 429.

Raising any of these needs a change reviewed by the owning team, because each
one is sized against the database connection pool described under Storage.

## Storage

The service keeps its state in its own schema of the `freight` PostgreSQL
database. No other service reads that schema directly; they go through the
internal API.

- **Record retention:** records are kept for 60 days, then deleted by a
  purge that runs inside the service every night.
- **Backups:** a snapshot every 3 hours, kept for 35 days like every database
  backup.
- **Restore:** restoring from the latest snapshot is expected to take about
  25 minutes. That is the recovery time objective for this schema.

A restore is the on-call's decision, made together with the owning team's
engineering manager, and it loses every write made since the snapshot.

## Scaling

The service runs as a stateless deployment behind the internal load balancer in
both regions. The standby region runs the minimum replica count until a
failover.

- **Replicas:** at least 3 and at most 9 per region.
- **Autoscaling:** a replica is added when average CPU stays above 45% for
  3 minutes.
- **Scale-down:** replicas are removed one at a time, no more often than once
  every 12 minutes.

If the maximum is reached and latency is still rising, the on-call may raise it
by hand for up to 24 hours, and must open a ticket for the owning team.

## Service level objective

The SLO is 99.7% monthly availability, measured as the share of internal API
requests answered without a 5xx response. Requests refused with 429 or 413
count as successes, since the service behaved as designed. The error budget is
reviewed at the owning team's weekly operations meeting.
