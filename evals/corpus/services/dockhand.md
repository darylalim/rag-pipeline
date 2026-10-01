# Dockhand (ingest service)

Dockhand accepts carrier events over HTTPS and publishes them to `shipments.raw`.

## Ownership

- **Team:** Pipeline
- **Language:** Go
- **On-call rotation:** Pipeline

## Runtime limits

- **Request timeout:** 30 seconds per carrier request.
- **Retries:** 3 attempts when publishing to `shipments.raw`, with exponential
  backoff starting at 200 milliseconds.
- **Payload limit:** 2 MB per request; larger batches must be split by the
  carrier.

## Service level objective

Dockhand's SLO is 99.5% monthly availability, measured as the share of carrier
requests answered with a non-5xx status.

## Notes

Dockhand rejects a payload without a `carrier_id` with HTTP 422 and never
retries it. Duplicate events are not removed here: deduplication happens in
Ballast.
