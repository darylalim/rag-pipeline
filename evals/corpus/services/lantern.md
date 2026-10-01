# Lantern (search service)

Lantern answers shipment searches for the shipper dashboard and the public API.

## Ownership

- **Team:** Discovery
- **Language:** Rust
- **On-call rotation:** Discovery

## Runtime limits

- **Request timeout:** 800 milliseconds, with a p99 latency target of 250
  milliseconds.
- **Retries:** 1 retry, for idempotent reads only.
- **Page size:** at most 100 results per query.

## Service level objective

Lantern's SLO is 99.9% monthly availability.

## Index

Lantern consumes `shipments.enriched` continuously, and its index is also
rebuilt from scratch every night at 02:00 UTC by a Tidewatch job. A rebuild
takes about 40 minutes; queries keep using the previous index until the new one
is swapped in.
