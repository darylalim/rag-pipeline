# Ballast (enrichment worker)

Ballast turns raw carrier events into enriched shipment records.

## Ownership

- **Team:** Pipeline
- **Language:** Go
- **On-call rotation:** Pipeline

## Processing

Ballast consumes `shipments.raw` with a consumer concurrency of 32, raised from
8 after the July 2025 backlog. It deduplicates events on the key (carrier_id,
tracking_number, event_time), attaches lane and facility data, and publishes to
`shipments.enriched`. When a shipment's status changes it also publishes to
`shipments.events`.

## Service level objective

99% of events are enriched within 2 minutes of reaching `shipments.raw`.
