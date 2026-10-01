# Postmortem: Dockhand ingest backlog, 9 July 2025

**Severity:** SEV2
**Duration:** 3 hours 20 minutes

## What happened

A carrier partner re-sent a full day's batch of events twice. The
`shipments.raw` stream grew to a peak backlog of 1.8 million messages, because
Ballast consumed it with a concurrency limit of 8. Shipment updates reached
shippers up to three hours late. No data was lost.

## Resolution

Ballast's consumer concurrency was raised from 8 to 32, and the backlog drained.

## Action items

- Deduplicate events in Ballast on the key (carrier_id, tracking_number,
  event_time).
- Alert when the `shipments.raw` backlog exceeds 100,000 messages.
