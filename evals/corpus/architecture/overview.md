# Tallowmere platform architecture

Tallowmere Systems runs a freight-tracking platform: carriers send shipment
events, and shippers search them and receive notifications. This page is the
map of how the services fit together; each service has its own page.

## Event flow

Carrier events arrive at **Dockhand**, the ingest service, which validates each
payload and publishes it to the `shipments.raw` stream. **Ballast**, the
enrichment worker owned by the Pipeline team, consumes `shipments.raw`, attaches
lane and facility data, and publishes the result to `shipments.enriched`.

**Lantern**, the search service, consumes `shipments.enriched` and keeps its own
search index. **Signalbox**, the notification service, subscribes to the
`shipments.events` stream, which Ballast also writes whenever a shipment changes
status. **Tidewatch**, the scheduler, runs the platform's recurring jobs.

## Infrastructure

All streams run on NATS JetStream. The system of record is PostgreSQL 16, in the
`freight` database; Lantern's index is built with Tantivy and is not a system of
record, so it can always be rebuilt from `shipments.enriched`. Production runs in
two regions: us-east-2 is primary and us-west-2 is the warm standby. A regional
failover is a manual decision made by the incident commander.
