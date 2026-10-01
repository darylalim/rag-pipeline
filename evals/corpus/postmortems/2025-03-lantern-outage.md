# Postmortem: Lantern search outage, 18 March 2025

**Severity:** SEV1
**Duration:** 47 minutes, 14:12 to 14:59 UTC

## What happened

A schema migration in Ballast removed the `lane_code` field from enriched
shipment records. Lantern's indexer required that field, and every query that
touched a new record panicked. Search was unavailable for all shippers.

## Detection

A synthetic search probe failed at 14:12 UTC and paged the Discovery on-call.

## Resolution

The Ballast migration was rolled back at 14:51 UTC, and Lantern recovered by
14:59 UTC.

## Action items

- Add contract tests between Ballast's output and Lantern's indexer.
- Alert on Lantern's panic rate rather than waiting for the synthetic probe.
