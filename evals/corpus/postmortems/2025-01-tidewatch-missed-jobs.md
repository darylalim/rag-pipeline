# Postmortem: Tidewatch missed nightly jobs, 22 January 2025

**Severity:** SEV3
**Duration:** about 9 hours, overnight

## What happened

After a node replacement, Tidewatch's leader election stalled and no instance
ran scheduled jobs. Three nightly jobs were missed, including a data-retention
purge and Lantern's index rebuild. Search served a day-old index until morning.

## Action items

- Alert when no Tidewatch job has started for 30 minutes.
