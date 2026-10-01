# Tidewatch (scheduler)

Tidewatch runs Tallowmere's recurring jobs: Lantern's nightly index rebuild,
the data-retention purges, and the weekly carrier reconciliation report.

## Ownership

- **Team:** Platform
- **Language:** Python
- **On-call rotation:** Platform

## Runtime limits

- **Maximum job runtime:** 45 minutes; a job still running then is killed and
  counted as failed.
- **Retries:** 2 retries for a failed job.
- **Time zone:** every schedule is written in UTC.

## Service level objective

Tidewatch's SLO is 99.5% of scheduled jobs starting within 5 minutes of their
scheduled time.

## Notes

The carrier reconciliation report runs every Monday at 06:00 UTC and is sent to
the Pipeline team.
