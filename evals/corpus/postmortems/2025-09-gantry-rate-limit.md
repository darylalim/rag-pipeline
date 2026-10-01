# Postmortem: Gantry rate-limit misconfiguration, 4 September 2025

**Severity:** SEV3
**Duration:** 25 minutes

## What happened

A configuration change set Gantry's per-key rate limit to 60 requests per
minute instead of 600. About 4% of public API requests received HTTP 429 until
the change was reverted.

## Action items

- Validate rate-limit changes against a minimum of 300 requests per minute.
