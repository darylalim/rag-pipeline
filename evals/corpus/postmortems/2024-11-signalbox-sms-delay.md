# Postmortem: Signalbox SMS delays, 12 November 2024

**Severity:** SEV3
**Duration:** 2 hours 5 minutes

## What happened

Signalbox's SMS provider throttled Tallowmere's sending account during a
carrier-wide delay, and SMS notifications arrived up to 40 minutes late. Email
and webhook notifications were not affected.

## Action items

- Add a second SMS provider and fail over to it when sends are throttled.
