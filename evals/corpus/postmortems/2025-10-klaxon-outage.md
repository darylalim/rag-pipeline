# Postmortem: Klaxon outage, October 2025

**Severity:** SEV2
**Duration:** 47 minutes

## What happened

Klaxon's message provider rejected every send for 47 minutes. A SEV2 page about
Dockhand reached no one for 31 minutes, because the phone tree depended on
managers who were themselves waiting to be paged.

## Action items

- Done: Klaxon now fails over automatically to SMS through a second provider
  after 2 minutes of failed sends. This replaces the phone tree in the Klaxon
  runbook, which is retired.
- Done: a synthetic page is sent every 5 minutes, and its failure alerts the
  Platform team by email.
