# Runbook: Lantern index rebuild failed

**Alert:** `LanternRebuildFailed`

1. Read the Tidewatch job log for the failed `lantern-rebuild` run.
2. Rerun it with `tidewatch run lantern-rebuild`.
3. If it fails a second time, leave the previous index in place. Search stays
   available but can be up to 24 hours stale; treat this as a SEV3 and hand it
   to Discovery in business hours.
