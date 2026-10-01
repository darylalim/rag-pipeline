# Runbook: JetStream disk filling up

**Alert:** a JetStream volume above 80% full.

1. Find the stream that grew, usually `shipments.raw` during a carrier replay.
2. Expand the volume by 50%; expansion is online and takes a few minutes.
3. Never delete messages from `shipments.raw` by hand: Ballast may not have
   consumed them yet, and raw payloads are the only copy until enrichment.
