# Runbook: Dockhand 5xx spike

**Alert:** Dockhand's 5xx rate above 1% for 10 minutes.

1. Check JetStream health first: most Dockhand 5xx come from failed publishes.
2. If JetStream is healthy, scale Dockhand from 6 to 12 pods.
3. If the errors come from a single carrier, pause it with
   `dockhand carriers pause <carrier_id>` and notify Partnerships.
