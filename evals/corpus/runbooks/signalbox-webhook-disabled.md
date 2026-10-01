# Runbook: a shipper's webhook was disabled

Signalbox disables a webhook endpoint after 50 consecutive failed deliveries.

1. Confirm with the shipper that their endpoint is fixed.
2. Re-enable the endpoint in the Signalbox admin console.
3. Re-enabling replays the last 72 hours of events to the endpoint; older
   events are not replayed.
