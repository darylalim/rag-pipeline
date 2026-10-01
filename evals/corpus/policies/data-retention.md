# Data retention policy

How long Tallowmere keeps each kind of data. Tidewatch runs the purge jobs that
enforce these limits.

| Data | Kept for |
| --- | --- |
| Raw carrier payloads (`shipments.raw`) | 30 days |
| Enriched shipment records | 2 years |
| Notification delivery logs | 180 days |
| Audit logs | 7 years |
| Database backups | 35 days |

## Deletion requests

A customer's request to delete their data is completed within 30 days,
including removal from backups as they expire.
