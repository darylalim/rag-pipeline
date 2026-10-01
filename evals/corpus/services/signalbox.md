# Signalbox (notification service)

Signalbox tells shippers when their shipments change status.

## Ownership

- **Team:** Customer Comms
- **Language:** Kotlin
- **On-call rotation:** Customer Comms

## Runtime limits

- **Request timeout:** 10 seconds per outbound delivery.
- **Retries:** 5 attempts, with backoff capped at 60 seconds between attempts.
- **Channels:** email, SMS and webhook.

## Service level objective

Signalbox's SLO is 99.0% monthly availability, measured on accepted delivery
requests.

## Webhooks

Every webhook is signed with HMAC-SHA256, and the signing key is rotated every
90 days. A webhook endpoint that fails 50 deliveries in a row is disabled, and
the shipper is emailed.
