# Gantry (API gateway)

Gantry fronts every public API call: it authenticates API keys, applies rate
limits and routes requests to Lantern and Dockhand.

## Ownership

- **Team:** Platform
- **Language:** Go
- **On-call rotation:** Platform

## Runtime limits

- **Request timeout:** 5 seconds at the gateway.
- **Rate limit:** 600 requests per minute per API key, with a burst of 100.
- **API keys:** expire after 365 days.

## Service level objective

Gantry's SLO is 99.95% monthly availability.
