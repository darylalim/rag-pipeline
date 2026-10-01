# Harbor (shipper dashboard)

Harbor is the web dashboard shippers use to search shipments and manage
notification settings. It calls Lantern for search and Signalbox for
notification preferences.

## Ownership

- **Team:** Web
- **Language:** TypeScript, with React
- **On-call rotation:** Web, business hours only

## Notes

- Harbor is served from a CDN; a deploy invalidates the CDN cache.
- A dashboard session times out after 12 hours.
- Harbor's SLO is 99.5% monthly availability of the dashboard's login page.
