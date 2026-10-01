# ADR 0022: HMAC signatures for webhooks

**Date:** June 2024
**Status:** accepted

We sign webhooks with HMAC-SHA256 rather than requiring mutual TLS, because most
shippers' endpoints could not be configured for mutual TLS. Shippers verify the
signature with a shared key that Signalbox rotates.
