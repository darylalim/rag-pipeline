# Public API rate limits

- Each API key may make 600 requests per minute, with a burst of 100.
- A request over the limit receives HTTP 429 with a `Retry-After` header.
- Enterprise customers can request a limit of up to 3,000 requests per minute.
