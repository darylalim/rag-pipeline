# ADR 0011: Tantivy for search

**Date:** September 2023
**Status:** accepted

We chose Tantivy, an embedded Rust search library, over running an Elasticsearch
cluster for Lantern. It gives lower query latency and no separate cluster to
run. The cost is that the index lives with Lantern and must be rebuilt from
`shipments.enriched` when it is lost, which is why the nightly rebuild exists.
