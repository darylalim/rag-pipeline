# ADR 0007: NATS JetStream for event streams

**Date:** April 2023
**Status:** accepted

We chose NATS JetStream over Apache Kafka for all event streams. With three SREs
we could not operate a Kafka cluster well, and JetStream's operational
footprint is much smaller. Each stream sets a maximum message age, which is
how the 30-day retention of `shipments.raw` is enforced.
