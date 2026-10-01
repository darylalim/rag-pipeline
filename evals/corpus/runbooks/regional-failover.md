# Runbook: regional failover

Failover from us-east-2 to us-west-2 is decided by the incident commander, never
automatically.

1. Promote the us-west-2 PostgreSQL replica to primary.
2. Run `shipit failover us-west-2` to move traffic.
3. Expect a recovery point objective (RPO) of 5 minutes and a recovery time
   objective (RTO) of 30 minutes.
