# Deployment guide (v1, deprecated)

Deprecated: this process was superseded by the v2 deployment guide on
3 November 2025 and is kept for reference only.

## Rollout stages

Under v1, deploys used the `deployctl` tool in two stages:

1. **Canary:** 10% of traffic for a 15-minute bake.
2. **Full:** 100% of traffic.

## Rollback

Rollback was manual: the deploying engineer watched the dashboards and ran
`deployctl rollback` if errors rose.

## When to deploy

Under v1, deploys could run on any weekday, including Fridays.
