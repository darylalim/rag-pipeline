# Deployment guide (v2, current)

This is the current deployment process. It replaced the v1 process on
3 November 2025.

## Rollout stages

Every production deploy goes through the `shipit` tool in three stages:

1. **Canary:** 5% of traffic for a 30-minute bake.
2. **Partial:** 25% of traffic for 1 hour.
3. **Full:** 100% of traffic.

## Automatic rollback

`shipit` rolls a deploy back automatically if the error rate stays above 2% for
5 minutes during any stage. A rollback pages the deploying team's on-call.

## When to deploy

Deploys run Monday to Thursday, 09:00 to 16:00 Pacific time. There are no
Friday deploys. A deploy needs one approval from the owning team.
