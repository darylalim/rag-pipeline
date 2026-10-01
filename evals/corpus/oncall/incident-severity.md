# Incident severity levels

Every incident gets a severity from SEV1 (worst) to SEV4. When in doubt, choose
the higher severity; it can be lowered later.

## SEV1

A customer-facing outage of tracking or search, or an error rate above 25% on
any customer-facing service. Page immediately. The on-call must acknowledge
within 5 minutes, the status page is updated within 15 minutes, and executives
get an update every hour.

## SEV2

A major feature is degraded but not down. The on-call must acknowledge within
15 minutes.

## SEV3

A minor or internal-only degradation. Handled the next business day.

## SEV4

A cosmetic issue. Goes to the owning team's backlog.

## Incident commanders and postmortems

SEV1 and SEV2 incidents get an incident commander from the IC rotation. Both
require a written postmortem within 5 business days.
