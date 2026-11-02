# Dependency updates

Third-party dependencies are updated in a monthly batch. On the first Monday of
each month, each team opens one pull request per service that updates every
dependency, and merges it once the tests pass.

## Security fixes

A dependency with a published security advisory is updated within 14 days of
the advisory, whatever its severity. The update can go out in the next monthly
batch if that is sooner.

## Major versions

A major version upgrade is never part of a batch. It gets a pull request of its
own, reviewed by the owning team.
