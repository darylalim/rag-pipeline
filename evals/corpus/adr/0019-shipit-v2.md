# ADR 0019: staged rollouts with shipit v2

**Date:** October 2025
**Status:** accepted

Three 2025 incidents ran longer than they needed to because v1 rollbacks were
manual. shipit v2 adds a partial stage between canary and full rollout, and
rolls back automatically on a sustained error rate. It replaced v1 on
3 November 2025.
