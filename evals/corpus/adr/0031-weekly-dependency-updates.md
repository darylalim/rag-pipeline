# ADR 0031: weekly dependency updates

**Date:** August 2025
**Status:** accepted

The monthly batch let a critical advisory wait up to 14 days, and a batch that
updated everything at once was hard to review when it broke something.

From September 2025, Dependabot opens dependency update pull requests every
week, and the owning team merges them within that week. The monthly batch is
retired. Advisories rated critical or high are patched within 72 hours;
others keep the 14-day limit. Major versions still get pull requests of their
own.
