# ADR 0014: upgrade PostgreSQL to 16

**Date:** February 2025
**Status:** done

We upgraded the `freight` database from PostgreSQL 15 to 16 using logical
replication to a new cluster. The cut-over took 4 minutes of write downtime,
inside an announced maintenance window.
