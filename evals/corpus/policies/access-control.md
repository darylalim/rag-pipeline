# Production access control

## Roles

Engineers hold one of three production roles: viewer, operator or admin.
Viewers can read dashboards and logs; operators can also restart services and
run approved runbooks; admins can change infrastructure.

## Database write access

Write access to the production `freight` database is granted per request. It
needs two approvals, one of them from the Platform team, and it expires
automatically after 8 hours.

## Break-glass access

In a SEV1 the incident commander can grant break-glass admin access without
approvals. Every break-glass session is written to the audit log and must be
reviewed within 1 business day.
