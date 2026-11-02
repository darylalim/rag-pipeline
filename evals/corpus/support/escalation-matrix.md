# Support escalation matrix

How quickly Support answers a customer's ticket, how often it updates the
customer, and when it brings in engineering, by the customer's plan and the
ticket's priority.

## Ticket priorities

Support sets a ticket's priority when it first reads it, and may change it
later:

- **P1:** the customer cannot track any shipment, or cannot reach the dashboard
  or the API at all.
- **P2:** a major feature is impaired for the customer, such as notifications
  or exports, and there is no workaround.
- **P3:** a minor problem, or a major one with a workaround.
- **P4:** a question, or a request for a feature.

If several customers open P1 tickets about the same problem, Support declares
an incident, and the incident severity levels apply from then on. The tickets
stay open and are answered from the incident's updates.

## Business hours

Business hours are 09:00 to 18:00 Pacific time, Monday to Friday, excluding
company holidays. A time given in business hours or business days pauses
outside them; a time given in hours or minutes does not.

## Escalating to engineering

When a ticket reaches its escalation time without a fix, Support opens an
engineering ticket for the owning team and links the two. For a P1, Support
also pages the owning team's on-call through Klaxon. The engineering team
answers on the engineering ticket, and Support relays the answer to the
customer; engineers do not write to customers directly.

## Response times and escalation by plan

| Plan | Priority | First response | Customer updates | Escalated to engineering after | Also notified |
| --- | --- | --- | --- | --- | --- |
| Starter | P1 | 4 business hours | Every business day | 1 business day | Nobody |
| Starter | P2 | 8 business hours | Every 2 business days | 3 business days | Nobody |
| Starter | P3 | 2 business days | Weekly | 10 business days | Nobody |
| Starter | P4 | 5 business days | None | Never | Nobody |
| Growth | P1 | 2 hours | Every 4 hours | 4 hours | Account manager |
| Growth | P2 | 4 business hours | Every business day | 2 business days | Nobody |
| Growth | P3 | 1 business day | Every 3 business days | 7 business days | Nobody |
| Growth | P4 | 3 business days | None | Never | Nobody |
| Business | P1 | 1 hour | Every 2 hours | 2 hours | Account manager |
| Business | P2 | 2 hours | Every 8 hours | 1 business day | Account manager |
| Business | P3 | 8 business hours | Every 2 business days | 5 business days | Nobody |
| Business | P4 | 2 business days | Weekly | Never | Nobody |
| Enterprise | P1 | 15 minutes | Every 30 minutes | Immediately | Account manager and Support director |
| Enterprise | P2 | 1 hour | Every 4 hours | 4 hours | Account manager |
| Enterprise | P3 | 4 business hours | Every business day | 3 business days | Nobody |
| Enterprise | P4 | 1 business day | Weekly | 15 business days | Nobody |

A customer can ask for a ticket's priority to be raised; Support raises it if
the ticket meets the higher priority's definition.
