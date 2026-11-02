# Customer plans

Tallowmere sells four plans: Starter, Growth, Business and Enterprise. Every
plan tracks shipments across all supported carriers and includes the shipper
dashboard and the public API; the plans differ in volume, retention, support and
a few features that larger shippers need.

## Choosing a plan

Starter is meant for a shipper trying Tallowmere out or moving a few hundred
shipments a month. Growth suits a single warehouse or a small e-commerce
business. Business is for shippers with several sites or their own engineering
team, who rely on webhooks and exports. Enterprise adds contractual terms,
single sign-on for every user and the highest limits, and is sold on annual
contracts only.

## Trials

A new account can try the Growth plan free for 14 days. At the end of the
trial the account moves to Starter unless a paid plan has been chosen; no data
is deleted, but anything over Starter's limits becomes read-only.

## Changing plans

An upgrade takes effect immediately, and the new limits apply at once. A
downgrade takes effect at the end of the current monthly cycle. If the account
is then over a limit of the smaller plan, the oldest items over the limit
become read-only: for example, webhook endpoints beyond the plan's number are
paused rather than deleted.

## Going over a limit

Tracked shipments are counted per calendar month. A Starter account that
reaches its monthly shipment limit cannot add shipments until the next month.
On every other plan, shipments over the limit are tracked as usual and charged
per additional 1,000 shipments. The other limits are hard limits on every plan:
an account cannot add a user, a webhook endpoint or a sandbox beyond its plan's
number.

## Plan limits

| Limit | Starter | Growth | Business | Enterprise |
| --- | --- | --- | --- | --- |
| Monthly price | $49 | $299 | $1,200 | By contract |
| Tracked shipments per month | 1,000 | 25,000 | 250,000 | Unlimited |
| Users per account | 3 | 15 | 75 | Unlimited |
| Public API rate limit | 600 requests per minute | 600 requests per minute | 600 requests per minute | 600 requests per minute, up to 3,000 on request |
| Support | Email, business hours | Email, business hours | Email and phone; 24/7 for P1 | Email, phone and chat; 24/7 |
| Notification channels | Email | Email and SMS | Email, SMS and webhook | Email, SMS and webhook |
| Data export | None | Monthly CSV | Daily CSV | Hourly CSV and export API |
| Shipment history in the dashboard | 90 days | 1 year | 2 years | 2 years |
| Single sign-on | No | No | Yes | Yes |
| Branded notification emails | No | No | Yes | Yes |
| Carrier integrations | 5 | 25 | Unlimited | Unlimited |
| Webhook endpoints | 1 | 5 | 25 | 100 |
| Document storage | 1 GB | 25 GB | 250 GB | 2 TB |
| Sandbox environments | None | 1 | 3 | 10 |

## Notes

Shipment history beyond the dashboard's window is not lost: it can still be
exported on any plan that includes exports, within the data retention policy.
Support response times for each plan are in the support escalation matrix.
