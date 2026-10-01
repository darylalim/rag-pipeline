# Runbook: rotate the webhook signing key

Signalbox's webhook signing key is rotated every 90 days.

1. Generate the new key in Vault.
2. Publish it to shippers 7 days before the old key expires.
3. During those 7 days Signalbox signs with the new key, and shippers accept
   either key. The old key is then revoked.
