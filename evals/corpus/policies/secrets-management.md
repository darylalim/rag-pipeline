# Secrets management

- All secrets live in Vault. Services read them at startup with short-lived
  tokens that expire after 1 hour.
- Production database credentials are rotated every 30 days.
- Carrier API credentials are rotated every 12 months.
- A secret must never be committed to a repository, even in an encrypted file.
