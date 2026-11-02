# Authentication policy

**Effective:** March 2025. This policy replaces the 2023 password policy.

## Single sign-on

Every internal tool is reached through single sign-on. The second factor is a
passkey or a hardware security key; codes sent by SMS are not accepted.

## Accounts that still have a password

A few accounts cannot use single sign-on: break-glass accounts and some vendor
consoles. Their passwords are generated randomly, stored in Vault like every
other secret, and are at least 14 characters long. Passwords do not expire on a schedule: one is
replaced only when it may have been exposed, and then within 24 hours.

## Lockout

Ten failed sign-in attempts in a row lock an account for 30 minutes.
