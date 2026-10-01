# Testing standards

- New code needs at least 80% line coverage.
- Every pair of services that exchange stream messages has contract tests.
  This became mandatory after the March 2025 Lantern outage.
- A flaky test is quarantined within 1 business day and fixed or deleted within
  2 weeks.
