# Data retention and deletion

NimbusLedger retention is driven by accounting controls, not by storage cost.

## Posted data

- Posted journals, posted invoices, and closed-period trial balances are retained for **7 years** from the posting date.
- After 7 years they are moved to cold object storage for 90 days and then hard-deleted unless a legal hold is present.

## Drafts and sandbox

- Draft invoices and unposted journals are retained for **90 days** of inactivity, then purged.
- Sandbox workspaces are wiped every **Sunday 00:00 UTC**. Sandbox data is never copied into production.

## Audit logs

Authentication, permission changes, and posting events are retained for **18 months**. Export jobs appear in the audit log but the exported file itself is not retained by NimbusLedger.

## Backups

Production RPO is **15 minutes**. Production RTO is **4 hours**. Backups follow the same 7-year clock as posted journals. Point-in-time restore is available for the last 14 days.

## Customer-initiated deletion

A workspace owner may request erasure. Posted journals under a legal hold cannot be erased until the hold is released. GDPR erasure of *personal* fields (user names, emails) is honored within 30 days even when the journal row itself is retained; those fields are replaced with a tombstone id.
