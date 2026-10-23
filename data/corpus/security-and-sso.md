# Security, compliance, and SSO

## Certifications

NimbusLedger holds **SOC 2 Type II** (annual) and **ISO 27001**. Pen tests run twice a year. A trust-center packet is available to Growth and Scale customers under NDA.

## Data residency

Primary regions are **us-east-1** and **eu-central-1**. A workspace is pinned at creation and cannot be moved without an export/import. Backups stay in the same region. We do not offer ap-south-1 as of FY2026.

## Encryption

Data at rest is AES-256. TLS 1.2+ in transit. Customer-managed keys (CMK) via AWS KMS or GCP KMS are available on Scale only.

## SSO and provisioning

- SAML 2.0 and OIDC are supported on **Growth and Scale**. Starter is password + TOTP only.
- SCIM 2.0 user provisioning is included on Growth and Scale.
- Just-in-time account creation is on by default for SSO workspaces.
- Session lifetime is 12 hours; idle timeout is 30 minutes.

## Roles

Built-in roles: Owner, Controller, Accountant, Read-only, Auditor. Custom roles are Scale-only. Posting rights cannot be granted to Auditor or Read-only.
