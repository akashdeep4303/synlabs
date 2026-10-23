# Period close

NimbusLedger treats a fiscal period as **open**, **soft-closed**, or **locked**.

## Timing

- Soft-close is allowed starting at period end. The system target is **T+2** (two business days after period end) for a complete close checklist on Growth and Scale.
- Starter has no close checklist; users still lock periods manually.

## Locking

A Controller (or Owner) signs off to lock. After lock:

- New journals in that period are rejected.
- Reversing journals must be dated in an open period.
- Unlock requires a second Controller (four-eyes) on Scale. Growth and Starter allow a single Controller to unlock, which is logged.

## 4-4-5 calendars

A workspace may use a **4-4-5** retail calendar instead of calendar months. 4-4-5 is configured once at workspace creation. You cannot mix 4-4-5 and calendar months in the same entity.

## Flux analysis

Close Copilot drafts variance comments for accounts moving more than 10% and $10,000 versus prior period. Copilot comments are suggestions; they are not posted until a human accepts them.

## Journal line limit

A single journal entry may contain at most **500 lines**. Split large imports across multiple journals. The import CSV template uses columns: date, account, debit, credit, entity, memo, currency.
