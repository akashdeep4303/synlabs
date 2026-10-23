from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rag.minipdf import write_pdf

CORPUS = ROOT / "data" / "corpus"


def main() -> None:
    write_pdf(
        CORPUS / "backup-and-dr.pdf",
        "NimbusLedger backups and disaster recovery",
        """
Production RPO is 15 minutes. Production RTO is 4 hours.
Backups are encrypted with the same AES-256 keys as live data and stay in the workspace region (us-east-1 or eu-central-1).
Point-in-time restore covers the last 14 days. Restores older than 14 days are full-backup restores and take longer.
We test a regional failover twice a year. Customers are notified at least 14 days before a planned DR test.
Sandbox is excluded from DR: sandbox workspaces are wiped every Sunday 00:00 UTC and are not restored.
A failed restore that exceeds the 4-hour RTO is treated as a P1 incident.
""".strip(),
    )
    write_pdf(
        CORPUS / "journal-limits.pdf",
        "Journal import limits and CSV template",
        """
A single journal entry may contain at most 500 lines. Split larger files.
CSV columns: date, account, debit, credit, entity, memo, currency.
Dates must be ISO-8601. Debit and credit are mutually exclusive per line.
Imports over 10 MB should be split. The API import endpoint shares the invoice API rate limit of 120 requests per minute.
Balanced journals only: total debit must equal total credit in the journal currency before FX translation.
Intercompany journals require both entities to belong to the same workspace.
""".strip(),
    )
    print("wrote PDFs")


if __name__ == "__main__":
    main()
