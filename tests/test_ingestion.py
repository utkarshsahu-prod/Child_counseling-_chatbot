import tempfile
import unittest
from pathlib import Path

from openpyxl import Workbook

from src.openly.ingestion import ContractError, SheetContract, ingest_sheet_with_contract


class TestIngestionContracts(unittest.TestCase):
    def _write_workbook(self, rows: list[list[object]]) -> Path:
        tmp = tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False)
        wb = Workbook()
        ws = wb.active
        ws.title = "Severity Escalation"
        for row in rows:
            ws.append(row)
        wb.save(tmp.name)
        return Path(tmp.name)

    def test_ingests_valid_sheet_with_report(self):
        path = self._write_workbook(
            [
                ["Rule ID", "Escalation Trigger", "Escalated Tier"],
                ["R1", "speech_delay", "Tier 1"],
            ]
        )
        records, report = ingest_sheet_with_contract(
            path,
            SheetContract(
                sheet_name="Severity Escalation",
                required_columns=("Rule ID", "Escalation Trigger", "Escalated Tier"),
                required_non_empty_columns=("Rule ID", "Escalation Trigger"),
            ),
        )
        self.assertEqual(len(records), 1)
        self.assertEqual(report.records_parsed, 1)

    def test_fails_on_empty_required_cell(self):
        path = self._write_workbook(
            [
                ["Rule ID", "Escalation Trigger", "Escalated Tier"],
                ["R1", "", "Tier 1"],
            ]
        )
        with self.assertRaises(ContractError):
            ingest_sheet_with_contract(
                path,
                SheetContract(
                    sheet_name="Severity Escalation",
                    required_columns=("Rule ID", "Escalation Trigger", "Escalated Tier"),
                    required_non_empty_columns=("Rule ID", "Escalation Trigger"),
                ),
            )

    def test_fails_on_duplicate_headers(self):
        path = self._write_workbook(
            [
                ["Rule ID", "Rule ID", "Escalated Tier"],
                ["R1", "speech_delay", "Tier 1"],
            ]
        )
        with self.assertRaises(ContractError):
            ingest_sheet_with_contract(
                path,
                SheetContract(
                    sheet_name="Severity Escalation",
                    required_columns=("Rule ID", "Escalated Tier"),
                ),
            )


if __name__ == "__main__":
    unittest.main()
